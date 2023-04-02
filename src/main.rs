use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Stream;
use rubato::{InterpolationParameters, InterpolationType, Resampler, SincFixedIn, WindowFunction};
use std::env::args;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::time::{Duration, Instant};
use whisper_rs::{convert_stereo_to_mono_audio, FullParams, SamplingStrategy, WhisperContext};

const VOLUME_THRESHOLD: f32 = 0.05;
const SILENCE_DURATION: Duration = Duration::from_secs(2);
const AUDIO_BUFFER: usize = 512;
const INPUT_SAMPLE_RATE: usize = 44_100;
const OUTPUT_SAMPLE_RATE: usize = 16_000;

fn audio_input_stream_data_callback(
    raw_stereo_samples: &[f32],
    tx: &SyncSender<f32>,
    resampler: &mut SincFixedIn<f32>,
) {
    // Convert stereo to mono
    let raw_mono_samples: Vec<f32> = convert_stereo_to_mono_audio(raw_stereo_samples).unwrap();

    // Resample the audio to get the target sample rate
    let mut mono_samples = resampler.process(&[raw_mono_samples], None).unwrap();

    // Send the audio to the main thread
    mono_samples.pop().unwrap().into_iter().for_each(|sample| {
        tx.send(sample)
            .expect("Failed to send audio sample to main thread");
    });
}

fn create_paused_audio_stream(tx: SyncSender<f32>) -> Stream {
    // Get the default host and input device
    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .expect("Failed to get default input device");
    println!("Default input device: {:?}", input_device.name());

    // Configure the input stream with default format
    // We want to use the default format
    let input_config = input_device
        .supported_input_configs()
        .expect("No supported input config found")
        .next()
        .expect("No supported input config found")
        .with_max_sample_rate()
        .into();
    println!("Input config: {:?}", input_config);

    // Create resampler to convert the audio from the input device's sample rate to 16 kHz
    let mut mono_resampler = SincFixedIn::<f32>::new(
        OUTPUT_SAMPLE_RATE as f64 / INPUT_SAMPLE_RATE as f64,
        2.0,
        InterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        },
        AUDIO_BUFFER,
        1,
    )
    .unwrap();

    // Build and play the input stream
    let stream = input_device
        .build_input_stream(
            &input_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                audio_input_stream_data_callback(data, &tx, &mut mono_resampler);
            },
            move |err| eprintln!("An error occurred on the input audio stream: {}", err),
            None,
        )
        .expect("Failed to build input stream");

    // Initialise with a paused stream
    stream.pause().expect("Failed to pause stream");

    stream
}

fn run_voice_activity_detection(rx: &Receiver<f32>, audio_data: &mut Vec<f32>) {
    // Record until silence is detected
    let mut last_voice_activity = Instant::now();
    while last_voice_activity.elapsed() < SILENCE_DURATION {
        if let Ok(sample) = rx.try_recv() {
            // Check for voice activity
            if sample.abs() > VOLUME_THRESHOLD {
                last_voice_activity = Instant::now();
            }

            // Add the sample to the audio_data buffer
            audio_data.push(sample);
        }
    }
}

fn main() {
    // CLI arguments
    let args: Vec<String> = args().collect();
    if args.len() != 2 {
        println!("Usage: whisper-agent <path_to_model>");
        return;
    }
    let path_to_model = &args[1];

    // Create a buffer to store the recorded audio
    let mut audio_data = Vec::new();

    // Load a context and model
    let mut ctx = WhisperContext::new(path_to_model).expect("failed to load model");

    // Channel for sending recorded data from the input callback to the main thread
    let (tx, rx) = mpsc::sync_channel(AUDIO_BUFFER);

    // Create an audio stream
    let stream = create_paused_audio_stream(tx);

    loop {
        // Start recording
        println!("Start recording");
        stream.play().expect("Failed to start recording");

        // Get the audio data from the input stream and run voice activity detection
        run_voice_activity_detection(&rx, &mut audio_data);

        // Pause the stream
        stream.pause().expect("Failed to pause stream");

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
        params.set_n_threads(1);
        params.set_translate(true);
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // Run the Whisper ASR model
        println!("Run ASR model");
        ctx.full(params, &audio_data[..])
            .expect("failed to run model");

        // Clear the audio data
        audio_data.clear();

        // Fetch the results
        let num_segments = ctx.full_n_segments();
        for i in 0..num_segments {
            let segment = ctx.full_get_segment_text(i).expect("failed to get segment");
            println!("{}", segment);
        }
    }
}

fn _empty() {
    // Create a WavWriter to save the recorded audio
    // let spec = WavSpec {
    //     channels: input_config.channels as _,
    //     sample_rate: input_config.sample_rate.0,
    //     bits_per_sample: 16,
    //     sample_format: hound::SampleFormat::Int,
    // };

    // let file = File::create("recorded_audio.wav").unwrap();
    // let writer = BufWriter::new(file);
    // let mut wav_writer = WavWriter::new(writer, spec).unwrap();

    // Record for 10 seconds
    // let record_duration = Duration::from_secs(10);
    // let start_time = std::time::Instant::now();
    // while start_time.elapsed() < record_duration {
    //     if let Ok(sample) = rx.try_recv() {
    //         wav_writer.write_sample(sample).unwrap();
    //     }
    // }

    // Finalize the WAV file
    // wav_writer.finalize().unwrap();
    // println!("Recording saved to 'recorded_audio.wav'");
}

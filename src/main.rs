use std::{
    env, process,
    sync::mpsc::{self, Receiver, SyncSender},
    time::{Duration, Instant},
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Stream,
};
use rubato::{InterpolationParameters, InterpolationType, Resampler, SincFixedIn, WindowFunction};
use whisper_rs::{convert_stereo_to_mono_audio, FullParams, SamplingStrategy, WhisperContext};

const VOLUME_THRESHOLD: f32 = 0.05;
const SILENCE_DURATION: Duration = Duration::from_secs(2);
const AUDIO_BUFFER: usize = 512;
const INPUT_SAMPLE_RATE: usize = 44_100;
const OUTPUT_SAMPLE_RATE: usize = 16_000;

struct Stt {
    ctx: WhisperContext,
    audio_data: Vec<f32>,
    audio_receiver: Receiver<f32>,
    stream: Stream,
}

fn audio_input_stream_data_callback(
    raw_stereo_samples: &[f32],
    tx: &SyncSender<f32>,
    resampler: &mut SincFixedIn<f32>,
) {
    // Convert stereo to mono
    let raw_mono_samples: Vec<f32> = convert_stereo_to_mono_audio(raw_stereo_samples).unwrap();

    // Resample the audio to get the target sample rate
    // TODO: Fix 'Wrong number of frames X in input channel 0, expected Y'
    let mut mono_samples = resampler
        .process(&[raw_mono_samples], None)
        .expect("failed to resample");

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

impl Stt {
    pub fn new(path_to_model: String) -> Self {
        let ctx = WhisperContext::new(&path_to_model).expect("failed to load model");

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
        params.set_n_threads(1);
        params.set_translate(true);
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        let (tx, audio_receiver) = mpsc::sync_channel(AUDIO_BUFFER);

        // Create an audio stream
        let stream = create_paused_audio_stream(tx);

        Self {
            ctx,
            audio_data: Vec::new(),
            audio_receiver,
            stream,
        }
    }

    /// Record until no voice activity is detected, then output the text.
    pub fn record(&mut self) -> String {
        // Start recording
        println!("Start recording");
        self.stream.play().expect("Failed to start recording");

        // Get the audio data from the input stream and run voice activity detection
        self.run_voice_activity_detection();

        // Pause the stream
        self.stream.pause().expect("Failed to pause stream");

        // Not sure how we store this value somewhere in the struct
        // without having to initialise it every time
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
        self.ctx
            .full(params, &self.audio_data[..])
            .expect("failed to run model");

        // Clear the audio data
        self.audio_data.clear();

        // Fetch the results
        let num_segments = self.ctx.full_n_segments();

        (0..num_segments)
            .map(|i| {
                self.ctx
                    .full_get_segment_text(i)
                    .expect("failed to get segment")
                    .trim()
                    .to_string()
            })
            .filter(|segment| segment != "[BLANK_AUDIO]")
            .collect::<Vec<String>>()
            .join("")
    }

    /// Simple voice activity detection using silence duration.
    ///
    /// Note that this function will block the main thread,
    /// while the audio data is being processed concurrently
    /// through the audio input stream
    fn run_voice_activity_detection(&mut self) {
        let mut last_voice_activity = Instant::now();
        while last_voice_activity.elapsed() < SILENCE_DURATION {
            if let Ok(sample) = self.audio_receiver.try_recv() {
                // Check for voice activity
                if sample.abs() > VOLUME_THRESHOLD {
                    last_voice_activity = Instant::now();
                }

                // Add the sample to the audio_data buffer
                self.audio_data.push(sample);
            }
        }
    }
}

fn main() {
    let Some(model) = env::args().nth(1) else {
        println!("Please provide a path to the model file");
        process::exit(1);
    };

    let mut stt = Stt::new(model);
    let text = stt.record();
    println!("{}", text);
}

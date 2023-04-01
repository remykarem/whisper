use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{WavSpec, WavWriter};
use std::fs::File;
use std::io::BufWriter;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

const VOLUME_THRESHOLD: f32 = 0.05;
const SILENCE_DURATION: Duration = Duration::from_secs(2);

fn main() {
    // Get the default host and input device
    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .expect("Failed to get default input device");

    // Configure the input stream with default format
    let input_format = input_device
        .default_input_config()
        .expect("Failed to get default input format");
    let input_config: cpal::StreamConfig = input_format.into();

    // Create a WavWriter to save the recorded audio
    // let spec = WavSpec {
    //     channels: input_config.channels as _,
    //     sample_rate: input_config.sample_rate.0,
    //     bits_per_sample: 16,
    //     sample_format: hound::SampleFormat::Int,
    // };

    // Load a context and model
    let mut ctx = WhisperContext::new("ggml-tiny.en.bin").expect("failed to load model");

    // Create a params object
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });

    // let file = File::create("recorded_audio.wav").unwrap();
    // let writer = BufWriter::new(file);
    // let mut wav_writer = WavWriter::new(writer, spec).unwrap();

    // Channel for sending recorded data from the input callback to the main thread
    let (tx, rx) = mpsc::sync_channel(1024);

    // Build and play the input stream
    let stream = input_device
        .build_input_stream(
            &input_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                for sample in data.iter() {
                    // Convert f32 sample to i16
                    let i16_sample = (sample * i16::MAX as f32) as i16;
                    tx.send(i16_sample).unwrap();
                }
            },
            move |err| eprintln!("An error occurred on the input audio stream: {}", err),
            None,
        )
        .expect("Failed to build input stream");

    stream.play().expect("Failed to start recording");

    // Record for 10 seconds
    // let record_duration = Duration::from_secs(10);
    // let start_time = std::time::Instant::now();
    // while start_time.elapsed() < record_duration {
    //     if let Ok(sample) = rx.try_recv() {
    //         wav_writer.write_sample(sample).unwrap();
    //     }
    // }

    // Record until silence is detected
    let mut last_voice_activity = Instant::now();
    let mut audio_data = Vec::new();

    while last_voice_activity.elapsed() < SILENCE_DURATION {
        if let Ok(sample) = rx.try_recv() {
            // wav_writer.write_sample(sample).unwrap();

            // Check for voice activity
            let f32_sample = (sample as f32) / i16::MAX as f32;
            if f32_sample > VOLUME_THRESHOLD {
                last_voice_activity = Instant::now();
            }

            // Add the sample to the audio_data buffer
            audio_data.push(f32_sample);
        }
    }

    let audio_data_mono = whisper_rs::convert_stereo_to_mono_audio(&audio_data).unwrap();

    // Drop the stream to close it
    drop(stream);

    // Now we can run the Whisper ASR model
    println!("Run ASR model");
    ctx.full(params, &audio_data_mono[..])
        .expect("failed to run model");

    // Fetch the results
    let num_segments = ctx.full_n_segments();
    for i in 0..num_segments {
        let segment = ctx.full_get_segment_text(i).expect("failed to get segment");
        let start_timestamp = ctx.full_get_segment_t0(i);
        let end_timestamp = ctx.full_get_segment_t1(i);
        println!("[{} - {}]: {}", start_timestamp,
        end_timestamp, segment);
    }

    // Finalize the WAV file
    // wav_writer.finalize().unwrap();
    // println!("Recording saved to 'recorded_audio.wav'");
}

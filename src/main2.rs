use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{InputCallbackInfo, Sample, SampleFormat, SampleRate, Stream};
use webrtc_vad::Vad;
use std::time::Duration;

#[tokio::main]
async fn main() {
    let vad = Vad::new();

    let mut buf = vec![0.0f32; 1024];
    let input_stream = create_paused_input_stream(&mut buf, &vad).await;
    input_stream.play().unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;
}

async fn create_paused_input_stream(buffer: &mut [f32], vad: &Vad) -> Stream {
    let frame_size = 960; // 20 ms at 48 kHz sample rate

    // Create an event loop and default input device
    let host = cpal::default_host();

    let device = host
        .default_input_device()
        .expect("Failed to get default input device");
    println!("Default input device: {:?}", device.name());

    // Get the format of the input device
    let config = device.default_input_config().unwrap().config();
    println!("{:?}", config);

    let stream = device
        .build_input_stream(
            &config,
            move |samples: &[f32], _: &InputCallbackInfo| {
                // react to stream events and read or write stream data here.
                let num_samples = samples.len() / config.channels as usize;

                // Convert the samples to a vector of i16 samples
                let mut i16_samples = Vec::with_capacity(num_samples);
                for i in 0..num_samples {
                    let sample = samples[i * config.channels as usize];
                    i16_samples.push((sample * i16::MAX as f32) as i16);
                }

                // Process the audio in frames and detect voice activity
                for i in (0..i16_samples.len()).step_by(frame_size) {
                    let end = std::cmp::min(i + frame_size, i16_samples.len());
                    let frame = &i16_samples[i..end];

                    let is_speech = vad.is_voice_segment(frame).unwrap();

                    if is_speech {
                        // Speech detected, do something
                        println!("Speech detected.");
                    } else {
                        // Speech not detected, do something else
                        println!("No speech detected.");
                    }
                }
            },
            move |_err| {
                // react to errors here.
            },
            None, // None=blocking, Some(Duration)=timeout
        )
        .unwrap();

    stream.pause().unwrap();

    stream
}

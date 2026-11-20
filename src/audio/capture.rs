//! Capture from the default input device (spec §8.2.6 step 2).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, SizedSample};

use crate::audio::resample::StreamResampler;
use crate::error::{Error, Result};

fn build<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    mut producer: rtrb::Producer<f32>,
) -> std::result::Result<cpal::Stream, cpal::Error>
where
    T: SizedSample,
    f32: cpal::FromSample<T>,
{
    let channels = config.channels as usize;
    device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            // Real-time thread: mix to mono into the ring buffer; drop samples if it is full.
            for frame in data.chunks(channels) {
                let sum: f32 = frame.iter().map(|s| f32::from_sample(*s)).sum();
                let _ = producer.push(sum / channels as f32);
            }
        },
        |_err| {},
        None,
    )
}

/// Records `seconds` from the default input, mixed to mono and resampled to `rate`.
/// No input device is an audio error (exit 4).
pub fn record(seconds: f32, rate: u32, stop: &Arc<AtomicBool>) -> Result<Vec<f32>> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| Error::audio("no input device"))?;
    let supported = device
        .default_input_config()
        .map_err(|e| Error::audio(format!("no usable input config: {e}")))?;
    let (format, config) = (supported.sample_format(), supported.config());
    let device_rate = config.sample_rate;
    let capacity = (device_rate as f32 * (seconds + 1.0)) as usize;
    let (producer, mut consumer) = rtrb::RingBuffer::<f32>::new(capacity.max(1024));
    let stream = match format {
        SampleFormat::F32 => build::<f32>(&device, config, producer),
        SampleFormat::I16 => build::<i16>(&device, config, producer),
        SampleFormat::U16 => build::<u16>(&device, config, producer),
        SampleFormat::I32 => build::<i32>(&device, config, producer),
        other => {
            return Err(Error::audio(format!(
                "unsupported input sample format {other}"
            )));
        }
    }
    .map_err(|e| Error::audio(format!("cannot open the input stream: {e}")))?;
    stream
        .play()
        .map_err(|e| Error::audio(format!("cannot start recording: {e}")))?;
    let want = (device_rate as f32 * seconds) as usize;
    let mut pcm = Vec::with_capacity(want);
    let deadline = std::time::Instant::now() + Duration::from_secs_f32(seconds + 2.0);
    while pcm.len() < want && !stop.load(Ordering::Relaxed) && std::time::Instant::now() < deadline
    {
        while let Ok(s) = consumer.pop() {
            pcm.push(s);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    drop(stream);
    pcm.truncate(want);
    if device_rate == rate {
        return Ok(pcm);
    }
    let mut out = Vec::with_capacity(pcm.len() * rate as usize / device_rate as usize + 16);
    let mut r = StreamResampler::new(device_rate, rate)?;
    let mut emit = |s: &[f32]| {
        out.extend_from_slice(s);
        Ok(true)
    };
    for block in pcm.chunks(4096) {
        r.process(block, &mut emit)?;
    }
    r.flush(&mut emit)?;
    Ok(out)
}

/// A recording that is exactly silent usually means the terminal lacks microphone permission.
pub fn is_all_zero(pcm: &[f32]) -> bool {
    pcm.iter().all(|s| *s == 0.0)
}

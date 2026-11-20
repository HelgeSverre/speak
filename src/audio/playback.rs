//! Playback through cpal: synth thread → (resampler) → ring buffer → device callback (spec §10.1).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, StreamConfig};

use crate::audio::resample::StreamResampler;
use crate::error::{Error, Result};
use crate::traits::{AudioSink, Flow, PcmWriter, SinkStats};

struct Shared {
    underruns: AtomicU64,
    /// Set by the producer once audio has been queued; underruns are counted from then on.
    started: AtomicBool,
    /// Set by the producer when no more audio will come.
    draining: AtomicBool,
    latency_us: AtomicU64,
    /// Frames handed to the device so far.
    played: AtomicU64,
    stop: Arc<AtomicBool>,
}

pub struct Playback {
    stream: Option<cpal::Stream>,
    producer: rtrb::Producer<f32>,
    shared: Arc<Shared>,
    device_rate: u32,
    resampler: Option<StreamResampler>,
    stats: SinkStats,
    scratch: Vec<f32>,
}

/// One output device for `-a '?'`.
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub is_default: bool,
}

fn audio_err(what: &str, e: impl std::fmt::Display) -> Error {
    Error::audio(format!("{what}: {e}. Run speak -a '?' to list devices"))
}

fn device_name(d: &cpal::Device) -> String {
    d.description()
        .map(|x| x.name().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

pub fn list_devices() -> Result<Vec<DeviceInfo>> {
    let host = cpal::default_host();
    let default = host.default_output_device().map(|d| device_name(&d));
    let devices = host
        .output_devices()
        .map_err(|e| audio_err("cannot list output devices", e))?;
    Ok(devices
        .enumerate()
        .map(|(index, d)| {
            let name = device_name(&d);
            DeviceInfo {
                index,
                is_default: Some(&name) == default.as_ref(),
                name,
            }
        })
        .collect())
}

fn find_device(selector: Option<&str>) -> Result<cpal::Device> {
    let host = cpal::default_host();
    let Some(sel) = selector.filter(|s| !s.is_empty()) else {
        return host
            .default_output_device()
            .ok_or_else(|| Error::audio("no default output device"));
    };
    let devices: Vec<cpal::Device> = host
        .output_devices()
        .map_err(|e| audio_err("cannot list output devices", e))?
        .collect();
    if let Ok(index) = sel.parse::<usize>() {
        return devices.into_iter().nth(index).ok_or_else(|| {
            Error::audio(format!(
                "no output device with index {index}. Run speak -a '?'"
            ))
        });
    }
    let needle = sel.to_lowercase();
    devices
        .into_iter()
        .find(|d| device_name(d).to_lowercase().contains(&needle))
        .ok_or_else(|| {
            Error::audio(format!(
                "no output device matching '{sel}'. Run speak -a '?'"
            ))
        })
}

fn pick_config(device: &cpal::Device, want_rate: u32) -> Result<(StreamConfig, SampleFormat)> {
    let default = device
        .default_output_config()
        .map_err(|e| audio_err("no usable output config", e))?;
    let force = std::env::var_os("SPEAK_FORCE_RESAMPLE").is_some_and(|v| v != "0" && !v.is_empty());
    if !force
        && default.sample_rate() != want_rate
        && let Ok(configs) = device.supported_output_configs()
    {
        let mut best: Option<cpal::SupportedStreamConfig> = None;
        for range in configs {
            if range.channels() != default.channels() {
                continue;
            }
            if let Some(cfg) = range.try_with_sample_rate(want_rate) {
                let better = match &best {
                    None => true,
                    Some(b) => {
                        b.sample_format() != SampleFormat::F32
                            && cfg.sample_format() == SampleFormat::F32
                    }
                };
                if better {
                    best = Some(cfg);
                }
            }
        }
        if let Some(cfg) = best {
            return Ok((cfg.config(), cfg.sample_format()));
        }
    }
    Ok((default.config(), default.sample_format()))
}

fn build<T>(
    device: &cpal::Device,
    config: StreamConfig,
    mut consumer: rtrb::Consumer<f32>,
    shared: Arc<Shared>,
) -> std::result::Result<cpal::Stream, cpal::Error>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let cb = shared.clone();
    device.build_output_stream(
        config,
        move |data: &mut [T], info: &cpal::OutputCallbackInfo| {
            // Real-time thread: no locks, allocation or I/O.
            let ts = info.timestamp();
            let latency = ts.playback.duration_since(ts.callback);
            cb.latency_us
                .store(latency.as_micros() as u64, Ordering::Relaxed);
            let stopped = cb.stop.load(Ordering::Relaxed);
            let mut short = false;
            let mut played = 0u64;
            for frame in data.chunks_mut(channels) {
                let v = if stopped {
                    0.0
                } else {
                    match consumer.pop() {
                        Ok(v) => {
                            played += 1;
                            v
                        }
                        Err(_) => {
                            short = true;
                            0.0
                        }
                    }
                };
                let s = T::from_sample(v);
                for out in frame.iter_mut() {
                    *out = s;
                }
            }
            cb.played.fetch_add(played, Ordering::Relaxed);
            if short && cb.started.load(Ordering::Relaxed) && !cb.draining.load(Ordering::Relaxed) {
                cb.underruns.fetch_add(1, Ordering::Relaxed);
            }
        },
        |_err| {},
        None,
    )
}

impl Playback {
    /// Opens the device and starts the stream (it plays silence until audio is written).
    /// `stop` aborts playback: queued audio is dropped.
    pub fn open(device: Option<&str>, want_rate: u32, stop: Arc<AtomicBool>) -> Result<Playback> {
        let device = find_device(device)?;
        let (config, format) = pick_config(&device, want_rate)?;
        let device_rate = config.sample_rate;
        let (producer, consumer) = rtrb::RingBuffer::<f32>::new(device_rate as usize * 2);
        let shared = Arc::new(Shared {
            underruns: AtomicU64::new(0),
            started: AtomicBool::new(false),
            draining: AtomicBool::new(false),
            latency_us: AtomicU64::new(0),
            played: AtomicU64::new(0),
            stop,
        });
        let s = shared.clone();
        let stream = match format {
            SampleFormat::F32 => build::<f32>(&device, config, consumer, s),
            SampleFormat::I16 => build::<i16>(&device, config, consumer, s),
            SampleFormat::U16 => build::<u16>(&device, config, consumer, s),
            SampleFormat::I32 => build::<i32>(&device, config, consumer, s),
            SampleFormat::U8 => build::<u8>(&device, config, consumer, s),
            other => {
                return Err(Error::audio(format!(
                    "unsupported device sample format {other}"
                )));
            }
        }
        .map_err(|e| audio_err("cannot open the output stream", e))?;
        stream
            .play()
            .map_err(|e| audio_err("cannot start the output stream", e))?;
        Ok(Playback {
            stream: Some(stream),
            producer,
            shared,
            device_rate,
            resampler: None,
            stats: SinkStats::default(),
            scratch: Vec::new(),
        })
    }

    pub fn device_rate(&self) -> u32 {
        self.device_rate
    }
}

/// Pushes samples, parking in 2 ms steps while the ring buffer is full.
fn push(producer: &mut rtrb::Producer<f32>, shared: &Shared, samples: &[f32]) -> bool {
    for s in samples {
        let mut v = *s;
        loop {
            match producer.push(v) {
                Ok(()) => break,
                Err(rtrb::PushError::Full(back)) => {
                    v = back;
                    if shared.stop.load(Ordering::Relaxed) {
                        return false;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
        }
    }
    shared.started.store(true, Ordering::Relaxed);
    !shared.stop.load(Ordering::Relaxed)
}

impl PcmWriter for Playback {
    fn write(&mut self, samples: &[f32]) -> Result<Flow> {
        if samples.is_empty() {
            return Ok(Flow::Continue);
        }
        if self.stats.first_sample.is_none() {
            self.stats.first_sample = Some(Instant::now());
        }
        self.stats.samples += samples.len() as u64;
        let (producer, shared) = (&mut self.producer, &*self.shared);
        let go = match self.resampler.as_mut() {
            Some(r) => r.process(samples, &mut |out| Ok(push(producer, shared, out)))?,
            None => push(producer, shared, samples),
        };
        Ok(if go { Flow::Continue } else { Flow::Stop })
    }
}

impl AudioSink for Playback {
    fn begin(&mut self, sample_rate: u32) -> Result<()> {
        self.resampler = if sample_rate == self.device_rate {
            None
        } else {
            Some(StreamResampler::new(sample_rate, self.device_rate)?)
        };
        self.stats = SinkStats::default();
        self.scratch.clear();
        Ok(())
    }

    fn clock(&self) -> Option<Arc<dyn Fn() -> f64 + Send + Sync>> {
        let (shared, rate) = (self.shared.clone(), self.device_rate as f64);
        Some(Arc::new(move || {
            let played = shared.played.load(Ordering::Relaxed) as f64 / rate;
            played - shared.latency_us.load(Ordering::Relaxed) as f64 / 1e6
        }))
    }

    fn finish(&mut self) -> Result<SinkStats> {
        let (producer, shared) = (&mut self.producer, &*self.shared);
        if let Some(r) = self.resampler.as_mut() {
            r.flush(&mut |out| Ok(push(producer, shared, out)))?;
        }
        self.shared.draining.store(true, Ordering::Relaxed);
        let capacity = self.producer.buffer().capacity();
        while self.producer.slots() < capacity && !self.shared.stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(2));
        }
        if !self.shared.stop.load(Ordering::Relaxed) {
            // The device still holds the tail: wait for its reported latency plus a margin.
            let latency = Duration::from_micros(self.shared.latency_us.load(Ordering::Relaxed));
            std::thread::sleep(latency.min(Duration::from_millis(500)) + Duration::from_millis(50));
        }
        self.stream = None;
        self.stats.underruns = self.shared.underruns.load(Ordering::Relaxed);
        Ok(self.stats)
    }
}

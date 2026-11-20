//! File output: WAVE, AIFF and raw, to a file or to stdout (spec §10.2).
//! Headers are written first and patched on `finish()`; stdout gets unknown-length headers.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::audio::format::{DataFormat, FileFormat, extended_80};
use crate::audio::resample::StreamResampler;
use crate::error::{Error, Result};
use crate::traits::{AudioSink, Flow, PcmWriter, SinkStats};

enum Target {
    File(BufWriter<File>),
    Stdout(BufWriter<std::io::Stdout>),
}

impl Target {
    fn write_all(&mut self, b: &[u8]) -> std::io::Result<()> {
        match self {
            Target::File(f) => f.write_all(b),
            Target::Stdout(s) => s.write_all(b),
        }
    }
}

pub struct FileSink {
    target: Target,
    format: FileFormat,
    data: DataFormat,
    channels: u16,
    out_rate: u32,
    resampler: Option<StreamResampler>,
    /// Frames written to the file.
    frames: u64,
    stats: SinkStats,
    stop: Option<Arc<AtomicBool>>,
    started: bool,
    finished: bool,
    scratch: Vec<u8>,
}

impl FileSink {
    /// WAVE, 16-bit, mono: the plain case.
    pub fn create(path: &Path) -> Result<FileSink> {
        FileSink::with_format(
            Some(path),
            FileFormat::Wave,
            FileFormat::Wave.default_data(),
            1,
        )
    }

    /// `path = None` writes to stdout.
    pub fn with_format(
        path: Option<&Path>,
        format: FileFormat,
        data: DataFormat,
        channels: u16,
    ) -> Result<FileSink> {
        format.check(&data)?;
        if !(1..=2).contains(&channels) {
            return Err(Error::usage(format!(
                "--channels {channels}: expected 1 or 2"
            )));
        }
        let target = match path {
            Some(p) => Target::File(BufWriter::new(File::create(p).map_err(|e| {
                Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("{}: {e}", p.display()),
                ))
            })?)),
            None => Target::Stdout(BufWriter::new(std::io::stdout())),
        };
        Ok(FileSink {
            target,
            format,
            data,
            channels,
            out_rate: 0,
            resampler: None,
            frames: 0,
            stats: SinkStats::default(),
            stop: None,
            started: false,
            finished: false,
            scratch: Vec::new(),
        })
    }

    /// Interrupt flag: once set, `write` returns `Flow::Stop`.
    pub fn with_stop(mut self, stop: Arc<AtomicBool>) -> FileSink {
        self.stop = Some(stop);
        self
    }

    fn header(&self, frames: Option<u64>) -> Vec<u8> {
        let block = self.data.bytes() as u32 * self.channels as u32;
        let data_bytes = frames.map(|f| (f * block as u64).min(u32::MAX as u64 - 64) as u32);
        let mut h = Vec::with_capacity(64);
        match self.format {
            FileFormat::Raw => {}
            FileFormat::Wave => {
                h.extend_from_slice(b"RIFF");
                h.extend_from_slice(&data_bytes.map(|d| d + 36).unwrap_or(u32::MAX).to_le_bytes());
                h.extend_from_slice(b"WAVEfmt ");
                h.extend_from_slice(&16u32.to_le_bytes());
                h.extend_from_slice(&(if self.data.float { 3u16 } else { 1u16 }).to_le_bytes());
                h.extend_from_slice(&self.channels.to_le_bytes());
                h.extend_from_slice(&self.out_rate.to_le_bytes());
                h.extend_from_slice(&(self.out_rate * block).to_le_bytes());
                h.extend_from_slice(&(block as u16).to_le_bytes());
                h.extend_from_slice(&self.data.bits.to_le_bytes());
                h.extend_from_slice(b"data");
                h.extend_from_slice(&data_bytes.unwrap_or(u32::MAX).to_le_bytes());
            }
            FileFormat::Aiff => {
                let d = data_bytes.unwrap_or(0);
                let padded = d + (d & 1);
                h.extend_from_slice(b"FORM");
                h.extend_from_slice(&(4 + 8 + 18 + 8 + 8 + padded).to_be_bytes());
                h.extend_from_slice(b"AIFFCOMM");
                h.extend_from_slice(&18u32.to_be_bytes());
                h.extend_from_slice(&(self.channels as i16).to_be_bytes());
                h.extend_from_slice(
                    &(frames.unwrap_or(0).min(u32::MAX as u64) as u32).to_be_bytes(),
                );
                h.extend_from_slice(&(self.data.bits as i16).to_be_bytes());
                h.extend_from_slice(&extended_80(self.out_rate));
                h.extend_from_slice(b"SSND");
                h.extend_from_slice(&(8 + d).to_be_bytes());
                h.extend_from_slice(&[0u8; 8]);
            }
        }
        h
    }

    fn emit(&mut self, samples: &[f32]) -> Result<()> {
        self.scratch.clear();
        for s in samples {
            for _ in 0..self.channels {
                self.data.encode(*s, &mut self.scratch);
            }
        }
        self.frames += samples.len() as u64;
        self.target.write_all(&self.scratch)?;
        Ok(())
    }

    /// Writes the final header. Safe to call more than once.
    fn finalize(&mut self) -> Result<()> {
        if self.finished || !self.started {
            return Ok(());
        }
        self.finished = true;
        let block = self.data.bytes() as u64 * self.channels as u64;
        if self.format == FileFormat::Aiff && (self.frames * block) % 2 == 1 {
            self.target.write_all(&[0])?;
        }
        let header = self.header(Some(self.frames));
        match &mut self.target {
            Target::File(f) => {
                f.flush()?;
                if !header.is_empty() {
                    f.seek(SeekFrom::Start(0))?;
                    f.write_all(&header)?;
                    f.flush()?;
                }
            }
            Target::Stdout(s) => s.flush()?,
        }
        Ok(())
    }
}

impl PcmWriter for FileSink {
    fn write(&mut self, samples: &[f32]) -> Result<Flow> {
        if !self.started {
            return Err(Error::Synthesis("sink not started".into()));
        }
        if self
            .stop
            .as_ref()
            .is_some_and(|s| s.load(Ordering::Relaxed))
        {
            return Ok(Flow::Stop);
        }
        if samples.is_empty() {
            return Ok(Flow::Continue);
        }
        if self.stats.first_sample.is_none() {
            self.stats.first_sample = Some(Instant::now());
        }
        self.stats.samples += samples.len() as u64;
        match self.resampler.take() {
            Some(mut r) => {
                let res = r.process(samples, &mut |out| self.emit(out).map(|_| true));
                self.resampler = Some(r);
                res?;
            }
            None => self.emit(samples)?,
        }
        Ok(Flow::Continue)
    }
}

impl AudioSink for FileSink {
    fn begin(&mut self, sample_rate: u32) -> Result<()> {
        self.out_rate = self.data.rate.unwrap_or(sample_rate);
        self.resampler = if self.out_rate == sample_rate {
            None
        } else {
            Some(StreamResampler::new(sample_rate, self.out_rate)?)
        };
        self.started = true;
        // A pipe cannot seek: WAVE gets unknown-length sizes, AIFF a zero frame count.
        let header = self.header(None);
        self.target.write_all(&header)?;
        Ok(())
    }

    fn finish(&mut self) -> Result<SinkStats> {
        if let Some(mut r) = self.resampler.take() {
            r.flush(&mut |out| self.emit(out).map(|_| true))?;
        }
        self.finalize()?;
        Ok(self.stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::format::Endian;

    fn tone(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect()
    }

    fn write(path: &Path, format: FileFormat, data: &str, channels: u16) -> DataFormat {
        let d = DataFormat::parse(data, format.default_data()).unwrap();
        let mut sink = FileSink::with_format(Some(path), format, d, channels).unwrap();
        sink.begin(16000).unwrap();
        for block in tone(8000).chunks(320) {
            sink.write(block).unwrap();
        }
        sink.finish().unwrap();
        d
    }

    fn decode(bytes: &[u8], d: &DataFormat) -> Vec<f32> {
        bytes
            .chunks_exact(d.bytes())
            .map(|b| {
                let mut v = [0u8; 4];
                let n = b.len();
                if d.endian == Endian::Big {
                    for (i, x) in b.iter().enumerate() {
                        v[n - 1 - i] = *x;
                    }
                } else {
                    v[..n].copy_from_slice(b);
                }
                if d.float {
                    f32::from_le_bytes(v)
                } else {
                    let shift = 32 - d.bits as u32;
                    ((i32::from_le_bytes(v) << shift) >> shift) as f32
                        / ((1i64 << (d.bits - 1)) - 1) as f32
                }
            })
            .collect()
    }

    #[test]
    fn every_container_and_data_format_round_trips() {
        let dir = std::env::temp_dir().join(format!("speak-fmt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let reference = tone(8000);
        let cases: &[(FileFormat, &[&str], usize)] = &[
            (FileFormat::Wave, &["LEI16", "LEI24", "LEI32", "LEF32"], 44),
            (FileFormat::Aiff, &["BEI16", "BEI24", "BEI32"], 54),
            (
                FileFormat::Raw,
                &[
                    "LEI16", "LEI24", "LEI32", "LEF32", "BEI16", "BEI24", "BEI32", "BEF32",
                ],
                0,
            ),
        ];
        for (format, datas, header) in cases {
            for data in *datas {
                for channels in [1u16, 2] {
                    let path = dir.join(format!("{}-{data}-{channels}", format.name()));
                    let d = write(&path, *format, data, channels);
                    let bytes = std::fs::read(&path).unwrap();
                    let pcm = decode(&bytes[*header..], &d);
                    assert_eq!(
                        pcm.len(),
                        8000 * channels as usize,
                        "{} {data}",
                        format.name()
                    );
                    let step = 1.0 / ((1i64 << (d.bits - 1)) - 1) as f32;
                    for (i, r) in reference.iter().enumerate() {
                        let got = pcm[i * channels as usize];
                        assert!(
                            (got - r).abs() <= step,
                            "{} {data}: {got} vs {r}",
                            format.name()
                        );
                        assert_eq!(got, pcm[i * channels as usize + channels as usize - 1]);
                    }
                }
            }
        }
        // WAVE headers are readable by an independent reader.
        let path = dir.join("check.wav");
        write(&path, FileFormat::Wave, "LEI24", 2);
        let r = hound::WavReader::open(&path).unwrap();
        assert_eq!(
            (
                r.spec().channels,
                r.spec().sample_rate,
                r.spec().bits_per_sample,
                r.duration()
            ),
            (2, 16000, 24, 8000)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_pairs_and_resampling() {
        let d = DataFormat::parse("BEI16", FileFormat::Wave.default_data()).unwrap();
        assert!(FileSink::with_format(None, FileFormat::Wave, d, 1).is_err());
        let dir = std::env::temp_dir().join(format!("speak-fmt2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.wav");
        write(&path, FileFormat::Wave, "LEF32@48000", 1);
        let r = hound::WavReader::open(&path).unwrap();
        assert_eq!(r.spec().sample_rate, 48000);
        assert_eq!(r.spec().sample_format, hound::SampleFormat::Float);
        assert_eq!(r.duration(), 24000);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stop_flag_stops_and_leaves_a_valid_file() {
        let dir = std::env::temp_dir().join(format!("speak-fmt3-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("int.wav");
        let stop = Arc::new(AtomicBool::new(false));
        let mut sink = FileSink::create(&path).unwrap().with_stop(stop.clone());
        sink.begin(16000).unwrap();
        assert_eq!(sink.write(&tone(1600)).unwrap(), Flow::Continue);
        stop.store(true, Ordering::Relaxed);
        assert_eq!(sink.write(&tone(1600)).unwrap(), Flow::Stop);
        sink.finish().unwrap();
        assert_eq!(hound::WavReader::open(&path).unwrap().duration(), 1600);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

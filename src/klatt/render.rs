//! Renderer: parameter frames → samples (spec §8.1.3).

use crate::dsp::source::{Impulse, Klglott88, Tilt, flutter_hz};
use crate::dsp::{AntiResonator, Decimator, Noise, Resonator, soft_limit};
use crate::error::{Error, Result};
use crate::klatt::gains::{K_A, K_AB, K_AF, K_AH, K_AV, K_OUT, lin};
use crate::klatt::voice::{SourceKind, Voice};
use crate::klatt::{FRAME_SAMPLES, Frame, OVERSAMPLE, SR, UPDATE_SAMPLES};
use crate::traits::{Flow, GlottalSource, PcmWriter, VoiceQuality};

/// Output block: 20 ms.
const BLOCK_FRAMES: usize = 4;

pub struct Renderer {
    voice: Voice,
    source: Box<dyn GlottalSource>,
    tilt: Tilt,
    decim: Decimator,
    noise: Noise,
    rnp: Resonator,
    rnz: AntiResonator,
    cascade: [Resonator; 8],
    /// R2p…R6p.
    parallel: [Resonator; 5],
    /// Output samples rendered so far (drives flutter).
    clock: u64,
    /// Output samples since the last glottal epoch.
    since_epoch: f32,
    /// Per-period perturbations: their own generator, so the noise sources are unaffected.
    wobble: Noise,
    jitter: f32,
    shimmer: f32,
    /// Epochs (output sample index) of the last `render` call, when recording is on.
    pub epochs: Option<Vec<u64>>,
    block: Vec<f32>,
}

fn make_source(kind: SourceKind, rate: f32) -> Box<dyn GlottalSource> {
    match kind {
        SourceKind::Klglott88 => Box::new(Klglott88::new(rate)),
        SourceKind::Impulse => Box::new(Impulse::new(rate)),
        SourceKind::Lf => Box::new(crate::dsp::source_lf::Lf::new(rate)),
    }
}

impl Renderer {
    pub fn new(voice: &Voice) -> Renderer {
        let sr = SR as f32;
        let os = sr * OVERSAMPLE as f32;
        Renderer {
            voice: voice.clone(),
            source: make_source(voice.source, os),
            tilt: Tilt::new(os),
            decim: Decimator::new(OVERSAMPLE, 32, 0.45 * sr, sr),
            noise: Noise::new(0x5EED_1234),
            rnp: Resonator::default(),
            rnz: AntiResonator::default(),
            cascade: [Resonator::default(); 8],
            parallel: [Resonator::default(); 5],
            clock: 0,
            since_epoch: 0.0,
            wobble: Noise::new(0x0B0E_5EED),
            jitter: 1.0,
            shimmer: 1.0,
            epochs: None,
            block: Vec::with_capacity(BLOCK_FRAMES * FRAME_SAMPLES),
        }
    }

    pub fn voice(&self) -> &Voice {
        &self.voice
    }

    /// Clears filter and source state; the next render starts from silence.
    pub fn reset(&mut self) {
        self.source.reset();
        self.tilt.reset();
        self.decim.reset();
        self.noise = Noise::new(0x5EED_1234);
        self.rnp.reset();
        self.rnz.reset();
        for r in self.cascade.iter_mut().chain(self.parallel.iter_mut()) {
            r.reset();
        }
        self.clock = 0;
        self.since_epoch = 0.0;
        self.wobble = Noise::new(0x0B0E_5EED);
        self.jitter = 1.0;
        self.shimmer = 1.0;
    }

    fn set_coefficients(&mut self, p: &Frame) {
        let sr = SR as f32;
        let v = &self.voice;
        self.rnp.set(v.fnp, v.bnp, sr);
        self.rnz.set(p.fnz, v.bnz, sr);
        let f = [
            p.f[0],
            p.f[1],
            p.f[2],
            p.f[3],
            v.f_high[0] * v.formant_scale,
            v.f_high[1] * v.formant_scale,
            v.f_high[2] * v.formant_scale,
            v.f_high[3] * v.formant_scale,
        ];
        let b = [
            p.b[0],
            p.b[1],
            p.b[2],
            v.b4,
            v.b_high[0],
            v.b_high[1],
            v.b_high[2],
            v.b_high[3],
        ];
        for i in 0..8 {
            self.cascade[i].set(f[i], b[i], sr);
        }
        for i in 0..5 {
            self.parallel[i].set(f[i + 1], b[i + 1], sr);
        }
    }

    /// Renders `frames` (5 ms each) and writes 20 ms blocks to `out`.
    pub fn render(
        &mut self,
        frames: &[Frame],
        volume: f32,
        out: &mut dyn PcmWriter,
    ) -> Result<Flow> {
        let sr = SR as f32;
        for (bi, block) in frames.chunks(BLOCK_FRAMES).enumerate() {
            self.block.clear();
            for (fi, frame) in block.iter().enumerate() {
                let next = frames.get(bi * BLOCK_FRAMES + fi + 1).unwrap_or(frame);
                for step in 0..FRAME_SAMPLES / UPDATE_SAMPLES {
                    let t = step as f32 * UPDATE_SAMPLES as f32 / FRAME_SAMPLES as f32;
                    let p = frame.lerp(next, t);
                    self.set_coefficients(&p);
                    self.tilt.set(p.tl);
                    let q = VoiceQuality {
                        oq: p.oq,
                        tl: p.tl,
                        sq: self.voice.sq,
                        di: self.voice.di,
                    };
                    let av = K_AV * lin(p.av);
                    let ah = K_AH * lin(p.ah);
                    let af = K_AF * lin(p.af);
                    let a: [f32; 5] = std::array::from_fn(|i| K_A[i] * lin(p.a[i]));
                    let ab = K_AB * lin(p.ab);
                    let voiced = p.f0 > 0.0 && p.av > 0.0;
                    for _ in 0..UPDATE_SAMPLES {
                        let f0 = if voiced {
                            p.f0 + flutter_hz(self.voice.flutter, p.f0, self.clock as f32 / sr)
                        } else {
                            0.0
                        };
                        // Voicing source at 4× the output rate.
                        let mut v = 0.0;
                        for _ in 0..OVERSAMPLE {
                            let s = self.source.next(f0 * self.jitter, &q) * self.shimmer;
                            if self.source.epoch() {
                                self.since_epoch = 0.0;
                                if self.voice.jitter > 0.0 || self.voice.shimmer > 0.0 {
                                    self.jitter =
                                        1.0 + self.voice.jitter / 100.0 * self.wobble.sample();
                                    self.shimmer =
                                        1.0 + self.voice.shimmer / 100.0 * self.wobble.sample();
                                }
                                if let Some(e) = self.epochs.as_mut() {
                                    e.push(self.clock);
                                }
                            }
                            if let Some(y) = self.decim.push(self.tilt.process(s)) {
                                v = y;
                            }
                        }
                        // Noise is halved during the second half of each glottal period.
                        let mut n = self.noise.sample();
                        if voiced && self.since_epoch > 0.5 * sr / f0.max(1.0) {
                            n *= 0.5;
                        }
                        self.since_epoch += 1.0;

                        let mut c = av * v + ah * n;
                        c = self.rnz.process(self.rnp.process(c));
                        for r in &mut self.cascade {
                            c = r.process(c);
                        }

                        let nf = af * n;
                        let mut par = 0.0;
                        if af > 0.0 {
                            par = self.parallel[4].process(a[4] * nf);
                            par = self.parallel[3].process(a[3] * nf) - par;
                            par = self.parallel[2].process(a[2] * nf) - par;
                            par = self.parallel[1].process(a[1] * nf) - par;
                            par = self.parallel[0].process(a[0] * nf) - par;
                            par = ab * nf - par;
                        }
                        self.block.push(soft_limit(K_OUT * volume * (c + par)));
                        self.clock += 1;
                    }
                }
            }
            if self.block.iter().any(|s| !s.is_finite()) {
                return Err(Error::Synthesis(
                    "non-finite sample in the formant renderer".into(),
                ));
            }
            if out.write(&self.block)? == Flow::Stop {
                return Ok(Flow::Stop);
            }
        }
        Ok(Flow::Continue)
    }
}

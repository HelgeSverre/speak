//! One 5 ms parameter frame (spec §8.1.5).

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    /// Hz; 0 = unvoiced.
    pub f0: f32,
    /// Voicing, aspiration and frication amplitudes in dB, 0..80; 0 = off.
    pub av: f32,
    pub ah: f32,
    pub af: f32,
    /// Open quotient, 0.3..0.8.
    pub oq: f32,
    /// Spectral tilt, dB down at 3 kHz, 0..34.
    pub tl: f32,
    /// F1–F4 in Hz.
    pub f: [f32; 4],
    /// B1–B3 in Hz.
    pub b: [f32; 3],
    /// Nasal zero in Hz; equal to FNP = no nasalization.
    pub fnz: f32,
    /// Parallel gains A2–A6 in dB.
    pub a: [f32; 5],
    /// Bypass gain in dB.
    pub ab: f32,
}

impl Frame {
    pub const SILENCE: Frame = Frame {
        f0: 0.0,
        av: 0.0,
        ah: 0.0,
        af: 0.0,
        oq: 0.5,
        tl: 0.0,
        f: [500.0, 1500.0, 2500.0, 3300.0],
        b: [60.0, 90.0, 150.0],
        fnz: 270.0,
        a: [0.0; 5],
        ab: 0.0,
    };

    pub const TSV_HEADER: &'static str =
        "t_ms\tf0\tav\tah\taf\toq\ttl\tf1\tf2\tf3\tf4\tb1\tb2\tb3\tfnz\ta2\ta3\ta4\ta5\ta6\tab";

    pub fn tsv(&self, t_ms: f32) -> String {
        format!(
            "{:.0}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.2}\t{:.1}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{:.0}\t{:.0}",
            t_ms,
            self.f0,
            self.av,
            self.ah,
            self.af,
            self.oq,
            self.tl,
            self.f[0],
            self.f[1],
            self.f[2],
            self.f[3],
            self.b[0],
            self.b[1],
            self.b[2],
            self.fnz,
            self.a[0],
            self.a[1],
            self.a[2],
            self.a[3],
            self.a[4],
            self.ab
        )
    }

    /// Linear interpolation toward `next`. F0 is not interpolated across a voicing change.
    pub fn lerp(&self, next: &Frame, t: f32) -> Frame {
        let l = |a: f32, b: f32| a + (b - a) * t;
        let mut out = *self;
        out.f0 = if self.f0 > 0.0 && next.f0 > 0.0 {
            l(self.f0, next.f0)
        } else {
            self.f0
        };
        out.av = l(self.av, next.av);
        out.ah = l(self.ah, next.ah);
        out.af = l(self.af, next.af);
        out.oq = l(self.oq, next.oq);
        out.tl = l(self.tl, next.tl);
        for i in 0..4 {
            out.f[i] = l(self.f[i], next.f[i]);
        }
        for i in 0..3 {
            out.b[i] = l(self.b[i], next.b[i]);
        }
        out.fnz = l(self.fnz, next.fnz);
        for i in 0..5 {
            out.a[i] = l(self.a[i], next.a[i]);
        }
        out.ab = l(self.ab, next.ab);
        out
    }
}

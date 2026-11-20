//! Unit lookup, time mapping and joining (spec §8.2.5).

use crate::chunk::{Chunk, Unit};
use crate::diphone::format::{Diphone, VoiceFile};
use crate::diphone::{FRAME_MS, inventory};
use crate::dsp::lpc::LpcFrame;
use crate::phone::{Class, Phone};

/// Frames around a stop release that keep their original length (20 ms).
const PROTECT_FRAMES: usize = 4;
/// Half-width of the smoothing at a join, in frames (15 ms).
const JOIN_FRAMES: usize = 3;

/// One selected diphone, for `--dump units`.
#[derive(Clone, Debug, PartialEq)]
pub struct Selected {
    pub a: Phone,
    pub b: Phone,
    pub source_ms: f32,
    pub target_ms: f32,
    /// The diphone was missing and replaced by `A-_` + `_-B`.
    pub fallback: bool,
}

/// A stretch of source frames mapped onto `[t0, t1)` of the output.
struct Piece<'a> {
    t0: f32,
    t1: f32,
    src: &'a [LpcFrame],
    /// Source frames at the start / end that are not stretched.
    keep_head: usize,
    keep_tail: usize,
    /// Where `src` comes from: (diphone index in the voice, first frame within it).
    origin: Option<(usize, usize)>,
}

impl Piece<'_> {
    /// Source position (in frames) for output time `t`.
    fn source_pos(&self, t: f32) -> f32 {
        let n = self.src.len() as f32;
        let dur = self.t1 - self.t0;
        if n <= 0.0 || dur <= 0.0 {
            return 0.0;
        }
        let local = t - self.t0;
        let (head, tail) = (self.keep_head as f32, self.keep_tail as f32);
        let (head_ms, tail_ms) = (head * FRAME_MS, tail * FRAME_MS);
        // The protected parts need room, and something must be left to stretch.
        if head_ms + tail_ms >= dur || head + tail >= n {
            return local / dur * n;
        }
        if local < head_ms {
            local / FRAME_MS
        } else if local >= dur - tail_ms {
            n - (dur - local) / FRAME_MS
        } else {
            head + (local - head_ms) / (dur - head_ms - tail_ms) * (n - head - tail)
        }
    }

    fn frame(&self, t: f32) -> LpcFrame {
        let pos = self
            .source_pos(t)
            .clamp(0.0, (self.src.len().max(1) - 1) as f32);
        let i = pos as usize;
        let next = (i + 1).min(self.src.len() - 1);
        self.src[i].lerp(&self.src[next], pos.fract())
    }
}

fn is_stop(p: Phone) -> bool {
    matches!(p.class(), Class::Stop | Class::Affricate)
}

pub struct Plan {
    /// One frame per 5 ms, F0 from the prosody targets.
    pub frames: Vec<LpcFrame>,
    pub units: Vec<Selected>,
    /// For each output frame: the diphone it came from and the position in it (in frames).
    pub map: Vec<Option<(usize, f32)>>,
}

/// F0 track from the segments' targets: linear between points, held across gaps.
fn f0_track(chunk: &Chunk, n: usize, default_hz: f32) -> Vec<f32> {
    let mut pts: Vec<(f32, f32)> = Vec::new();
    let mut t = 0.0f32;
    for u in &chunk.units {
        if let Unit::Seg(s) = u {
            for (pos, hz) in &s.f0 {
                let at = t + pos * s.dur_ms;
                if pts.last().is_none_or(|l| at > l.0) {
                    pts.push((at, *hz));
                }
            }
        }
        t += u.dur_ms();
    }
    if pts.is_empty() {
        pts.push((0.0, default_hz));
    }
    let mut cursor = 0usize;
    (0..n)
        .map(|j| {
            let t = j as f32 * FRAME_MS;
            while cursor + 1 < pts.len() && pts[cursor + 1].0 <= t {
                cursor += 1;
            }
            let a = pts[cursor];
            match pts.get(cursor + 1) {
                Some(b) if t > a.0 => a.1 + (b.1 - a.1) * (t - a.0) / (b.0 - a.0).max(1e-6),
                _ => a.1,
            }
        })
        .collect()
}

/// Builds the frame sequence for a timed chunk, including the pause after it.
pub fn plan(chunk: &Chunk, voice: &VoiceFile, default_hz: f32) -> Plan {
    // Phones with durations, silence at both ends.
    let mut phones: Vec<(Phone, f32)> = vec![(Phone::SIL, 0.0)];
    for u in &chunk.units {
        match u {
            Unit::Seg(s) if s.dur_ms > 0.0 => phones.push((s.phone, s.dur_ms)),
            Unit::Silence { ms } if *ms > 0.0 => phones.push((Phone::SIL, *ms)),
            _ => {}
        }
    }
    phones.push((Phone::SIL, chunk.pause_after_ms.max(0.0)));
    let total: f32 = phones.iter().map(|p| p.1).sum();
    let n = (total / FRAME_MS).round() as usize;

    let empty: &[LpcFrame] = &[];
    let halves =
        |d: &Diphone| -> (usize, usize) { (d.boundary.min(d.frames.len()), d.frames.len()) };
    let mut pieces: Vec<Piece> = Vec::new();
    let mut units = Vec::new();
    let mut joins: Vec<f32> = Vec::new();
    let mut t = 0.0f32;
    for i in 0..phones.len() - 1 {
        let ((a, da), (b, db)) = (phones[i], phones[i + 1]);
        let (first_ms, second_ms) = (da / 2.0, db / 2.0);
        // First half covers the second half of A, second half the first half of B.
        type Half<'v> = (&'v [LpcFrame], Option<(usize, usize)>);
        let (first, second, fallback): (Half, Half, bool) = match voice.get(a, b) {
            Some(d) => {
                let (bd, end) = halves(d);
                let id = voice.index_of(a, b);
                (
                    (&d.frames[..bd], id.map(|i| (i, 0))),
                    (&d.frames[bd..end], id.map(|i| (i, bd))),
                    false,
                )
            }
            None => {
                // Missing: A-_ then _-B, without the silence between them.
                let first: Half = match voice.get(a, Phone::SIL) {
                    Some(d) => (
                        &d.frames[..halves(d).0],
                        voice.index_of(a, Phone::SIL).map(|i| (i, 0)),
                    ),
                    None => (empty, None),
                };
                let second: Half = match voice.get(Phone::SIL, b) {
                    Some(d) => {
                        let (bd, end) = halves(d);
                        (
                            &d.frames[bd..end],
                            voice.index_of(Phone::SIL, b).map(|i| (i, bd)),
                        )
                    }
                    None => (empty, None),
                };
                (first, second, !(a == Phone::SIL && b == Phone::SIL))
            }
        };
        let (src_first, src_second) = (first.0, second.0);
        let source_ms = (src_first.len() + src_second.len()) as f32 * FRAME_MS;
        if first_ms + second_ms > 0.0 {
            units.push(Selected {
                a,
                b,
                source_ms,
                target_ms: first_ms + second_ms,
                fallback,
            });
        }
        // The burst of a stop sits just before the boundary, its aspiration just after: keep both.
        let protect = if is_stop(a) { PROTECT_FRAMES } else { 0 };
        pieces.push(Piece {
            t0: t,
            t1: t + first_ms,
            src: src_first,
            keep_head: 0,
            keep_tail: protect,
            origin: first.1,
        });
        t += first_ms;
        pieces.push(Piece {
            t0: t,
            t1: t + second_ms,
            src: src_second,
            keep_head: protect,
            keep_tail: 0,
            origin: second.1,
        });
        t += second_ms;
        // The next diphone starts at the middle of B: a join.
        if i + 2 < phones.len() && b != Phone::SIL {
            joins.push(t);
        }
    }

    let f0 = f0_track(chunk, n, default_hz);
    let mut frames = Vec::with_capacity(n);
    let mut map = Vec::with_capacity(n);
    let mut pi = 0usize;
    for (j, hz) in f0.iter().enumerate() {
        let time = j as f32 * FRAME_MS;
        while pi + 1 < pieces.len()
            && (time >= pieces[pi].t1
                || pieces[pi].src.is_empty() && pieces[pi].t1 <= pieces[pi].t0)
        {
            pi += 1;
        }
        let p = &pieces[pi];
        let mut f = if p.src.is_empty() {
            LpcFrame::SILENT
        } else {
            p.frame(time)
        };
        f.f0 = *hz;
        frames.push(f);
        map.push(match p.origin {
            Some((id, first)) if !p.src.is_empty() => Some((
                id,
                first as f32 + p.source_pos(time).clamp(0.0, p.src.len() as f32),
            )),
            _ => None,
        });
    }

    // Joins: cross-interpolate filter and gain over ±3 frames.
    for join in joins {
        let j = (join / FRAME_MS).round() as usize;
        if j < JOIN_FRAMES || j + JOIN_FRAMES >= frames.len() {
            continue;
        }
        let (left, right) = (
            frames[j - JOIN_FRAMES].clone(),
            frames[j + JOIN_FRAMES].clone(),
        );
        // Only smooth where both sides carry sound; a join into silence stays sharp.
        if left.gain <= 0.0 || right.gain <= 0.0 {
            continue;
        }
        for d in 1..2 * JOIN_FRAMES {
            let x = d as f32 / (2 * JOIN_FRAMES) as f32;
            let target = left.lerp(&right, x);
            let cur = &mut frames[j - JOIN_FRAMES + d];
            // Blend half-way toward the straight line: removes the step, keeps the detail.
            for i in 0..cur.k.len() {
                cur.k[i] = 0.5 * (cur.k[i] + target.k[i]);
            }
            cur.gain = 0.5 * (cur.gain + target.gain);
        }
    }
    Plan { frames, units, map }
}

/// Text for `--dump units`.
pub fn dump_units(units: &[Selected]) -> String {
    let mut s = String::new();
    for u in units {
        s.push_str(&format!(
            "{:<8} {:>5.0} ms -> {:>5.0} ms{}\n",
            inventory::name(u.a, u.b),
            u.source_ms,
            u.target_ms,
            if u.fallback {
                "  (missing: fallback)"
            } else {
                ""
            }
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pho::parse_pho;

    fn voice(pairs: &[(Phone, Phone)]) -> VoiceFile {
        let frame = |g: f32| {
            let mut f = LpcFrame::SILENT;
            f.gain = g;
            f.k[0] = g;
            f.voicing = 1.0;
            f
        };
        let diphones = pairs
            .iter()
            .enumerate()
            .map(|(i, (a, b))| Diphone {
                a: *a,
                b: *b,
                frames: (0..10)
                    .map(|k| frame(0.1 + 0.01 * i as f32 + 0.001 * k as f32))
                    .collect(),
                boundary: 5,
                residual: Vec::new(),
                epochs: Vec::new(),
            })
            .collect();
        VoiceFile::new("test".into(), diphones)
    }

    #[test]
    fn frame_count_matches_the_durations() {
        let v = voice(&[
            (Phone::SIL, Phone::HH),
            (Phone::HH, Phone::AX),
            (Phone::AX, Phone::SIL),
        ]);
        let c = parse_pho("HH 60\nAX 120 0 110 100 90\n_ 100\n")
            .unwrap()
            .remove(0);
        let p = plan(&c, &v, 110.0);
        assert_eq!(p.frames.len(), 56);
        assert_eq!(p.units.len(), 4);
        assert_eq!(p.units[0].a, Phone::SIL);
        assert!(p.units.iter().take(3).all(|u| !u.fallback));
        // F0 comes from the targets, not from the store.
        assert!((p.frames[12].f0 - 110.0).abs() < 1.0);
        assert!(p.frames[35].f0 < 100.0);
        assert!(dump_units(&p.units).starts_with("_-HH"));
    }

    #[test]
    fn missing_diphones_fall_back_through_silence() {
        let v = voice(&[
            (Phone::SIL, Phone::AA),
            (Phone::AA, Phone::SIL),
            (Phone::SIL, Phone::B),
            (Phone::B, Phone::SIL),
        ]);
        let c = parse_pho("AA 100 0 110\nB 80\n").unwrap().remove(0);
        let p = plan(&c, &v, 110.0);
        let ab = p
            .units
            .iter()
            .find(|u| u.a == Phone::AA && u.b == Phone::B)
            .unwrap();
        assert!(ab.fallback);
        assert!(
            p.frames.iter().take(30).all(|f| f.gain > 0.0),
            "the fallback still produces sound"
        );
    }

    #[test]
    fn protected_frames_are_not_stretched() {
        let src = vec![LpcFrame::SILENT; 10];
        let piece = Piece {
            t0: 0.0,
            t1: 200.0,
            src: &src,
            keep_head: 0,
            keep_tail: 4,
            origin: None,
        };
        // The last 20 ms map one to one onto the last four frames.
        assert!((piece.source_pos(180.0) - 6.0).abs() < 1e-3);
        assert!((piece.source_pos(190.0) - 8.0).abs() < 1e-3);
        // The rest is stretched: 180 ms for six frames.
        assert!((piece.source_pos(90.0) - 3.0).abs() < 1e-3);
    }
}

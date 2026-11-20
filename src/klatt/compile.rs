//! Rule compiler: timed phones → parameter frames (spec §8.1.7).
//!
//! Step 1 expands phones to sub-segments, step 2 builds formant and bandwidth tracks with
//! rank-and-locus boundaries, step 3 builds the source tracks, step 4 samples every 5 ms.
//! All numbers come from `tables.rs`.

use crate::chunk::{Chunk, Unit};
use crate::klatt::context::{self, VowelContext};
use crate::klatt::tables::context as ctx_table;
use crate::klatt::tables::{Transition, amp, burst_ms, row, transition};
use crate::klatt::voice::Voice;
use crate::klatt::{FRAME_MS, Frame};
use crate::phone::{Class, Phone, Stress};

/// One phone with its timing and resolved targets. Parameter order: F1 F2 F3 B1 B2 B3.
#[derive(Clone, Debug)]
struct Ph {
    phone: Phone,
    stress: Stress,
    word: usize,
    start: f32,
    dur: f32,
    t_start: [f32; 6],
    t_end: [f32; 6],
    /// False for silence: no transitions into or out of it.
    has_targets: bool,
    trans: Transition,
    /// Boundary values and transition times at the left and right edge.
    left: [f32; 6],
    right: [f32; 6],
    tl: f32,
    tr: f32,
    /// Class of the vowel this consonant leans on.
    ctx: Option<VowelContext>,
    /// F2/F3 locus used at the boundaries of a nasal (a stop carries it in its targets).
    locus: Option<[f32; 2]>,
    /// F3 the vowel glides to before `R`.
    f3_tail: Option<f32>,
}

/// Source settings of one sub-segment.
#[derive(Clone, Copy, Debug)]
struct Sub {
    start: f32,
    end: f32,
    av: f32,
    ah: f32,
    af: f32,
    a: [f32; 5],
    ab: f32,
    /// Tilt override; `None` = the voice's TL.
    tl: Option<f32>,
    /// B1 and F1 overrides (aspiration, voice bar).
    b1: Option<f32>,
    f1: Option<f32>,
    /// No amplitude ramp at the start (closure → burst).
    abrupt: bool,
}

impl Sub {
    fn silent(start: f32, end: f32) -> Sub {
        Sub {
            start,
            end,
            av: 0.0,
            ah: 0.0,
            af: 0.0,
            a: [0.0; 5],
            ab: 0.0,
            tl: None,
            b1: None,
            f1: None,
            abrupt: false,
        }
    }
}

fn w(u: f32) -> f32 {
    if (0.0..1.0).contains(&u) {
        0.5 * (1.0 + (std::f32::consts::PI * u).cos())
    } else {
        0.0
    }
}

fn targets(p: Phone) -> ([f32; 6], [f32; 6]) {
    let r = row(p);
    (
        [r.f[0].0, r.f[1].0, r.f[2].0, r.b[0].0, r.b[1].0, r.b[2].0],
        [r.f[0].1, r.f[1].1, r.f[2].1, r.b[0].1, r.b[1].1, r.b[2].1],
    )
}

fn collect(chunk: &Chunk) -> Vec<Ph> {
    let mut phones = Vec::new();
    let mut t = 0.0f32;
    let mut word = 0usize;
    let mut push = |phone: Phone, stress: Stress, dur: f32, word: usize, t: &mut f32| {
        if dur <= 0.0 {
            return;
        }
        let (t_start, t_end) = targets(phone);
        phones.push(Ph {
            phone,
            stress,
            word,
            start: *t,
            dur,
            t_start,
            t_end,
            has_targets: phone != Phone::SIL,
            trans: transition(phone),
            left: t_start,
            right: t_end,
            tl: 0.0,
            tr: 0.0,
            ctx: None,
            locus: None,
            f3_tail: None,
        });
        *t += dur;
    };
    for u in &chunk.units {
        match u {
            Unit::Seg(s) => push(s.phone, s.stress, s.dur_ms, word, &mut t),
            Unit::Silence { ms } => {
                word += 1;
                push(Phone::SIL, Stress::None, *ms, word, &mut t)
            }
            Unit::WordBreak | Unit::Punct(_) => word += 1,
        }
    }
    push(
        Phone::SIL,
        Stress::None,
        chunk.pause_after_ms,
        word + 1,
        &mut t,
    );
    phones
}

/// `HH` and `Q` have no targets of their own: they take the neighbour's values.
fn resolve_copies(phones: &mut [Ph]) {
    for i in 0..phones.len() {
        if !matches!(phones[i].phone, Phone::HH | Phone::Q) {
            continue;
        }
        let next = phones
            .get(i + 1)
            .filter(|p| p.has_targets && p.trans.rank > 0)
            .map(|p| (p.t_start, p.trans));
        let prev = i
            .checked_sub(1)
            .map(|j| &phones[j])
            .filter(|p| p.has_targets && p.trans.rank > 0)
            .map(|p| (p.t_end, p.trans));
        // Prefer the next vowel; else whatever is next; else the previous phone.
        if let Some((t, trans)) = next.or(prev) {
            let b = [
                phones[i].t_start[3],
                phones[i].t_start[4],
                phones[i].t_start[5],
            ];
            phones[i].t_start = [t[0], t[1], t[2], b[0], b[1], b[2]];
            phones[i].t_end = phones[i].t_start;
            phones[i].trans = Transition { rank: 0, ..trans };
        }
    }
}

/// Context rules that change targets (spec §8.1.8): loci, dark L, R colouring, reduction.
fn apply_context(phones: &mut [Ph]) {
    let n = phones.len();
    for i in 0..n {
        let phone = phones[i].phone;
        let word = phones[i].word;
        // The vowel a consonant leans on: the next one in the word (through liquids), else the previous.
        let mut ctx = None;
        if phone.is_consonant() {
            let mut j = i + 1;
            while j < n && phones[j].word == word {
                let c = phones[j].phone;
                if let Some(k) = context::class_of(c).filter(|_| {
                    c != Phone::R
                        || !phones
                            .get(j + 1)
                            .is_some_and(|v| v.phone.is_vowel() && v.word == word)
                }) {
                    ctx = Some(k);
                    break;
                }
                if !matches!(c.class(), Class::Liquid | Class::Glide) {
                    break;
                }
                if c == Phone::L && matches!(phone, Phone::K | Phone::G) {
                    ctx = Some(VowelContext::BackUnrounded);
                    break;
                }
                j += 1;
            }
            if ctx.is_none() && i > 0 && phones[i - 1].phone.is_vowel() {
                ctx = context::class_of(phones[i - 1].phone);
            }
        }
        phones[i].ctx = ctx;
        let next_in_word = phones
            .get(i + 1)
            .filter(|p| p.word == word)
            .map(|p| p.phone);
        if let Some(c) = ctx
            && let Some((f2, f3)) = context::locus(phone, c)
        {
            if phone.class() == Class::Stop {
                // The closure and burst sit on the locus.
                for t in [&mut phones[i].t_start, &mut phones[i].t_end] {
                    t[1] = f2;
                    if let Some(f3) = f3 {
                        t[2] = f3;
                    }
                }
            } else {
                // A nasal keeps its murmur formants; its transitions point at the locus.
                phones[i].locus = Some([f2, f3.unwrap_or(phones[i].t_start[2])]);
            }
        }
        if phone == Phone::L && !next_in_word.is_some_and(|p| p.is_vowel()) {
            for t in [&mut phones[i].t_start, &mut phones[i].t_end] {
                t[1] = ctx_table::DARK_L_F2;
                t[2] = ctx_table::DARK_L_F3;
            }
        }
        if phone.is_vowel() {
            if next_in_word == Some(Phone::R) {
                phones[i].f3_tail = Some(ctx_table::R_COLOUR_F3);
            }
            // Short vowels move toward schwa, except before a pause.
            let phrase_final = phones.get(i + 1).is_none_or(|p| p.phone == Phone::SIL);
            if !phrase_final
                && !phone.is_reduced()
                && phones[i].dur < ctx_table::REDUCE_BELOW * row(phone).inh
            {
                let (schwa, _) = targets(Phone::AX);
                for t in [&mut phones[i].t_start, &mut phones[i].t_end] {
                    for k in 0..3 {
                        t[k] += ctx_table::REDUCE_AMOUNT * (schwa[k] - t[k]);
                    }
                }
            }
        }
    }
}

/// Step 2: boundary values and transition times (Holmes-style ranks, locus-type boundaries).
fn boundaries(phones: &mut [Ph]) {
    for ph in phones.iter_mut() {
        ph.left = ph.t_start;
        ph.right = ph.t_end;
    }
    for i in 0..phones.len().saturating_sub(1) {
        let (a, b) = (phones[i].clone(), phones[i + 1].clone());
        if !a.has_targets || !b.has_targets {
            continue;
        }
        let (ra, rb) = (a.trans.rank, b.trans.rank);
        let mut value = [0.0f32; 6];
        for (p, v) in value.iter_mut().enumerate() {
            // A nasal's transitions use its locus, not its murmur formants.
            let at = |ph: &Ph, own: f32| match (ph.locus, p) {
                (Some(l), 1) => l[0],
                (Some(l), 2) => l[1],
                _ => own,
            };
            let (ta, tb) = (at(&a, a.t_end[p]), at(&b, b.t_start[p]));
            let first_formant = p == 0 || p == 3;
            *v = if ra > rb {
                let k = if first_formant {
                    a.trans.k1
                } else {
                    a.trans.k23
                };
                ta + k * (tb - ta)
            } else if rb > ra {
                let k = if first_formant {
                    b.trans.k1
                } else {
                    b.trans.k23
                };
                tb + k * (ta - tb)
            } else {
                0.5 * (ta + tb)
            };
        }
        // The dominant phone sets the times: t_int inside itself, t_ext inside the neighbour.
        let (time_a, time_b) = if ra > rb {
            (a.trans.t_int, a.trans.t_ext)
        } else if rb > ra {
            (b.trans.t_ext, b.trans.t_int)
        } else {
            (a.trans.t_int, b.trans.t_int)
        };
        // A nasal holds its murmur formants up to the edge; only the neighbour moves to the locus.
        if a.locus.is_none() {
            phones[i].right = value;
            phones[i].tr = time_a;
        }
        if b.locus.is_none() {
            phones[i + 1].left = value;
            phones[i + 1].tl = time_b;
        }
    }
}

/// Value of parameter `p` of phone `ph` at local time `t` (ms).
fn track(ph: &Ph, p: usize, t: f32) -> f32 {
    let d = ph.dur.max(1e-3);
    let x = ((t / d - amp::GLIDE_START) / (amp::GLIDE_END - amp::GLIDE_START)).clamp(0.0, 1.0);
    let mut own = ph.t_start[p] + (ph.t_end[p] - ph.t_start[p]) * x;
    // R colouring: F3 glides down over the end of the vowel.
    if p == 2
        && let Some(target) = ph.f3_tail
        && t / d > ctx_table::R_COLOUR_FROM
    {
        own += (target - own)
            * ((t / d - ctx_table::R_COLOUR_FROM) / (1.0 - ctx_table::R_COLOUR_FROM));
    }
    let mut v = own;
    if ph.tl > 0.0 {
        v += (ph.left[p] - ph.t_start[p]) * w(t / ph.tl);
    }
    if ph.tr > 0.0 {
        v += (ph.right[p] - ph.t_end[p]) * w((d - t) / ph.tr);
    }
    v
}

fn is_sonorant_target(p: Phone) -> bool {
    matches!(p.class(), Class::Vowel | Class::Liquid | Class::Glide)
}

/// Length of the aspirated start of phone `i` (spec §8.1.7, voiceless stop row).
fn aspiration_ms(phones: &[Ph], i: usize) -> f32 {
    let ph = &phones[i];
    if i == 0 || !is_sonorant_target(ph.phone) {
        return 0.0;
    }
    let stop = &phones[i - 1];
    if !stop.phone.is_voiceless_stop() {
        return 0.0;
    }
    if i >= 2 && phones[i - 2].phone == Phone::S && phones[i - 2].word == stop.word {
        return 0.0;
    }
    // Stressed vowel reached through the syllable onset (possibly via a liquid or glide).
    let mut j = i;
    while j < phones.len() && phones[j].word == stop.word && !phones[j].phone.is_vowel() {
        if !matches!(phones[j].phone.class(), Class::Liquid | Class::Glide) {
            break;
        }
        j += 1;
    }
    let stressed = phones
        .get(j)
        .is_some_and(|v| v.phone.is_vowel() && v.word == stop.word && v.stress != Stress::None);
    let n = if stressed {
        amp::ASP_STRESSED_MS
    } else {
        amp::ASP_OTHER_MS
    };
    n.min(ph.dur * 0.5)
}

/// Step 1 and the source half of step 3: sub-segments with their source settings.
fn sub_segments(phones: &[Ph], voice: &Voice) -> Vec<Sub> {
    let mut subs = Vec::with_capacity(phones.len() * 2);
    for (i, ph) in phones.iter().enumerate() {
        let r = row(ph.phone);
        let (start, end) = (ph.start, ph.start + ph.dur);
        let next = phones.get(i + 1);
        let before_pause = next.is_none_or(|n| n.phone == Phone::SIL);
        let before_stop =
            next.is_some_and(|n| matches!(n.phone.class(), Class::Stop | Class::Affricate));
        let mut s = Sub::silent(start, end);
        match ph.phone.class() {
            Class::Vowel | Class::Liquid | Class::Glide | Class::Flap => {
                s.av = match ph.phone.class() {
                    Class::Vowel => match ph.stress {
                        Stress::Primary => amp::VOWEL_PRIMARY,
                        Stress::Secondary => amp::VOWEL_SECONDARY,
                        Stress::None => amp::VOWEL_UNSTRESSED,
                    },
                    Class::Flap => amp::FLAP_AV,
                    _ => amp::LIQUID_GLIDE,
                };
                if ph.phone.is_vowel() {
                    s.ah = voice.breath;
                }
                let mut asp = aspiration_ms(phones, i);
                let mut asp_ah = amp::ASP_AH;
                // L R W Y after a voiceless stop: the first half is devoiced.
                if asp > 0.0 && matches!(ph.phone.class(), Class::Liquid | Class::Glide) {
                    asp = ph.dur * ctx_table::DEVOICE_FRACTION;
                    asp_ah = ctx_table::DEVOICE_AH;
                }
                if asp > 0.0 {
                    let mut a = Sub::silent(start, start + asp);
                    a.ah = asp_ah;
                    a.b1 = Some(amp::ASP_B1);
                    subs.push(a);
                    s.start = start + asp;
                }
                subs.push(s);
            }
            Class::Nasal => {
                s.av = amp::NASAL;
                subs.push(s);
            }
            Class::Fricative => {
                s.a = context::fricative_gains(ph.phone, ph.ctx, r.a);
                s.ab = r.ab;
                if ph.phone.is_voiced() {
                    s.av = amp::VFRIC_AV;
                    s.tl = Some(amp::VFRIC_TL);
                    s.af = amp::VFRIC_AF;
                } else {
                    s.af = amp::FRIC_AF;
                }
                subs.push(s);
            }
            Class::Stop => {
                let voiced = ph.phone.is_voiced();
                let mut burst = if before_stop {
                    0.0
                } else {
                    burst_ms(ph.phone).min(ph.dur * 0.5)
                };
                if burst < 1.0 {
                    burst = 0.0;
                }
                let mut closure = Sub::silent(start, end - burst);
                if voiced {
                    closure.av = amp::VOICEBAR_AV;
                    closure.tl = Some(amp::VOICEBAR_TL);
                    closure.f1 = Some(amp::VOICEBAR_F1);
                    closure.b1 = Some(amp::VOICEBAR_B1);
                }
                subs.push(closure);
                if burst > 0.0 {
                    let mut b = Sub::silent(end - burst, end);
                    b.af = amp::BURST_AF
                        - if before_pause {
                            amp::WEAK_BURST_DB
                        } else {
                            0.0
                        };
                    b.a = context::burst_gains(ph.phone, ph.ctx, r.a);
                    b.ab = r.ab;
                    b.abrupt = true;
                    if voiced {
                        b.av = amp::VOICEBAR_AV;
                        b.tl = Some(amp::VOICEBAR_TL);
                    }
                    subs.push(b);
                }
            }
            Class::Affricate => {
                let voiced = ph.phone.is_voiced();
                let split = start + ph.dur * amp::AFFRICATE_CLOSURE;
                let mut closure = Sub::silent(start, split);
                if voiced {
                    closure.av = amp::VOICEBAR_AV;
                    closure.tl = Some(amp::VOICEBAR_TL);
                    closure.f1 = Some(amp::VOICEBAR_F1);
                    closure.b1 = Some(amp::VOICEBAR_B1);
                }
                subs.push(closure);
                let mut f = Sub::silent(split, end);
                f.a = r.a;
                f.ab = r.ab;
                f.abrupt = true;
                if voiced {
                    f.av = amp::VFRIC_AV;
                    f.tl = Some(amp::VFRIC_TL);
                    f.af = amp::VFRIC_AF;
                } else {
                    f.af = amp::FRIC_AF;
                }
                subs.push(f);
            }
            Class::Aspirate => {
                s.ah = amp::HH_AH;
                s.b1 = Some(amp::HH_B1);
                subs.push(s);
            }
            Class::Glottal | Class::Silence => subs.push(s),
        }
    }
    subs
}

/// A scalar source parameter with a 10 ms linear ramp centred on each sub-segment boundary.
fn ramped(subs: &[Sub], i: usize, t: f32, get: impl Fn(&Sub) -> f32) -> f32 {
    let s = &subs[i];
    let half = amp::RAMP_MS / 2.0;
    let cur = get(s);
    if t - s.start < half && i > 0 && !s.abrupt {
        let x = 0.5 + (t - s.start) / amp::RAMP_MS;
        return get(&subs[i - 1]) + (cur - get(&subs[i - 1])) * x;
    }
    if s.end - t <= half && i + 1 < subs.len() && !subs[i + 1].abrupt {
        let x = 0.5 - (s.end - t) / amp::RAMP_MS;
        return cur + (get(&subs[i + 1]) - cur) * x;
    }
    cur
}

/// F0 track: the segments' `(position, Hz)` points, linearly interpolated, held across gaps.
fn f0_points(chunk: &Chunk, default_hz: f32) -> Vec<(f32, f32)> {
    let mut pts: Vec<(f32, f32)> = Vec::new();
    let mut t = 0.0f32;
    for u in &chunk.units {
        if let Unit::Seg(s) = u {
            for (pos, hz) in &s.f0 {
                let at = t + pos * s.dur_ms;
                match pts.last() {
                    Some(last) if at <= last.0 => {}
                    _ => pts.push((at, *hz)),
                }
            }
        }
        t += u.dur_ms();
    }
    if pts.is_empty() {
        pts.push((0.0, default_hz));
    }
    pts
}

fn interp(points: &[(f32, f32)], cursor: &mut usize, t: f32) -> f32 {
    while *cursor + 1 < points.len() && points[*cursor + 1].0 <= t {
        *cursor += 1;
    }
    let a = points[*cursor];
    match points.get(*cursor + 1) {
        Some(b) if t > a.0 => a.1 + (b.1 - a.1) * (t - a.0) / (b.0 - a.0).max(1e-6),
        _ => a.1,
    }
}

/// Klatt 1980 Table I ranges, scaled by the voice.
fn clamp_formants(f: &mut Frame, scale: f32) {
    const RANGE: [(f32, f32); 4] = [
        (150.0, 900.0),
        (500.0, 2500.0),
        (1300.0, 3500.0),
        (2500.0, 4500.0),
    ];
    for (k, (lo, hi)) in RANGE.iter().enumerate() {
        f.f[k] = f.f[k].clamp(lo * scale, hi * scale);
    }
    for b in &mut f.b {
        *b = b.clamp(30.0, 1000.0);
    }
}

/// Compiles one chunk (timed by the prosody stage) into 5 ms frames, including the pause after it.
pub fn compile(chunk: &Chunk, voice: &Voice) -> Vec<Frame> {
    let mut phones = collect(chunk);
    if phones.is_empty() {
        return Vec::new();
    }
    apply_context(&mut phones);
    resolve_copies(&mut phones);
    boundaries(&mut phones);
    let subs = sub_segments(&phones, voice);
    let f0 = f0_points(chunk, voice.base_hz);

    let total: f32 = phones.iter().map(|p| p.dur).sum();
    let falling = !matches!(
        chunk.terminator,
        crate::chunk::Terminator::Question
            | crate::chunk::Terminator::Comma
            | crate::chunk::Terminator::Semicolon
            | crate::chunk::Terminator::Colon
            | crate::chunk::Terminator::Dash
    );
    let n = (total / FRAME_MS).round() as usize;
    let mut frames = Vec::with_capacity(n);
    let (mut pi, mut si, mut fi) = (0usize, 0usize, 0usize);
    for j in 0..n {
        let t = j as f32 * FRAME_MS;
        while pi + 1 < phones.len() && t >= phones[pi].start + phones[pi].dur {
            pi += 1;
        }
        while si + 1 < subs.len() && t >= subs[si].end {
            si += 1;
        }
        let ph = &phones[pi];
        let local = t - ph.start;
        let sub = &subs[si];
        let mut f = Frame::SILENCE;

        // Step 2: formant and bandwidth tracks.
        for k in 0..3 {
            f.f[k] = track(ph, k, local);
            f.b[k] = track(ph, k + 3, local);
        }
        f.f[3] = amp::F4;
        if let Some(f1) = sub.f1 {
            f.f[0] = f1;
        }
        if let Some(b1) = sub.b1 {
            f.b[0] = b1;
        }

        // Nasal zero: 450 in the nasal, ramping over the adjacent 40 ms of neighbouring vowels.
        f.fnz = voice.fnp;
        let span = amp::NASAL_FNZ - voice.fnp;
        if ph.phone.class() == Class::Nasal {
            f.fnz = amp::NASAL_FNZ;
        } else if ph.phone.is_vowel() {
            let to_end = ph.dur - local;
            if phones
                .get(pi + 1)
                .is_some_and(|p| p.phone.class() == Class::Nasal)
                && to_end < amp::NASALIZE_MS
            {
                f.fnz = voice.fnp + span * (1.0 - to_end / amp::NASALIZE_MS);
            }
            if pi > 0 && phones[pi - 1].phone.class() == Class::Nasal && local < amp::NASALIZE_MS {
                f.fnz = f
                    .fnz
                    .max(voice.fnp + span * (1.0 - local / amp::NASALIZE_MS));
            }
        }

        // Nasalized spans have a wider B1.
        if f.fnz > voice.fnp + 1.0 && ph.phone.is_vowel() {
            f.b[0] = f.b[0].max(amp::NASALIZED_B1);
        }

        // Step 3: source tracks.
        f.av = ramped(&subs, si, t, |s| s.av);
        f.ah = ramped(&subs, si, t, |s| s.ah);
        f.af = ramped(&subs, si, t, |s| s.af);
        f.a = sub.a;
        f.ab = sub.ab;
        // Keep the parallel gains of a neighbouring obstruent while AF ramps across the boundary.
        if f.af > 0.0 && sub.af <= 0.0 {
            let neighbour = if t - sub.start < amp::RAMP_MS && si > 0 {
                &subs[si - 1]
            } else {
                &subs[(si + 1).min(subs.len() - 1)]
            };
            f.a = neighbour.a;
            f.ab = neighbour.ab;
        }
        f.tl = sub.tl.unwrap_or(voice.tl);
        f.oq = voice.oq;
        // AV falls by 6 dB over the last 100 ms before a pause.
        if f.av > 0.0 {
            let pause_at = phones[pi..]
                .iter()
                .find(|p| p.phone == Phone::SIL)
                .map(|p| p.start)
                .unwrap_or(total);
            let left = pause_at - t;
            if left < amp::FINAL_FALL_MS {
                f.av -= amp::FINAL_FALL_DB * (1.0 - left / amp::FINAL_FALL_MS);
            }
        }

        // Utterance-level voice quality (spec §8.1.6).
        if f.av > 0.0 && !voice.whisper {
            let pause_at = phones[pi..]
                .iter()
                .find(|p| p.phone == Phone::SIL)
                .map(|p| p.start)
                .unwrap_or(total);
            let left = pause_at - t;
            if left < amp::OFFSET_MS {
                // Breathy offset.
                let x = 1.0 - left / amp::OFFSET_MS;
                f.oq = (f.oq + amp::OFFSET_OQ * x).min(0.8);
                f.tl += amp::OFFSET_TL * x;
                f.ah = f.ah.max(amp::OFFSET_AH_FLOOR) + amp::OFFSET_AH * x;
            }
            // Hard onset after a pause or a glottal stop.
            let after_gap = pi == 0 || matches!(phones[pi - 1].phone, Phone::SIL | Phone::Q);
            if after_gap && local < amp::ONSET_MS {
                f.oq = (f.oq - amp::ONSET_OQ).max(0.3);
            }
        }

        // Phrase-final creak: on a falling ending the pitch drops and the glottis tightens.
        let fry_x = if voice.fry > 0.0
            && falling
            && f.av > 0.0
            && total - chunk.pause_after_ms - t < amp::FRY_MS
        {
            voice.fry * (1.0 - (total - chunk.pause_after_ms - t) / amp::FRY_MS).clamp(0.0, 1.0)
        } else {
            0.0
        };
        f.oq = (f.oq - amp::FRY_OQ * fry_x).max(0.3);

        // F0, with the glottal stop's dip in its neighbourhood.
        f.f0 = interp(&f0, &mut fi, t);
        for q in phones.iter().filter(|p| p.phone == Phone::Q) {
            if t >= q.start - amp::Q_SPAN_MS && t < q.start + q.dur + amp::Q_SPAN_MS {
                f.f0 *= 1.0 - amp::Q_F0_DROP;
            }
        }

        f.f0 *= 1.0 - amp::FRY_F0_DROP * fry_x;

        // Step 4: voice.
        for k in 0..4 {
            f.f[k] *= voice.formant_scale;
        }
        clamp_formants(&mut f, voice.formant_scale);
        if voice.whisper {
            if f.av > 0.0 {
                f.ah = f.ah.max(f.av - 5.0);
            }
            f.av = 0.0;
        }
        frames.push(f);
    }
    frames
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Terminator;
    use crate::phone::parse_arpabet;
    use crate::text::RuleProsody;
    use crate::traits::Prosody;

    pub fn timed(text: &str, voice: &Voice) -> Chunk {
        let units = parse_arpabet(text).unwrap();
        let terminator = match units.last() {
            Some(Unit::Punct(c)) => Terminator::from_char(*c),
            _ => Terminator::None,
        };
        let mut c = Chunk {
            units,
            terminator,
            pause_after_ms: terminator.pause_ms(),
            ..Chunk::default()
        };
        RuleProsody.apply(&mut c, &voice.prosody());
        c
    }

    fn male() -> Voice {
        Voice::find("klatt-male").unwrap()
    }

    #[test]
    fn frame_count_matches_duration() {
        let v = male();
        for text in [
            "HH AX L OW1 / W ER1 L D .",
            "S T R IY1 T S",
            "AA1",
            "B AH1 DX AXR , / Q AH1 Q OW0 ?",
        ] {
            let c = timed(text, &v);
            let frames = compile(&c, &v);
            let want = c.dur_ms() + c.pause_after_ms;
            assert!(
                (frames.len() as f32 * FRAME_MS - want).abs() <= FRAME_MS,
                "{text}"
            );
        }
    }

    #[test]
    fn parameters_stay_in_range() {
        for name in ["klatt-male", "klatt-female", "klatt-child"] {
            let v = Voice::find(name).unwrap();
            let c = timed(
                "DH AX / K W IH1 K / B R AW1 N / F AA1 K S / JH AH1 M P S / OW1 V AXR / DH AX / L EY1 Z IY0 / D AO1 G .",
                &v,
            );
            for f in compile(&c, &v) {
                let s = v.formant_scale;
                assert!(
                    f.f[0] >= 150.0 * s - 1.0 && f.f[0] <= 900.0 * s + 1.0,
                    "F1 {}",
                    f.f[0]
                );
                assert!(
                    f.f[1] >= 500.0 * s - 1.0 && f.f[1] <= 2500.0 * s + 1.0,
                    "F2 {}",
                    f.f[1]
                );
                assert!(
                    f.f[2] >= 1300.0 * s - 1.0 && f.f[2] <= 3500.0 * s + 1.0,
                    "F3 {}",
                    f.f[2]
                );
                assert!(f.f0.is_finite() && f.f0 > 40.0 && f.f0 < 600.0);
                assert!(
                    (0.0..=80.0).contains(&f.av)
                        && (0.0..=80.0).contains(&f.af)
                        && (0.0..=80.0).contains(&f.ah)
                );
            }
        }
    }

    #[test]
    fn voiceless_stop_aspirates_the_following_vowel() {
        let v = male();
        let c = timed("P AA1", &v);
        let frames = compile(&c, &v);
        let p_ms = c.segments().next().unwrap().dur_ms;
        let first_vowel = (p_ms / FRAME_MS).ceil() as usize + 1;
        // Aspiration: no voicing, AH on, while the formants are already those of the vowel side.
        assert!(
            frames[first_vowel].ah > 40.0 && frames[first_vowel].av < 30.0,
            "{:?}",
            frames[first_vowel]
        );
        let later = first_vowel + 12;
        assert!(frames[later].av > 55.0 && frames[later].ah < 5.0);
        // After S there is none.
        let c = timed("S P AA1", &v);
        let frames = compile(&c, &v);
        let onset: f32 = c.segments().take(2).map(|s| s.dur_ms).sum();
        let i = (onset / FRAME_MS).ceil() as usize + 2;
        assert!(frames[i].av > 50.0 && frames[i].ah < 5.0);
    }

    #[test]
    fn short_vowel_undershoots() {
        let v = male();
        let f1_mid = |ms: f32| {
            let mut c = Chunk {
                units: parse_arpabet("B AA1 B").unwrap(),
                timed: true,
                ..Chunk::default()
            };
            for u in &mut c.units {
                if let Unit::Seg(s) = u {
                    s.dur_ms = if s.phone == Phone::AA { ms } else { 60.0 };
                    s.f0 = vec![(0.0, 110.0)];
                }
            }
            let frames = compile(&c, &v);
            frames[((60.0 + ms / 2.0) / FRAME_MS) as usize].f[0]
        };
        assert!(
            f1_mid(50.0) < f1_mid(240.0) - 50.0,
            "{} vs {}",
            f1_mid(50.0),
            f1_mid(240.0)
        );
        assert!((f1_mid(240.0) - 700.0).abs() < 10.0);
    }

    #[test]
    fn whisper_has_no_voicing() {
        let v = Voice::find("klatt-whisper").unwrap();
        let c = timed("HH AX L OW1 .", &v);
        let frames = compile(&c, &v);
        assert!(frames.iter().all(|f| f.av == 0.0));
        assert!(frames.iter().any(|f| f.ah > 50.0));
    }

    #[test]
    fn nasal_sets_the_zero() {
        let v = male();
        let c = timed("M AA1 N", &v);
        let frames = compile(&c, &v);
        assert_eq!(frames[2].fnz, 450.0);
        let mid = frames.len() / 2;
        assert_eq!(frames[mid].fnz, 270.0);
    }
}

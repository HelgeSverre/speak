//! Context-dependent rules (spec §8.1.8): what a consonant looks like depends on the vowel
//! next to it. The numbers live in `tables::context`.

use crate::klatt::tables::context as t;
use crate::phone::{Class, Phone, Place};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VowelContext {
    Front,
    BackUnrounded,
    BackRounded,
}

impl VowelContext {
    fn index(self) -> usize {
        match self {
            VowelContext::Front => 0,
            VowelContext::BackUnrounded => 1,
            VowelContext::BackRounded => 2,
        }
    }
}

/// Context class of a vowel; glides and `R` colour a neighbouring consonant the same way.
pub fn class_of(p: Phone) -> Option<VowelContext> {
    use Phone::*;
    match p {
        IY | IH | EY | EH | AE | IX | Y => Some(VowelContext::Front),
        AA | AH | AX | AY | AW | ER | AXR | R => Some(VowelContext::BackUnrounded),
        AO | OW | UH | UW | OY | W => Some(VowelContext::BackRounded),
        _ => None,
    }
}

/// F2 and F3 locus of a stop or nasal next to a vowel of class `ctx`.
pub fn locus(p: Phone, ctx: VowelContext) -> Option<(f32, Option<f32>)> {
    if !matches!(p.class(), Class::Stop | Class::Nasal) {
        return None;
    }
    let i = ctx.index();
    match p.place() {
        Place::Labial => Some((t::LABIAL_F2[i], None)),
        Place::Alveolar => Some((t::ALVEOLAR_F2[i], None)),
        Place::Velar => Some((t::VELAR_F2[i], Some(t::VELAR_F3[i]))),
        _ => None,
    }
}

/// Parallel gains of a stop burst in context: a velar burst before a back vowel sits on F2.
pub fn burst_gains(p: Phone, ctx: Option<VowelContext>, a: [f32; 5]) -> [f32; 5] {
    let back = matches!(
        ctx,
        Some(VowelContext::BackUnrounded | VowelContext::BackRounded)
    );
    if p.place() == Place::Velar && back {
        let cut = |v: f32| (v - t::VELAR_BACK_CUT_DB).max(0.0);
        [t::VELAR_BACK_A2, cut(a[1]), cut(a[2]), cut(a[3]), cut(a[4])]
    } else {
        a
    }
}

/// `S` and `SH` before rounded vowels: the noise centre moves down one formant.
pub fn fricative_gains(p: Phone, ctx: Option<VowelContext>, a: [f32; 5]) -> [f32; 5] {
    if matches!(p, Phone::S | Phone::SH | Phone::Z | Phone::ZH)
        && ctx == Some(VowelContext::BackRounded)
    {
        [a[1], a[2], a[3], a[4], 0.0]
    } else {
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn velar_pinch_and_back_burst() {
        let front = locus(Phone::G, VowelContext::Front).unwrap();
        let round = locus(Phone::G, VowelContext::BackRounded).unwrap();
        assert!(front.0 - round.0 >= 600.0);
        // Before front vowels F2 and F3 start close together.
        assert!(front.1.unwrap() - front.0 < 700.0);
        let a = [0.0, 53.0, 43.0, 45.0, 45.0];
        assert_eq!(burst_gains(Phone::K, Some(VowelContext::Front), a), a);
        let back = burst_gains(Phone::K, Some(VowelContext::BackRounded), a);
        assert!(back[0] >= 55.0 && back[1] < a[1]);
        assert_eq!(burst_gains(Phone::T, Some(VowelContext::BackRounded), a), a);
        assert!(locus(Phone::S, VowelContext::Front).is_none());
    }

    #[test]
    fn s_lowers_before_rounded_vowels() {
        let s = [0.0, 0.0, 0.0, 0.0, 52.0];
        assert_eq!(
            fricative_gains(Phone::S, Some(VowelContext::BackRounded), s),
            [0.0, 0.0, 0.0, 52.0, 0.0]
        );
        assert_eq!(fricative_gains(Phone::S, Some(VowelContext::Front), s), s);
        assert_eq!(
            fricative_gains(Phone::F, Some(VowelContext::BackRounded), s),
            s
        );
    }
}

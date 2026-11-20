//! The phone inventory (spec §6.1) and the ARPAbet text format (spec §6.2).

use crate::chunk::{Segment, Unit};
use crate::error::{Error, Result};

macro_rules! phones {
    ($($name:ident),* $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(u8)]
        pub enum Phone { $($name),* }

        impl Phone {
            /// Every phone, silence last. The discriminant is the index.
            pub const ALL: &'static [Phone] = &[$(Phone::$name),*];

            pub fn name(self) -> &'static str {
                match self { $(Phone::$name => stringify!($name)),* }
            }
        }
    };
}

phones!(
    IY, IH, EY, EH, AE, AA, AO, AH, AX, IX, OW, UH, UW, ER, AXR, AY, AW, OY, P, B, T, D, K, G, CH,
    JH, F, V, TH, DH, S, Z, SH, ZH, HH, M, N, NG, L, R, W, Y, DX, Q, SIL,
);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    Vowel,
    Stop,
    Affricate,
    Fricative,
    Aspirate,
    Nasal,
    Liquid,
    Glide,
    Flap,
    Glottal,
    Silence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Place {
    Labial,
    Dental,
    Alveolar,
    Palatal,
    Velar,
    Glottal,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum Stress {
    #[default]
    None,
    Secondary,
    Primary,
}

impl Stress {
    pub fn digit(self) -> u8 {
        match self {
            Stress::None => 0,
            Stress::Primary => 1,
            Stress::Secondary => 2,
        }
    }

    pub fn from_digit(d: u8) -> Option<Stress> {
        match d {
            0 => Some(Stress::None),
            1 => Some(Stress::Primary),
            2 => Some(Stress::Secondary),
            _ => None,
        }
    }
}

impl Phone {
    pub const COUNT: usize = Phone::ALL.len();

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(i: usize) -> Option<Phone> {
        Phone::ALL.get(i).copied()
    }

    /// Text form: `_` for silence, the ARPAbet name otherwise.
    pub fn text(self) -> &'static str {
        if self == Phone::SIL { "_" } else { self.name() }
    }

    pub fn from_name(s: &str) -> Option<Phone> {
        if s == "_" || s == "SIL" {
            return Some(Phone::SIL);
        }
        Phone::ALL
            .iter()
            .copied()
            .find(|p| *p != Phone::SIL && p.name() == s)
    }

    pub fn class(self) -> Class {
        use Phone::*;
        match self {
            IY | IH | EY | EH | AE | AA | AO | AH | AX | IX | OW | UH | UW | ER | AXR | AY | AW
            | OY => Class::Vowel,
            P | B | T | D | K | G => Class::Stop,
            CH | JH => Class::Affricate,
            F | V | TH | DH | S | Z | SH | ZH => Class::Fricative,
            HH => Class::Aspirate,
            M | N | NG => Class::Nasal,
            L | R => Class::Liquid,
            W | Y => Class::Glide,
            DX => Class::Flap,
            Q => Class::Glottal,
            SIL => Class::Silence,
        }
    }

    pub fn place(self) -> Place {
        use Phone::*;
        match self {
            P | B | M | F | V | W => Place::Labial,
            TH | DH => Place::Dental,
            T | D | N | S | Z | L | DX | R => Place::Alveolar,
            CH | JH | SH | ZH | Y => Place::Palatal,
            K | G | NG => Place::Velar,
            HH | Q => Place::Glottal,
            _ => Place::None,
        }
    }

    pub fn is_vowel(self) -> bool {
        self.class() == Class::Vowel
    }

    pub fn is_silence(self) -> bool {
        self == Phone::SIL
    }

    pub fn is_consonant(self) -> bool {
        !self.is_vowel() && !self.is_silence()
    }

    /// Vowels, liquids, glides, nasals and the flap.
    pub fn is_sonorant(self) -> bool {
        matches!(
            self.class(),
            Class::Vowel | Class::Liquid | Class::Glide | Class::Nasal | Class::Flap
        )
    }

    pub fn is_voiced(self) -> bool {
        use Phone::*;
        !matches!(self, P | T | K | CH | F | TH | S | SH | HH | Q | SIL)
    }

    /// Stops, affricates and fricatives.
    pub fn is_obstruent(self) -> bool {
        matches!(
            self.class(),
            Class::Stop | Class::Affricate | Class::Fricative
        )
    }

    pub fn is_voiceless_stop(self) -> bool {
        matches!(self, Phone::P | Phone::T | Phone::K)
    }

    pub fn is_voiced_stop(self) -> bool {
        matches!(self, Phone::B | Phone::D | Phone::G)
    }

    pub fn is_reduced(self) -> bool {
        matches!(self, Phone::AX | Phone::IX | Phone::AXR)
    }

    pub fn is_diphthong(self) -> bool {
        matches!(
            self,
            Phone::EY | Phone::OW | Phone::AY | Phone::AW | Phone::OY
        )
    }
}

pub const PUNCTUATION: &[char] = &['.', ',', '?', '!', ';', ':'];

/// Splits `AE1` into (`AE`, stress). Consonants must not carry a digit.
pub fn parse_phone_token(tok: &str) -> Result<(Phone, Stress)> {
    let bad = || Error::usage(format!("unknown phoneme '{tok}'"));
    let (name, stress) = match tok.as_bytes().last() {
        Some(d @ b'0'..=b'9') => {
            let s = Stress::from_digit(d - b'0').ok_or_else(bad)?;
            (&tok[..tok.len() - 1], Some(s))
        }
        _ => (tok, None),
    };
    let upper = name.to_ascii_uppercase();
    let phone = Phone::from_name(&upper).ok_or_else(bad)?;
    if stress.is_some() && !phone.is_vowel() {
        return Err(bad());
    }
    Ok((phone, stress.unwrap_or_default()))
}

pub fn phone_token(phone: Phone, stress: Stress) -> String {
    if phone.is_vowel() && stress != Stress::None {
        format!("{}{}", phone.name(), stress.digit())
    } else {
        phone.text().to_string()
    }
}

/// Parses the phoneme text format of spec §6.2. Word indices count `/` separators.
pub fn parse_arpabet(text: &str) -> Result<Vec<Unit>> {
    let mut units = Vec::new();
    let mut word = 0u32;
    for tok in text.split_whitespace() {
        if tok == "/" {
            units.push(Unit::WordBreak);
            word += 1;
            continue;
        }
        let mut chars = tok.chars();
        if let (Some(c), None) = (chars.next(), chars.next())
            && PUNCTUATION.contains(&c)
        {
            units.push(Unit::Punct(c));
            continue;
        }
        let (phone, stress) = parse_phone_token(tok)?;
        if phone == Phone::SIL {
            units.push(Unit::Silence { ms: 0.0 });
        } else {
            units.push(Unit::Seg(Segment::new(phone, stress, word)));
        }
    }
    Ok(units)
}

pub fn print_arpabet(units: &[Unit]) -> String {
    let mut out = String::new();
    for u in units {
        if !out.is_empty() {
            out.push(' ');
        }
        match u {
            Unit::Seg(s) => out.push_str(&phone_token(s.phone, s.stress)),
            Unit::WordBreak => out.push('/'),
            Unit::Punct(c) => out.push(*c),
            Unit::Silence { .. } => out.push('_'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_size() {
        assert_eq!(Phone::COUNT, 45);
        for (i, p) in Phone::ALL.iter().enumerate() {
            assert_eq!(p.index(), i);
            assert_eq!(Phone::from_name(p.text()), Some(*p));
        }
    }

    #[test]
    fn round_trip() {
        let text = "HH AX L OW1 / W ER1 L D .";
        let units = parse_arpabet(text).unwrap();
        assert_eq!(print_arpabet(&units), text);
        assert_eq!(parse_arpabet(&print_arpabet(&units)).unwrap(), units);
    }

    #[test]
    fn round_trip_all_phones() {
        // Property: parse(print(x)) == x over every phone and stress.
        let mut units = Vec::new();
        for (i, p) in Phone::ALL.iter().enumerate() {
            if *p == Phone::SIL {
                units.push(Unit::Silence { ms: 0.0 });
                continue;
            }
            let stress = if p.is_vowel() {
                Stress::from_digit((i % 3) as u8).unwrap()
            } else {
                Stress::None
            };
            units.push(Unit::Seg(Segment::new(*p, stress, (i / 4) as u32)));
            if i % 4 == 3 {
                units.push(Unit::WordBreak);
            }
        }
        units.push(Unit::Punct('?'));
        assert_eq!(parse_arpabet(&print_arpabet(&units)).unwrap(), units);
    }

    #[test]
    fn rejects_bad_tokens() {
        assert!(parse_arpabet("XX").is_err());
        assert!(parse_arpabet("T1").is_err());
        assert!(parse_arpabet("AE7").is_err());
        assert!(parse_arpabet("ae1 t").is_ok());
    }
}

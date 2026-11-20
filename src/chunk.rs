//! The chunk: one intonation phrase travelling through the pipeline (spec §6.3).

use crate::phone::{Phone, Stress};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WordSource {
    User,
    Builtin,
    Cmudict,
    Cart,
    Rules,
    Spelled,
}

impl WordSource {
    pub const ALL: [WordSource; 6] = [
        WordSource::User,
        WordSource::Builtin,
        WordSource::Cmudict,
        WordSource::Cart,
        WordSource::Rules,
        WordSource::Spelled,
    ];

    pub fn name(self) -> &'static str {
        match self {
            WordSource::User => "user",
            WordSource::Builtin => "builtin",
            WordSource::Cmudict => "cmudict",
            WordSource::Cart => "cart",
            WordSource::Rules => "rules",
            WordSource::Spelled => "spelled",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    /// Byte span in the original input; (0, 0) for words made by an expansion.
    pub span: (usize, usize),
    pub source: WordSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Terminator {
    Period,
    Question,
    Exclaim,
    Comma,
    Semicolon,
    Colon,
    Dash,
    Paragraph,
    #[default]
    None,
}

impl Terminator {
    pub fn from_char(c: char) -> Terminator {
        match c {
            '.' | '…' => Terminator::Period,
            '?' => Terminator::Question,
            '!' => Terminator::Exclaim,
            ',' => Terminator::Comma,
            ';' => Terminator::Semicolon,
            ':' => Terminator::Colon,
            '—' => Terminator::Dash,
            _ => Terminator::None,
        }
    }

    pub fn symbol(self) -> &'static str {
        match self {
            Terminator::Period => ".",
            Terminator::Question => "?",
            Terminator::Exclaim => "!",
            Terminator::Comma => ",",
            Terminator::Semicolon => ";",
            Terminator::Colon => ":",
            Terminator::Dash => "—",
            Terminator::Paragraph => "¶",
            Terminator::None => "",
        }
    }

    /// Pause after a chunk with this terminator, before rate scaling (spec §7.3 rule 5).
    pub fn pause_ms(self) -> f32 {
        match self {
            Terminator::Period | Terminator::Question | Terminator::Exclaim => 350.0,
            Terminator::Semicolon | Terminator::Colon | Terminator::Dash => 250.0,
            Terminator::Comma => 150.0,
            Terminator::Paragraph => 700.0,
            Terminator::None => 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Controls {
    /// 1.0 = 175 wpm.
    pub rate: f32,
    pub pitch_base_hz: Option<f32>,
    pub pitch_shift_st: f32,
    pub volume: f32,
}

impl Default for Controls {
    fn default() -> Self {
        Controls {
            rate: 1.0,
            pitch_base_hz: None,
            pitch_shift_st: 0.0,
            volume: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub phone: Phone,
    pub stress: Stress,
    /// Index into `Chunk::words`.
    pub word: u32,
    /// 0.0 until prosody runs.
    pub dur_ms: f32,
    /// (position 0..=1 within the segment, Hz); empty until prosody runs.
    pub f0: Vec<(f32, f32)>,
    /// Duration multiplier set by postlexical rules (geminates, syllabic reduction).
    pub dur_scale: f32,
}

impl Segment {
    pub fn new(phone: Phone, stress: Stress, word: u32) -> Segment {
        Segment {
            phone,
            stress,
            word,
            dur_ms: 0.0,
            f0: Vec::new(),
            dur_scale: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Unit {
    Seg(Segment),
    WordBreak,
    Punct(char),
    Silence { ms: f32 },
}

impl Unit {
    pub fn seg(&self) -> Option<&Segment> {
        match self {
            Unit::Seg(s) => Some(s),
            _ => None,
        }
    }

    /// Duration this unit occupies in the audio.
    pub fn dur_ms(&self) -> f32 {
        match self {
            Unit::Seg(s) => s.dur_ms,
            Unit::Silence { ms } => *ms,
            _ => 0.0,
        }
    }
}

/// What the chunker hands to the G2P: the chunk's content before phones exist.
#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    /// Index into `Chunk::words`; `spell` = say the letter name.
    Word {
        index: usize,
        spell: bool,
    },
    Punct(char),
    Silence(f32),
    /// Inline ARPAbet.
    Phonemes(String),
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Chunk {
    pub index: usize,
    /// Normalized text of this chunk.
    pub text: String,
    pub words: Vec<Word>,
    /// Filled by the chunker, consumed by the G2P.
    pub items: Vec<Item>,
    pub units: Vec<Unit>,
    pub terminator: Terminator,
    pub pause_after_ms: f32,
    pub controls: Controls,
    /// True when durations and F0 came from the input (`--phonemes=pho`): prosody is skipped.
    pub timed: bool,
}

impl Chunk {
    pub fn segments(&self) -> impl Iterator<Item = &Segment> {
        self.units.iter().filter_map(Unit::seg)
    }

    /// Sum of all unit durations, without the pause after the chunk.
    pub fn dur_ms(&self) -> f32 {
        self.units.iter().map(Unit::dur_ms).sum()
    }
}

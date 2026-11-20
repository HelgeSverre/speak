//! Chunker: token stream → chunks, one sentence per chunk (spec §7.3).

use crate::chunk::{Chunk, Controls, Item, Terminator, Word, WordSource};
use crate::traits::{EmbeddedCommand, Token};

pub const MAX_WORDS: usize = 60;

struct Builder {
    chunks: Vec<Chunk>,
    cur: Chunk,
    controls: Controls,
}

impl Builder {
    fn word_count(&self) -> usize {
        self.cur.words.len()
    }

    fn has_content(&self) -> bool {
        self.cur.items.iter().any(|i| !matches!(i, Item::Punct(_)))
    }

    fn flush(&mut self, terminator: Terminator, pause_ms: f32) {
        if !self.has_content() {
            self.cur = Chunk::default();
            return;
        }
        let mut c = std::mem::take(&mut self.cur);
        c.index = self.chunks.len();
        c.terminator = terminator;
        c.controls = self.controls;
        c.pause_after_ms = pause_ms / self.controls.rate.max(0.1);
        c.text = text_of(&c);
        self.chunks.push(c);
    }

    /// Rule 2: split an over-long sentence at the last `; : —`, else the last `,`, else here.
    fn split_long(&mut self) {
        let find = |marks: &[char]| {
            self.cur
                .items
                .iter()
                .rposition(|i| matches!(i, Item::Punct(c) if marks.contains(c)))
        };
        let at = find(&[';', ':', '—']).or_else(|| find(&[',']));
        match at {
            Some(pos) if pos + 1 < self.cur.items.len() => {
                let mark = match self.cur.items[pos] {
                    Item::Punct(c) => c,
                    _ => ',',
                };
                let tail_items = self.cur.items.split_off(pos + 1);
                // Words referenced by the tail move with it.
                let first_tail_word = tail_items.iter().find_map(|i| {
                    if let Item::Word { index, .. } = i {
                        Some(*index)
                    } else {
                        None
                    }
                });
                let tail_words = match first_tail_word {
                    Some(w) => self.cur.words.split_off(w),
                    None => Vec::new(),
                };
                let shift = first_tail_word.unwrap_or(0);
                let terminator = Terminator::from_char(mark);
                self.flush(terminator, terminator.pause_ms());
                self.cur.words = tail_words;
                self.cur.items = tail_items
                    .into_iter()
                    .map(|i| match i {
                        Item::Word { index, spell } => Item::Word {
                            index: index - shift,
                            spell,
                        },
                        other => other,
                    })
                    .collect();
            }
            _ => self.flush(Terminator::None, 0.0),
        }
    }
}

fn text_of(c: &Chunk) -> String {
    let mut s = String::new();
    for item in &c.items {
        match item {
            Item::Word { index, .. } => {
                if !s.is_empty() {
                    s.push(' ');
                }
                s.push_str(&c.words[*index].text);
            }
            Item::Punct(p) => {
                if *p == '—' {
                    s.push(' ');
                }
                s.push(*p);
            }
            Item::Phonemes(p) => {
                if !s.is_empty() {
                    s.push(' ');
                }
                s.push_str(&format!("[[{p}]]"));
            }
            Item::Silence(_) => {}
        }
    }
    s
}

fn is_capitalized(t: Option<&Token>) -> bool {
    matches!(t, Some(Token::Word { text, .. }) if text.chars().next().is_some_and(|c| c.is_uppercase()))
}

pub fn chunk(tokens: &[Token], controls: Controls) -> Vec<Chunk> {
    let mut b = Builder {
        chunks: Vec::new(),
        cur: Chunk::default(),
        controls,
    };
    for (i, t) in tokens.iter().enumerate() {
        match t {
            Token::Word { text, span, spell } => {
                if b.word_count() >= MAX_WORDS {
                    b.split_long();
                }
                b.cur.words.push(Word {
                    text: text.clone(),
                    span: *span,
                    source: WordSource::Rules,
                });
                let index = b.cur.words.len() - 1;
                b.cur.items.push(Item::Word {
                    index,
                    spell: *spell,
                });
            }
            Token::Punct(c) => {
                if matches!(c, '.' | '!' | '?' | '…') {
                    b.cur.items.push(Item::Punct(*c));
                    let terminator = Terminator::from_char(*c);
                    b.flush(terminator, terminator.pause_ms());
                } else if b.has_content() {
                    b.cur.items.push(Item::Punct(*c));
                }
            }
            Token::PhonemeLiteral(p) => b.cur.items.push(Item::Phonemes(p.clone())),
            Token::Command(EmbeddedCommand::Silence(ms)) => b.cur.items.push(Item::Silence(*ms)),
            Token::Command(cmd) => {
                // A control change starts a new chunk (rule 4).
                b.flush(Terminator::None, 0.0);
                match cmd {
                    EmbeddedCommand::Rate(wpm) => b.controls.rate = wpm / crate::DEFAULT_WPM,
                    EmbeddedCommand::Volume(v) => b.controls.volume = *v,
                    EmbeddedCommand::PitchBase(hz) => b.controls.pitch_base_hz = Some(*hz),
                    EmbeddedCommand::Silence(_) => {}
                }
            }
            Token::ParagraphBreak => {
                b.flush(Terminator::Paragraph, Terminator::Paragraph.pause_ms())
            }
            Token::Newline => {
                // A line that ends without punctuation, followed by a capital, ends a sentence.
                let after_punct = matches!(b.cur.items.last(), Some(Item::Punct(_)) | None);
                if !after_punct && is_capitalized(tokens.get(i + 1)) {
                    b.flush(Terminator::Period, Terminator::Period.pause_ms());
                }
            }
        }
    }
    // A trailing clause mark decides the last chunk's tune.
    let last = match b.cur.items.last() {
        Some(Item::Punct(c)) => Terminator::from_char(*c),
        _ => Terminator::None,
    };
    b.flush(last, last.pause_ms());
    b.chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::EnglishNormalizer;
    use crate::traits::Normalizer;

    fn chunks(text: &str) -> Vec<Chunk> {
        let n = EnglishNormalizer::new(Box::new(|_| false));
        chunk(&n.normalize(text), Controls::default())
    }

    fn texts(text: &str) -> Vec<String> {
        chunks(text).into_iter().map(|c| c.text).collect()
    }

    #[test]
    fn sentences_and_terminators() {
        let c = chunks("First sentence. Second one, with a clause! Third?");
        assert_eq!(c.len(), 3);
        assert_eq!(c[0].terminator, Terminator::Period);
        assert_eq!(c[1].terminator, Terminator::Exclaim);
        assert_eq!(c[2].terminator, Terminator::Question);
        assert_eq!(c[1].text, "Second one, with a clause!");
        assert_eq!(c[0].pause_after_ms, 350.0);
        assert!(c[1].items.contains(&Item::Punct(',')));
    }

    #[test]
    fn abbreviations_decimals_and_initials_do_not_split() {
        assert_eq!(
            texts("Dr. Smith saw 3.14 e.g. pi. Next."),
            [
                "doctor Smith saw three point one four for example pi.",
                "Next."
            ]
        );
        assert_eq!(texts("J. K. Rowling wrote it."), ["j k Rowling wrote it."]);
        assert_eq!(
            texts("We met at 5 p.m. Then we left."),
            ["We met at five p m.", "Then we left."]
        );
        assert_eq!(texts("The U.S. Senate voted."), ["The u s Senate voted."]);
    }

    #[test]
    fn quotes_and_ellipsis() {
        assert_eq!(
            texts("\"Stop!\" she said. Wait... go."),
            ["Stop!", "she said.", "Wait…", "go."]
        );
    }

    #[test]
    fn newlines() {
        assert_eq!(texts("Line one\nLine two"), ["Line one", "Line two"]);
        assert_eq!(texts("wrapped\ntext here."), ["wrapped text here."]);
        let c = chunks("Para one\n\npara two");
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].terminator, Terminator::Paragraph);
        assert_eq!(c[0].pause_after_ms, 700.0);
    }

    #[test]
    fn long_sentences_split_at_clause_marks() {
        let mut text = String::new();
        for i in 0..100 {
            text.push_str("word ");
            if i == 40 {
                text.push_str("mark; ");
            }
            if i == 50 {
                text.push_str("comma, ");
            }
        }
        let c = chunks(&text);
        assert!(c.len() >= 2);
        assert!(c.iter().all(|c| c.words.len() <= MAX_WORDS));
        assert_eq!(c[0].terminator, Terminator::Semicolon);
        assert_eq!(c.iter().map(|c| c.words.len()).sum::<usize>(), 102);
        // Word indices are valid after the split.
        for ch in &c {
            for item in &ch.items {
                if let Item::Word { index, .. } = item {
                    assert!(*index < ch.words.len());
                }
            }
        }
        let c = chunks(&"word ".repeat(130));
        assert_eq!(c.len(), 3);
        assert_eq!(c[0].pause_after_ms, 0.0);
    }

    #[test]
    fn commands_split_and_update_controls() {
        let c = chunks("Slow part [[rate 350]] fast part. [[volm 0.5]] quiet [[slnc 400]] end.");
        assert_eq!(c.len(), 3);
        assert_eq!(c[0].controls.rate, 1.0);
        assert_eq!(c[1].controls.rate, 2.0);
        assert_eq!(c[2].controls.volume, 0.5);
        assert!(c[2].items.contains(&Item::Silence(400.0)));
        // Pauses scale with rate.
        assert_eq!(c[1].pause_after_ms, 175.0);
    }

    #[test]
    fn empty_input() {
        assert!(chunks("").is_empty());
        assert!(chunks(" ... !? ").is_empty());
    }
}

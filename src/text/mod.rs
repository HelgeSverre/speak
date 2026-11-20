//! Text frontend: normalizer, chunker, G2P, postlexical rules, prosody (spec §7).

pub mod abbrev;
pub mod chunker;
pub mod commands;
pub mod funcwords;
pub mod g2p;
pub mod lexicon;
pub mod lts;
pub mod normalize;
pub mod numbers;
pub mod postlexical;
pub mod prosody;
pub mod sing;
pub mod units;

use std::path::Path;
use std::sync::Arc;

pub use g2p::LexiconG2p;
pub use lexicon::Lexicons;
pub use normalize::EnglishNormalizer;
pub use prosody::RuleProsody;

use crate::chunk::{Chunk, Controls, WordSource};
use crate::error::{Error, Result};
use crate::traits::{G2p, Normalizer, Token};

/// The whole text frontend up to (not including) prosody.
pub struct Frontend {
    pub lexicons: Arc<Lexicons>,
    normalizer: EnglishNormalizer,
    g2p: LexiconG2p,
}

impl Frontend {
    /// `lts`: `cart` (the default when this build has a tree) or `rules`.
    /// `data_dir` holds the optional compiled CMUdict.
    pub fn new(
        user_lexicon: Option<&Path>,
        data_dir: Option<&Path>,
        lts: Option<&str>,
    ) -> Result<Frontend> {
        let lexicons = Arc::new(Lexicons::load(user_lexicon, data_dir));
        type Lts = (Box<dyn crate::traits::LetterToSound>, WordSource);
        let rules = || -> Lts { (Box::new(lts::RuleLts::new()), WordSource::Rules) };
        let (lts_impl, source): Lts = match lts {
            Some("rules") => rules(),
            Some("cart") | None => match lts::CartLts::embedded() {
                Some(c) => (Box::new(c), WordSource::Cart),
                // No tree in this build: the rules are the fallback.
                None => rules(),
            },
            Some(other) => {
                return Err(Error::usage(format!(
                    "-P lts={other}: expected rules or cart"
                )));
            }
        };
        let known = lexicons.clone();
        Ok(Frontend {
            normalizer: EnglishNormalizer::new(Box::new(move |w| known.contains(w))),
            g2p: LexiconG2p::new(lexicons.clone(), lts_impl, source),
            lexicons,
        })
    }

    pub fn normalize(&self, text: &str) -> Vec<Token> {
        self.normalizer.normalize(text)
    }

    /// Characters the normalizer dropped so far.
    pub fn dropped(&self) -> usize {
        self.normalizer.dropped()
    }

    /// normalize → chunk → G2P → postlexical.
    pub fn chunks(&self, text: &str, controls: Controls) -> Result<Vec<Chunk>> {
        let tokens = self.normalize(text);
        let mut chunks = chunker::chunk(&tokens, controls);
        for c in &mut chunks {
            self.g2p.phonemize(c)?;
            postlexical::postlexical(c);
        }
        Ok(chunks)
    }
}

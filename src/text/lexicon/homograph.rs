//! Homographs: words with two pronunciations and a context rule written by us (spec §7.4).
//! No part-of-speech tagger: the rules look at neighbouring words.

use crate::text::lexicon::{Pron, parse_pron};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Rule {
    /// The alternative is the verb: after `to`, a pronoun or a modal.
    Verb,
    /// `read`: past tense after have/has/had/was/were/already…, or with a past-time word in the sentence.
    Past,
    /// The alternative applies after one of these words.
    After(&'static [&'static str]),
}

struct Homograph {
    word: &'static str,
    default: &'static str,
    alt: &'static str,
    rule: Rule,
}

const VERB_CUES: &[&str] = &[
    "to", "i", "you", "we", "they", "he", "she", "it", "will", "would", "can", "could", "shall",
    "should", "may", "might", "must", "don't", "doesn't", "didn't", "won't", "can't", "not",
    "please", "let's", "who", "that",
];
const PAST_CUES: &[&str] = &[
    "have", "has", "had", "was", "were", "been", "be", "being", "already", "is", "are", "i've",
    "you've", "we've", "they've", "having", "got",
];
const PAST_TIME: &[&str] = &["yesterday", "ago", "earlier", "once", "before"];

const fn h(word: &'static str, default: &'static str, alt: &'static str, rule: Rule) -> Homograph {
    Homograph {
        word,
        default,
        alt,
        rule,
    }
}

const TABLE: &[Homograph] = &[
    h("read", "R IY1 D", "R EH1 D", Rule::Past),
    h("record", "R EH1 K AXR D", "R IX K AO1 R D", Rule::Verb),
    h(
        "present",
        "P R EH1 Z AX N T",
        "P R IX Z EH1 N T",
        Rule::Verb,
    ),
    h("object", "AA1 B JH EH K T", "AX B JH EH1 K T", Rule::Verb),
    h(
        "project",
        "P R AA1 JH EH K T",
        "P R AX JH EH1 K T",
        Rule::Verb,
    ),
    h("produce", "P R OW1 D UW S", "P R AX D UW1 S", Rule::Verb),
    h("permit", "P ER1 M IX T", "P AXR M IH1 T", Rule::Verb),
    h(
        "conduct",
        "K AA1 N D AX K T",
        "K AX N D AH1 K T",
        Rule::Verb,
    ),
    h(
        "contract",
        "K AA1 N T R AE K T",
        "K AX N T R AE1 K T",
        Rule::Verb,
    ),
    h(
        "contest",
        "K AA1 N T EH S T",
        "K AX N T EH1 S T",
        Rule::Verb,
    ),
    h("convert", "K AA1 N V AXR T", "K AX N V ER1 T", Rule::Verb),
    h("desert", "D EH1 Z AXR T", "D IX Z ER1 T", Rule::Verb),
    h("increase", "IH1 N K R IY S", "IX N K R IY1 S", Rule::Verb),
    h("decrease", "D IY1 K R IY S", "D IX K R IY1 S", Rule::Verb),
    h("insult", "IH1 N S AX L T", "IX N S AH1 L T", Rule::Verb),
    h("rebel", "R EH1 B AX L", "R IX B EH1 L", Rule::Verb),
    h("refuse", "R EH1 F Y UW S", "R IX F Y UW1 Z", Rule::Verb),
    h(
        "subject",
        "S AH1 B JH IX K T",
        "S AX B JH EH1 K T",
        Rule::Verb,
    ),
    h(
        "suspect",
        "S AH1 S P EH K T",
        "S AX S P EH1 K T",
        Rule::Verb,
    ),
    h("import", "IH1 M P AO R T", "IX M P AO1 R T", Rule::Verb),
    h("export", "EH1 K S P AO R T", "IX K S P AO1 R T", Rule::Verb),
    h(
        "separate",
        "S EH1 P R AX T",
        "S EH1 P AXR EY2 T",
        Rule::Verb,
    ),
    h(
        "estimate",
        "EH1 S T AX M AX T",
        "EH1 S T AX M EY2 T",
        Rule::Verb,
    ),
    h(
        "duplicate",
        "D UW1 P L AX K AX T",
        "D UW1 P L AX K EY2 T",
        Rule::Verb,
    ),
    h(
        "use",
        "Y UW1 Z",
        "Y UW1 S",
        Rule::After(&[
            "the", "a", "of", "no", "in", "any", "its", "this", "that", "good", "little", "much",
            "for",
        ]),
    ),
    h(
        "close",
        "K L OW1 Z",
        "K L OW1 S",
        Rule::After(&[
            "is", "was", "are", "were", "very", "so", "too", "this", "that", "how", "as", "stay",
            "get", "getting", "come", "came", "pretty", "really",
        ]),
    ),
    h(
        "live",
        "L IH1 V",
        "L AY1 V",
        Rule::After(&[
            "a", "the", "is", "was", "go", "going", "went", "goes", "are", "be", "in",
        ]),
    ),
    h(
        "lives",
        "L IH1 V Z",
        "L AY1 V Z",
        Rule::After(&[
            "their", "our", "many", "the", "of", "your", "nine", "two", "saving", "save", "saves",
            "saved",
        ]),
    ),
    h(
        "lead",
        "L IY1 D",
        "L EH1 D",
        Rule::After(&["of", "with", "like", "contains", "contain", "made"]),
    ),
    h("wind", "W IH1 N D", "W AY1 N D", Rule::Verb),
    h("tear", "T IH1 R", "T EH1 R", Rule::Verb),
    h(
        "bass",
        "B EY1 S",
        "B AE1 S",
        Rule::After(&["sea", "striped", "largemouth", "caught", "fishing"]),
    ),
    h(
        "minute",
        "M IH1 N AX T",
        "M AY N UW1 T",
        Rule::After(&["very", "extremely"]),
    ),
];

/// The pronunciation of `lower` in context, if it is a homograph.
/// `prev`: the word before it (lowercase), `sentence`: all words of the chunk (lowercase).
pub fn pronounce(lower: &str, prev: Option<&str>, sentence: &[String]) -> Option<Pron> {
    let entry = TABLE.iter().find(|e| e.word == lower)?;
    let alt = match entry.rule {
        Rule::Verb => prev.is_some_and(|p| VERB_CUES.contains(&p)),
        Rule::Past => {
            prev.is_some_and(|p| PAST_CUES.contains(&p))
                || sentence.iter().any(|w| PAST_TIME.contains(&w.as_str()))
        }
        Rule::After(cues) => prev.is_some_and(|p| cues.contains(&p)),
    };
    parse_pron(if alt { entry.alt } else { entry.default })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_entry_parses() {
        for e in TABLE {
            assert!(
                parse_pron(e.default).is_some() && parse_pron(e.alt).is_some(),
                "{}",
                e.word
            );
        }
        assert!(TABLE.len() >= 30);
    }

    #[test]
    fn context_rules() {
        let none: Vec<String> = Vec::new();
        let say = |w: &str, prev: Option<&str>, sent: &[String]| {
            pronounce(w, prev, sent)
                .unwrap()
                .iter()
                .map(|p| p.0.name())
                .collect::<Vec<_>>()
                .join(" ")
        };
        assert_eq!(
            say(
                "read",
                Some("i"),
                &["i".into(), "read".into(), "it".into(), "yesterday".into()]
            ),
            "R EH D"
        );
        assert_eq!(say("read", Some("will"), &none), "R IY D");
        assert_eq!(say("read", Some("have"), &none), "R EH D");
        assert_eq!(say("record", Some("to"), &none), "R IX K AO R D");
        assert_eq!(say("record", Some("the"), &none), "R EH K AXR D");
        assert_eq!(say("use", Some("the"), &none), "Y UW S");
        assert_eq!(say("use", Some("to"), &none), "Y UW Z");
        assert_eq!(say("close", Some("very"), &none), "K L OW S");
        assert!(pronounce("cat", None, &none).is_none());
    }
}

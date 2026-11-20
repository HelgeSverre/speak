//! Normalizer table test (spec §16.3).

use speak::text::EnglishNormalizer;
use speak::text::normalize::tokens_to_text;
use speak::traits::{Normalizer, Token};

fn normalizer() -> EnglishNormalizer {
    EnglishNormalizer::new(Box::new(|w| {
        matches!(w, "nasa" | "ok" | "com" | "org" | "the" | "a" | "is")
    }))
}

#[test]
fn table() {
    let n = normalizer();
    let cases = include_str!("fixtures/normalize_cases.tsv");
    let mut count = 0;
    let mut failures = Vec::new();
    for line in cases.lines() {
        // "# 5" is a case; "# text" without a tab is a comment.
        let Some((input, want)) = line.split_once('\t') else {
            continue;
        };
        count += 1;
        let got = tokens_to_text(&n.normalize(input));
        if got != want {
            failures.push(format!("{input:?}\n   want {want:?}\n    got {got:?}"));
        }
    }
    assert!(count >= 120, "only {count} cases");
    assert!(
        failures.is_empty(),
        "{} of {count} failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn never_emits_digits_and_counts_drops() {
    let n = normalizer();
    let text =
        "qzxwv blorptastic 1969 4x4 a1b2 99problems 3.14.15 1-2-3 12:99 5/0 $ 7,00 😀 2024-13-45 ٣";
    for t in n.normalize(text) {
        if let Token::Word { text, .. } = t {
            assert!(
                !text.chars().any(|c| c.is_ascii_digit()),
                "digit in {text:?}"
            );
            assert!(!text.is_empty());
        }
    }
    assert!(n.dropped() >= 1);
}

#[test]
fn spans_point_into_the_input() {
    let n = normalizer();
    let input = "Dr. Smith paid $3.50 today";
    let tokens = n.normalize(input);
    let Token::Word { text, span, .. } = &tokens[1] else {
        panic!()
    };
    assert_eq!(text, "Smith");
    assert_eq!(&input[span.0..span.1], "Smith");
    let Token::Word { span, .. } = &tokens[3] else {
        panic!()
    };
    assert_eq!(&input[span.0..span.1], "$3.50");
}

#[test]
fn embedded_commands() {
    use speak::traits::EmbeddedCommand;
    let n = normalizer();
    let t = n.normalize("One.[[slnc 500]] Two [[rate 300]] three [[inpt PHON]]HH AX L OW1[[inpt TEXT]] four [[ W ER1 L D ]]");
    assert!(t.contains(&Token::Command(EmbeddedCommand::Silence(500.0))));
    assert!(t.contains(&Token::Command(EmbeddedCommand::Rate(300.0))));
    assert!(t.contains(&Token::PhonemeLiteral("HH AX L OW1".into())));
    assert!(t.contains(&Token::PhonemeLiteral("W ER1 L D".into())));
    let words: Vec<_> = t
        .iter()
        .filter_map(|t| {
            if let Token::Word { text, .. } = t {
                Some(text.as_str())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(words, ["One", "Two", "three", "four"]);
}

#[test]
fn newlines() {
    let n = normalizer();
    let t = n.normalize("Line one\nLine two\n\nNew paragraph");
    assert_eq!(t.iter().filter(|t| **t == Token::Newline).count(), 1);
    assert_eq!(t.iter().filter(|t| **t == Token::ParagraphBreak).count(), 1);
}

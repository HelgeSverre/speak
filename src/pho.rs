//! The `.pho` text format: timed phones, the seam between frontend and backends (spec §6.4).

use crate::chunk::{Chunk, Controls, Segment, Terminator, Unit};
use crate::error::{Error, Result};
use crate::phone::{Phone, parse_arpabet, parse_phone_token, phone_token, print_arpabet};

/// Prints one chunk. Word breaks and punctuation carry no duration and are not printed.
pub fn print_pho(chunk: &Chunk, out: &mut String) {
    out.push_str(&format!("; chunk {}  \"{}\"\n", chunk.index, chunk.text));
    for u in &chunk.units {
        match u {
            Unit::Seg(s) => {
                out.push_str(&format!(
                    "{:<4} {:.0}",
                    phone_token(s.phone, s.stress),
                    s.dur_ms
                ));
                for (pos, hz) in &s.f0 {
                    out.push_str(&format!("  {:.0} {:.0}", pos * 100.0, hz));
                }
                out.push('\n');
            }
            Unit::Silence { ms } if *ms > 0.0 => out.push_str(&format!("_    {ms:.0}\n")),
            _ => {}
        }
    }
    if chunk.pause_after_ms > 0.0 {
        out.push_str(&format!("_    {:.0}\n", chunk.pause_after_ms));
    }
}

/// Parses a `.pho` document into chunks. A `; chunk` comment starts a new chunk. `|` also
/// ends a line, so a whole document fits on one command line.
pub fn parse_pho(text: &str) -> Result<Vec<Chunk>> {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut cur: Option<Chunk> = None;
    let start = |index: usize, text: String| Chunk {
        index,
        text,
        timed: true,
        ..Chunk::default()
    };
    let lines = text.lines().flat_map(|l| {
        let comment = l.trim_start().starts_with(';');
        l.split(move |c| c == '|' && !comment)
    });
    for (n, raw) in lines.enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix(';') {
            if comment.trim_start().starts_with("chunk") {
                if let Some(c) = cur.take() {
                    chunks.push(c);
                }
                let label = comment.split('"').nth(1).unwrap_or("").to_string();
                cur = Some(start(chunks.len(), label));
            }
            continue;
        }
        let line = line.split(';').next().unwrap_or("");
        let bad = |why: &str| Error::usage(format!("pho line {}: {why}: '{raw}'", n + 1));
        let mut it = line.split_whitespace();
        let tok = it.next().ok_or_else(|| bad("missing phone"))?;
        let (phone, stress) = parse_phone_token(tok).map_err(|_| bad("unknown phone"))?;
        let dur: f32 = it
            .next()
            .and_then(|d| d.parse().ok())
            .filter(|d: &f32| d.is_finite() && *d >= 0.0)
            .ok_or_else(|| bad("bad duration"))?;
        let rest: Vec<f32> = it
            .map(|v| v.parse::<f32>().map_err(|_| bad("bad F0 target")))
            .collect::<Result<_>>()?;
        if !rest.len().is_multiple_of(2) {
            return Err(bad("F0 targets must be position/Hz pairs"));
        }
        let chunk = cur.get_or_insert_with(|| start(chunks.len(), String::new()));
        if phone == Phone::SIL {
            chunk.units.push(Unit::Silence { ms: dur });
        } else {
            let mut seg = Segment::new(phone, stress, 0);
            seg.dur_ms = dur;
            seg.f0 = rest
                .chunks(2)
                .map(|p| ((p[0] / 100.0).clamp(0.0, 1.0), p[1]))
                .collect();
            chunk.units.push(Unit::Seg(seg));
        }
    }
    if let Some(c) = cur.take() {
        chunks.push(c);
    }
    Ok(chunks)
}

/// Chunks from ARPAbet text: one per sentence mark.
pub fn arpabet_chunks(text: &str, controls: Controls) -> Result<Vec<Chunk>> {
    let units = parse_arpabet(text)?;
    let mut chunks = Vec::new();
    let mut cur: Vec<Unit> = Vec::new();
    let mut flush = |cur: &mut Vec<Unit>, terminator: Terminator| {
        while matches!(cur.first(), Some(Unit::WordBreak)) {
            cur.remove(0);
        }
        if cur
            .iter()
            .any(|u| matches!(u, Unit::Seg(_) | Unit::Silence { .. }))
        {
            chunks.push(Chunk {
                index: chunks.len(),
                text: print_arpabet(cur),
                units: std::mem::take(cur),
                terminator,
                pause_after_ms: terminator.pause_ms() / controls.rate,
                controls,
                ..Chunk::default()
            });
        }
        cur.clear();
    };
    for u in units {
        let mark = if let Unit::Punct(c) = u {
            Some(c)
        } else {
            None
        };
        cur.push(u);
        if let Some(c @ ('.' | '?' | '!')) = mark {
            flush(&mut cur, Terminator::from_char(c));
        }
    }
    let last = match cur.last() {
        Some(Unit::Punct(c)) => Terminator::from_char(*c),
        _ => Terminator::None,
    };
    flush(&mut cur, last);
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phone::Stress;

    #[test]
    fn bars_separate_lines() {
        let one =
            parse_pho("; chunk 0  \"a | b\"\nHH 50 | AA1 120  0 110  100 100 | _ 60").unwrap();
        let many = parse_pho("HH 50\nAA1 120  0 110  100 100\n_ 60").unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].text, "a | b");
        assert_eq!(one[0].units, many[0].units);
    }

    #[test]
    fn round_trip() {
        let mut chunk = Chunk {
            text: "hello".into(),
            ..Chunk::default()
        };
        let mut a = Segment::new(Phone::HH, Stress::None, 0);
        a.dur_ms = 62.0;
        let mut b = Segment::new(Phone::OW, Stress::Primary, 0);
        b.dur_ms = 182.0;
        b.f0 = vec![(0.0, 124.0), (0.4, 131.0), (1.0, 112.0)];
        chunk.units = vec![
            Unit::Seg(a),
            Unit::WordBreak,
            Unit::Seg(b),
            Unit::Punct('.'),
        ];
        chunk.pause_after_ms = 350.0;
        let mut text = String::new();
        print_pho(&chunk, &mut text);
        let parsed = parse_pho(&text).unwrap();
        assert_eq!(parsed.len(), 1);
        let segs: Vec<_> = parsed[0].segments().cloned().collect();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[1].phone, Phone::OW);
        assert_eq!(segs[1].stress, Stress::Primary);
        assert_eq!(segs[1].f0, vec![(0.0, 124.0), (0.4, 131.0), (1.0, 112.0)]);
        assert!((parsed[0].dur_ms() - 594.0).abs() < 1e-3);
        let mut again = String::new();
        print_pho(&parsed[0], &mut again);
        assert_eq!(parse_pho(&again).unwrap()[0].units, parsed[0].units);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_pho("ZZ 10").is_err());
        assert!(parse_pho("AA x").is_err());
        assert!(parse_pho("AA 100 50").is_err());
    }
}

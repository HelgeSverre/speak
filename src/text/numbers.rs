//! Number expansion (spec §7.1). Our own code: cardinals to 10¹⁵, ordinals, years, decimals.

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const SCALES: [(u64, &str); 5] = [
    (1_000_000_000_000_000, "quadrillion"),
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
];

fn below_thousand(n: u64, out: &mut Vec<String>) {
    let (h, rest) = (n / 100, n % 100);
    if h > 0 {
        out.push(ONES[h as usize].into());
        out.push("hundred".into());
    }
    if rest >= 20 {
        out.push(TENS[(rest / 10) as usize].into());
        if rest % 10 > 0 {
            out.push(ONES[(rest % 10) as usize].into());
        }
    } else if rest > 0 || n == 0 {
        out.push(ONES[rest as usize].into());
    }
}

/// `1234` → one thousand two hundred thirty four.
pub fn cardinal(n: u64) -> Vec<String> {
    let mut out = Vec::new();
    if n == 0 {
        return vec!["zero".into()];
    }
    let mut rest = n;
    for (scale, name) in SCALES {
        if rest >= scale {
            // Above 999 quadrillion the leading group is itself expanded.
            let lead = rest / scale;
            if lead >= 1000 {
                out.extend(cardinal(lead));
            } else {
                below_thousand(lead, &mut out);
            }
            out.push(name.into());
            rest %= scale;
        }
    }
    if rest > 0 {
        below_thousand(rest, &mut out);
    }
    out
}

fn ordinal_word(card: &str) -> String {
    match card {
        "one" => "first".into(),
        "two" => "second".into(),
        "three" => "third".into(),
        "five" => "fifth".into(),
        "eight" => "eighth".into(),
        "nine" => "ninth".into(),
        "twelve" => "twelfth".into(),
        w if w.ends_with('y') => format!("{}ieth", &w[..w.len() - 1]),
        w => format!("{w}th"),
    }
}

/// `21` → twenty first.
pub fn ordinal(n: u64) -> Vec<String> {
    let mut words = cardinal(n);
    if let Some(last) = words.last_mut() {
        *last = ordinal_word(last);
    }
    words
}

/// Each digit by name. The fallback that never drops a number.
pub fn digits(s: &str) -> Vec<String> {
    s.chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| ONES[d as usize].to_string())
        .collect()
}

/// A digit string of any length: cardinal when it fits, digit by digit otherwise.
pub fn cardinal_str(s: &str) -> Vec<String> {
    match s.parse::<u64>() {
        Ok(n) if s.len() <= 16 && !(s.len() > 1 && s.starts_with('0')) => cardinal(n),
        _ => digits(s),
    }
}

/// Years 1100–2099: nineteen eighty four, two thousand five, twenty twenty six.
pub fn year(n: u64) -> Vec<String> {
    let (hi, lo) = (n / 100, n % 100);
    if (2000..2010).contains(&n) || n.is_multiple_of(1000) {
        return cardinal(n);
    }
    let mut out = cardinal(hi);
    if lo == 0 {
        out.push("hundred".into());
    } else if lo < 10 {
        out.push("oh".into());
        out.push(ONES[lo as usize].into());
    } else {
        out.extend(cardinal(lo));
    }
    out
}

/// `3.14` → three point one four. `int` and `frac` are digit strings.
pub fn decimal(int: &str, frac: &str) -> Vec<String> {
    let mut out = if int.is_empty() {
        vec!["zero".to_string()]
    } else {
        cardinal_str(int)
    };
    out.push("point".into());
    out.extend(digits(frac));
    out
}

/// Two digits of a clock time or of cents: `05` → oh five.
pub fn two_digit(n: u64, leading_oh: bool) -> Vec<String> {
    if n < 10 && leading_oh {
        vec!["oh".into(), ONES[n as usize].into()]
    } else {
        cardinal(n)
    }
}

/// `3/4` → three quarters.
pub fn fraction(num: u64, den: u64) -> Vec<String> {
    let mut out = cardinal(num);
    let plural = num != 1;
    match den {
        2 => out.push(if plural {
            "halves".into()
        } else {
            "half".into()
        }),
        4 => out.push(if plural {
            "quarters".into()
        } else {
            "quarter".into()
        }),
        _ => {
            let mut d = ordinal(den);
            if plural && let Some(last) = d.last_mut() {
                last.push('s');
            }
            out.extend(d);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(n: u64) -> String {
        cardinal(n).join(" ")
    }

    #[test]
    fn cardinals() {
        let small = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ];
        for (n, w) in small.iter().enumerate() {
            assert_eq!(c(n as u64), *w);
        }
        assert_eq!(c(21), "twenty one");
        assert_eq!(c(30), "thirty");
        assert_eq!(c(99), "ninety nine");
        assert_eq!(c(100), "one hundred");
        assert_eq!(c(101), "one hundred one");
        assert_eq!(c(999), "nine hundred ninety nine");
        assert_eq!(c(1000), "one thousand");
        assert_eq!(c(1234), "one thousand two hundred thirty four");
        assert_eq!(c(10_000), "ten thousand");
        assert_eq!(c(100_000), "one hundred thousand");
        assert_eq!(c(1_000_000), "one million");
        assert_eq!(c(1_000_001), "one million one");
        assert_eq!(c(1_000_000_000), "one billion");
        assert_eq!(c(1_000_000_000_000), "one trillion");
        assert_eq!(c(2_500_000_000_000), "two trillion five hundred billion");
        assert_eq!(c(1_000_000_000_000_000), "one quadrillion");
    }

    #[test]
    fn ordinals() {
        let o = |n| ordinal(n).join(" ");
        assert_eq!(o(1), "first");
        assert_eq!(o(2), "second");
        assert_eq!(o(3), "third");
        assert_eq!(o(5), "fifth");
        assert_eq!(o(8), "eighth");
        assert_eq!(o(9), "ninth");
        assert_eq!(o(12), "twelfth");
        assert_eq!(o(20), "twentieth");
        assert_eq!(o(21), "twenty first");
        assert_eq!(o(100), "one hundredth");
        assert_eq!(o(1000), "one thousandth");
    }

    #[test]
    fn years_decimals_digits() {
        let y = |n| year(n).join(" ");
        assert_eq!(y(1984), "nineteen eighty four");
        assert_eq!(y(2005), "two thousand five");
        assert_eq!(y(2026), "twenty twenty six");
        assert_eq!(y(1900), "nineteen hundred");
        assert_eq!(y(1905), "nineteen oh five");
        assert_eq!(y(2000), "two thousand");
        assert_eq!(decimal("3", "14").join(" "), "three point one four");
        assert_eq!(decimal("", "5").join(" "), "zero point five");
        assert_eq!(digits("007").join(" "), "zero zero seven");
        assert_eq!(cardinal_str("007").join(" "), "zero zero seven");
        assert_eq!(cardinal_str("12345678901234567890").len(), 20);
        assert_eq!(fraction(1, 2).join(" "), "one half");
        assert_eq!(fraction(3, 4).join(" "), "three quarters");
        assert_eq!(fraction(2, 3).join(" "), "two thirds");
        assert_eq!(fraction(1, 8).join(" "), "one eighth");
    }
}

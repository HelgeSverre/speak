//! Units of measure (spec §7.1). `(symbol, singular, plural)`; symbols are case-sensitive.

pub const UNITS: &[(&str, &str, &str)] = &[
    ("km", "kilometer", "kilometers"),
    ("m", "meter", "meters"),
    ("cm", "centimeter", "centimeters"),
    ("mm", "millimeter", "millimeters"),
    ("mi", "mile", "miles"),
    ("ft", "foot", "feet"),
    ("in", "inch", "inches"),
    ("kg", "kilogram", "kilograms"),
    ("g", "gram", "grams"),
    ("mg", "milligram", "milligrams"),
    ("lb", "pound", "pounds"),
    ("lbs", "pound", "pounds"),
    ("oz", "ounce", "ounces"),
    ("l", "liter", "liters"),
    ("L", "liter", "liters"),
    ("ml", "milliliter", "milliliters"),
    ("mL", "milliliter", "milliliters"),
    ("s", "second", "seconds"),
    ("sec", "second", "seconds"),
    ("ms", "millisecond", "milliseconds"),
    ("ns", "nanosecond", "nanoseconds"),
    ("min", "minute", "minutes"),
    ("h", "hour", "hours"),
    ("hr", "hour", "hours"),
    ("hrs", "hour", "hours"),
    ("Hz", "hertz", "hertz"),
    ("kHz", "kilohertz", "kilohertz"),
    ("MHz", "megahertz", "megahertz"),
    ("GHz", "gigahertz", "gigahertz"),
    ("B", "byte", "bytes"),
    ("KB", "kilobyte", "kilobytes"),
    ("kB", "kilobyte", "kilobytes"),
    ("MB", "megabyte", "megabytes"),
    ("GB", "gigabyte", "gigabytes"),
    ("TB", "terabyte", "terabytes"),
    ("W", "watt", "watts"),
    ("kW", "kilowatt", "kilowatts"),
    ("V", "volt", "volts"),
    ("dB", "decibel", "decibels"),
    ("mph", "mile per hour", "miles per hour"),
    ("kph", "kilometer per hour", "kilometers per hour"),
    ("fps", "frame per second", "frames per second"),
    ("wpm", "word per minute", "words per minute"),
    ("°C", "degree celsius", "degrees celsius"),
    ("°F", "degree fahrenheit", "degrees fahrenheit"),
    ("st", "semitone", "semitones"),
];

pub fn unit(symbol: &str) -> Option<(&'static str, &'static str)> {
    UNITS.iter().find(|u| u.0 == symbol).map(|u| (u.1, u.2))
}

/// Symbols that are also common words: only a unit when glued to the number (`5m`), not as a
/// separate token (`5 in`, `10 s`).
pub fn needs_glue(symbol: &str) -> bool {
    matches!(
        symbol,
        "m" | "in" | "s" | "h" | "l" | "L" | "g" | "B" | "W" | "V" | "st" | "mi" | "min"
    )
}

/// Words that mark the preceding number as a quantity, not a year.
pub fn is_unit_word(word: &str) -> bool {
    let w = word.to_lowercase();
    UNITS.iter().any(|u| u.1 == w || u.2 == w)
        || matches!(
            w.as_str(),
            "dollars"
                | "dollar"
                | "euros"
                | "euro"
                | "pounds"
                | "cents"
                | "people"
                | "times"
                | "items"
                | "percent"
                | "years"
                | "days"
                | "hours"
                | "minutes"
                | "seconds"
                | "words"
                | "lines"
                | "files"
                | "points"
        )
}

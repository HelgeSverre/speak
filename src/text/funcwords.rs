//! Function words: no pitch accent, no primary stress (spec §7.6, §7.7).

pub const FUNCTION_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "nor", "so", "if", "than", "as", "of", "to", "in", "on",
    "at", "by", "for", "from", "with", "into", "onto", "upon", "about", "is", "am", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "can", "could", "may", "might", "must", "i", "me", "my", "we", "us", "our",
    "you", "your", "he", "him", "his", "she", "her", "it", "its", "they", "them", "their", "that",
    "this", "there", "which", "who", "some", "i'm", "i'll", "i've", "i'd", "you're", "you'll",
    "he's", "she's", "it's", "we're", "they're", "that's", "there's",
];

pub fn is_function_word(word: &str) -> bool {
    let w = word.to_lowercase();
    FUNCTION_WORDS.contains(&w.as_str())
}

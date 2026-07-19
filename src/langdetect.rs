//! Text-based language detection for the auto-mode language probe.
//!
//! Wraps [`whatlang`] and restricts its output to the ISO 639-1 codes the
//! Qwen3-ASR server accepts (mirrors [`media::lang_from_name`]). Used to pin
//! one language for an entire file before transcribing all of its segments,
//! instead of letting the server auto-detect language independently for each
//! segment (which drifts within a single video).

use std::collections::HashMap;

use whatlang::{detect, Lang};

/// Map whatlang's ISO 639-3 variants to the ISO 639-1 codes the ASR server
/// accepts. Returns `None` for unsupported langs so the caller skips the vote
/// — forcing an unsupported language (Vietnamese, Thai, …) onto a supported
/// one would be worse than returning `None` and falling back to auto-detect,
/// which is why we use plain `detect()` rather than `Detector::with_allowlist`.
fn supported_code(lang: Lang) -> Option<&'static str> {
    match lang {
        // whatlang has no Cantonese (Yue) variant; Cmn is the only Chinese arm.
        // Qwen3-ASR treats `zh` as Chinese broadly, so this is acceptable.
        Lang::Cmn => Some("zh"),
        Lang::Jpn => Some("ja"),
        Lang::Eng => Some("en"),
        Lang::Kor => Some("ko"),
        Lang::Fra => Some("fr"),
        Lang::Deu => Some("de"),
        Lang::Spa => Some("es"),
        Lang::Rus => Some("ru"),
        _ => None,
    }
}

/// Minimum non-whitespace chars a probe transcription must have before we run
/// whatlang on it — guards against short/noisy ASR-on-music output.
const MIN_TEXT_CHARS: usize = 10;

/// Drop whatlang guesses below this confidence. whatlang's own `is_reliable()`
/// is far too strict for short ASR sentences (it returns false even at ~0.55
/// confidence), so we use a low floor to reject only pure-noise guesses and let
/// the supported-code allowlist + plurality vote absorb the remaining
/// short-text misclassifications (which tend to land on unsupported langs).
const MIN_CONFIDENCE: f64 = 0.05;

/// Plurality vote over probe transcriptions. Returns the winning ISO 639-1
/// code, or `None` if no text yielded a reliable, supported-language detection.
pub fn detect_language_code(texts: &[&str]) -> Option<String> {
    let mut votes: HashMap<&'static str, u32> = HashMap::new();
    for text in texts {
        let trimmed = text.trim();
        let non_ws = trimmed.chars().filter(|c| !c.is_whitespace()).count();
        if non_ws < MIN_TEXT_CHARS {
            continue;
        }
        let info = match detect(trimmed) {
            Some(i) => i,
            None => continue,
        };
        if info.confidence() < MIN_CONFIDENCE {
            continue;
        }
        if let Some(code) = supported_code(info.lang()) {
            *votes.entry(code).or_insert(0) += 1;
        }
    }
    votes
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(code, _)| code.to_string())
}

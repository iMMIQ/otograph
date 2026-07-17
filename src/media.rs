//! Media-file discovery + path helpers.

use anyhow::Result;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub fn default_extensions() -> Vec<String> {
    [
        "mp4", "mkv", "mov", "avi", "webm", "m4v", "mpg", "mpeg", "ts", "m2ts", "flv",
        "mp3", "wav", "flac", "m4a", "aac", "opus", "ogg", "oga", "wma", "amr",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

pub fn discover(root: &Path, exts: &[String]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if root.is_file() {
        out.push(root.to_path_buf());
    } else if root.is_dir() {
        for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }
            if let Some(ext) = entry.path().extension().and_then(|x| x.to_str()) {
                if exts.iter().any(|x| x.eq_ignore_ascii_case(ext)) {
                    out.push(entry.path().to_path_buf());
                }
            }
        }
    } else {
        anyhow::bail!("path does not exist: {}", root.display());
    }
    out.sort();
    Ok(out)
}

/// `foo/bar/video.mp4` -> `foo/bar/video.srt`
pub fn srt_path_for(media: &Path) -> PathBuf {
    media.with_extension("srt")
}

/// Parse a language code from the filename (e.g. `talk.zh.mp4` -> `zh`,
/// `movie_[ja].mkv` -> `ja`). Returns None if no token matches.
pub fn lang_from_name(media: &Path) -> Option<String> {
    let stem = media.file_stem()?.to_str()?.to_lowercase();
    let table: &[(&str, &str)] = &[
        ("zh", "zh"), ("chinese", "zh"), ("cn", "zh"), ("mandarin", "zh"), ("cantonese", "zh"),
        ("ja", "ja"), ("jp", "ja"), ("japanese", "ja"),
        ("en", "en"), ("english", "en"),
        ("ko", "ko"), ("korean", "ko"),
        ("fr", "fr"), ("french", "fr"),
        ("de", "de"), ("german", "de"),
        ("es", "es"), ("spanish", "es"),
        ("ru", "ru"), ("russian", "ru"),
    ];
    for tok in stem.split(|c: char| !c.is_alphanumeric()) {
        for (k, v) in table {
            if tok == *k {
                return Some(v.to_string());
            }
        }
    }
    None
}

//! SRT formatting + writing.

use anyhow::Result;
use std::path::Path;

pub struct SrtEntry {
    pub start: f64, // seconds
    pub end: f64,
    pub text: String,
}

pub fn format_timestamp(seconds: f64) -> String {
    // round to milliseconds, then split
    let total_ms = (seconds * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let m = (total_s / 60) % 60;
    let h = total_s / 3600;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

pub fn write_srt(path: &Path, entries: &[SrtEntry]) -> Result<()> {
    let mut out = String::new();
    for (i, e) in entries.iter().enumerate() {
        let text = e.text.trim();
        if text.is_empty() {
            continue;
        }
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!("{} --> {}\n", format_timestamp(e.start), format_timestamp(e.end)));
        out.push_str(text);
        out.push_str("\n\n");
    }
    std::fs::write(path, out)?;
    Ok(())
}

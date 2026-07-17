//! Extract 16 kHz mono raw PCM (signed 16-bit little-endian) from any
//! ffmpeg-readable media file by piping ffmpeg's stdout into memory.
//! Raw s16le (no WAV container) sidesteps streaming-WAV header issues and
//! avoids parsing overhead.

use anyhow::{Context, Result};
use std::path::Path;
use std::process::{Command, Stdio};

pub fn extract_pcm16_16k_mono(input: &Path) -> Result<Vec<u8>> {
    let out = Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(input)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "s16le", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("failed to spawn ffmpeg")?;

    if !out.status.success() {
        let e = String::from_utf8_lossy(&out.stderr);
        let e = e.trim();
        anyhow::bail!("ffmpeg failed (exit {:?}): {}", out.status.code(), &e[..e.len().min(500)]);
    }
    Ok(out.stdout)
}

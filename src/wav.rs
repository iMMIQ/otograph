//! PCM helpers: decode raw s16le bytes into f32, and re-encode a slice back to a
//! 16k mono PCM16 WAV (for the ASR multipart upload).

/// Decode raw little-endian signed 16-bit PCM bytes into mono f32 samples in [-1, 1].
pub fn samples_from_s16le(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 2; // drop a trailing odd byte if present
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lo = bytes[i * 2] as i16;
        let hi = bytes[i * 2 + 1] as i16;
        let v = (hi << 8) | (lo & 0xff);
        out.push(v as f32 / 32768.0);
    }
    out
}

/// Encode f32 samples in [-1, 1] as a 16k mono PCM16 WAV byte buffer.
pub fn encode_pcm16_mono(samples: &[f32], sr: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut buf = Vec::with_capacity(44 + samples.len() * 2);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sr.to_le_bytes());
    buf.extend_from_slice(&(sr * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

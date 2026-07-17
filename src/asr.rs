//! Async client for the Qwen3-ASR vLLM transcription server.
//!
//! Mirrors the verified request shape (see README / project memory):
//!   POST {server}/v1/audio/transcriptions   (multipart)
//!     model=/models/Qwen3-ASR-1.7B  language=<lang>  max_completion_tokens=200  file=@seg.wav
//! response: {"text": "..."}

use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration;

#[derive(Deserialize)]
struct TranscriptionResp {
    text: Option<String>,
}

#[derive(Clone)]
pub struct AsrClient {
    http: Client,
    url: String,
    model: Arc<str>,
    language: Option<Arc<str>>,
    max_tokens: u32,
}

impl AsrClient {
    pub fn new(server: String, model: String, language: Option<String>, max_tokens: u32) -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(600))
            .pool_idle_timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(64)
            .build()?;
        let url = format!("{}/v1/audio/transcriptions", server.trim_end_matches('/'));
        Ok(Self {
            http,
            url,
            model: model.into(),
            language: language.map(|s| Arc::<str>::from(s)),
            max_tokens,
        })
    }

    pub async fn transcribe(&self, wav_bytes: Vec<u8>) -> Result<String> {
        let mut form = reqwest::multipart::Form::new()
            .text("model", self.model.to_string())
            .text("max_completion_tokens", self.max_tokens.to_string());
        if let Some(lang) = &self.language {
            form = form.text("language", lang.to_string());
        }
        let part = reqwest::multipart::Part::bytes(wav_bytes)
            .file_name("seg.wav")
            .mime_str("audio/wav")?;
        form = form.part("file", part);

        let resp = self.http.post(&self.url).multipart(form).send().await?;
        let status = resp.status();
        let body = resp.bytes().await?;
        if !status.is_success() {
            let txt = String::from_utf8_lossy(&body);
            return Err(anyhow!("ASR HTTP {status}: {}", &txt[..txt.len().min(400)]));
        }
        let parsed: TranscriptionResp =
            serde_json::from_slice(&body).map_err(|e| anyhow!("decode ASR json: {e} | body: {}", {
                let t = String::from_utf8_lossy(&body);
                t[..t.len().min(200)].to_string()
            }))?;
        Ok(parsed.text.unwrap_or_default())
    }
}

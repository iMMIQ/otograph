mod asr;
mod ffmpeg;
mod media;
mod srt;
mod vad;
mod wav;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use tokio::sync::Semaphore;

use asr::AsrClient;
use vad::{Speech, VadModel, VadParams};

/// Generate SRT subtitles from video/audio via Silero VAD + Qwen3-ASR.
///
/// Discovers media files (recursively under a directory, or a single file),
/// splits each into speech segments with Silero VAD, transcribes each segment
/// with the Qwen3-ASR server, and writes `<media>.srt` next to the source.
#[derive(Parser, Debug)]
#[command(name = "otograph", version, about)]
struct Cli {
    /// File or directory to process (directories are scanned recursively).
    path: PathBuf,

    /// Qwen3-ASR server base URL.
    #[arg(long, default_value = "http://localhost:8002", env = "OTOGRAPH_SERVER")]
    server: String,

    /// Served model name (must match the server).
    #[arg(long, default_value = "/models/Qwen3-ASR-1.7B", env = "OTOGRAPH_MODEL")]
    model: String,

    /// Language for ALL files (e.g. zh / ja / en). Omit for server auto-detect
    /// (less reliable — pass an explicit language for best results).
    #[arg(long, env = "OTOGRAPH_LANGUAGE")]
    language: Option<String>,

    /// Parse language from each filename (e.g. `talk.zh.mp4` -> zh). Overrides --language.
    #[arg(long)]
    lang_from_name: bool,

    /// Path to silero_vad.onnx.
    #[arg(long, default_value = "model/silero_vad.onnx", env = "OTOGRAPH_VAD_MODEL")]
    vad_model: PathBuf,

    /// Concurrency: max simultaneous ASR requests per file.
    #[arg(long, default_value_t = 8)]
    concurrency: usize,

    /// VAD speech probability threshold.
    #[arg(long, default_value_t = 0.5)]
    vad_threshold: f32,

    /// Drop speech shorter than this (ms).
    #[arg(long, default_value_t = 250)]
    vad_min_speech_ms: u32,

    /// Silence of at least this length splits two segments (ms).
    #[arg(long, default_value_t = 300)]
    vad_min_silence_ms: u32,

    /// Pad each segment by this much on each side (ms).
    #[arg(long, default_value_t = 30)]
    vad_speech_pad_ms: u32,

    /// Hard cap on a single segment's length (s); longer ones are split at the
    /// best internal silence. Keep < 30 (the ASR native chunk length).
    #[arg(long, default_value_t = 28.0)]
    vad_max_segment_s: f32,

    /// max_completion_tokens sent to the ASR server (bounds decode length).
    #[arg(long, default_value_t = 200)]
    max_completion_tokens: u32,

    /// Comma-separated list of media extensions to consider.
    #[arg(long, default_value = "")]
    extensions: String,

    /// Overwrite existing .srt files.
    #[arg(long)]
    force: bool,

    /// Discover + VAD-segment only; do not call the ASR server (prints a plan).
    #[arg(long)]
    dry_run: bool,
}

struct Prep {
    samples: Vec<f32>,
    segs: Vec<Speech>,
}

fn resolve_language(cli: &Cli, media: &Path) -> Option<String> {
    if cli.lang_from_name {
        media::lang_from_name(media).or_else(|| cli.language.clone())
    } else {
        cli.language.clone()
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let exts = if cli.extensions.is_empty() {
        media::default_extensions()
    } else {
        cli.extensions.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
    };

    let files = media::discover(&cli.path, &exts)?;
    if files.is_empty() {
        bail!("no media files found under {}", cli.path.display());
    }
    eprintln!("found {} media file(s)", files.len());

    // Load the VAD model once; ping it across files via spawn_blocking.
    let mut model = VadModel::load(&cli.vad_model)
        .with_context(|| format!("could not load VAD model at {}", cli.vad_model.display()))?;

    if !cli.dry_run {
        // sanity: server reachable?
        match reqwest::Client::builder().timeout(std::time::Duration::from_secs(5)).build() {
            Ok(c) => {
                if let Err(e) = c.get(format!("{}/v1/models", cli.server.trim_end_matches('/'))).send().await {
                    eprintln!("⚠️  ASR server {} not reachable ({}); transcription will fail unless it comes up.", cli.server, e);
                }
            }
            Err(_) => {}
        }
    }

    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {bar:40} {pos}/{len} {wide_msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );

    let mut ok = 0usize;
    let mut failed: Vec<PathBuf> = Vec::new();

    for media_path in &files {
        pb.set_message(media_path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default());

        let srt_path = media::srt_path_for(media_path);
        if srt_path.exists() && !cli.force {
            pb.inc(1);
            eprintln!("skip (srt exists): {}", media_path.display());
            continue;
        }

        // ---- blocking prep: ffmpeg decode + VAD (run off the async runtime) ----
        let params = VadParams {
            threshold: cli.vad_threshold,
            min_speech_duration_ms: cli.vad_min_speech_ms,
            max_speech_duration_s: cli.vad_max_segment_s,
            min_silence_duration_ms: cli.vad_min_silence_ms,
            speech_pad_ms: cli.vad_speech_pad_ms,
            ..Default::default()
        };
        let path_clone = media_path.clone();
        let (returned_model, prep_res) = tokio::task::spawn_blocking(move || -> (VadModel, Result<Prep>) {
            let r = (|| {
                let bytes = ffmpeg::extract_pcm16_16k_mono(&path_clone)?;
                let samples = wav::samples_from_s16le(&bytes);
                if samples.is_empty() {
                    bail!("ffmpeg produced 0 samples — input may be corrupt/unreadable or empty");
                }
                let segs = model.speech_timestamps(&samples, &params)?;
                Ok(Prep { samples, segs })
            })();
            (model, r)
        })
        .await?;
        model = returned_model;

        let prep = match prep_res {
            Ok(p) => p,
            Err(e) => {
                eprintln!("❌ {}: prep failed: {e}", media_path.display());
                failed.push(media_path.clone());
                pb.inc(1);
                continue;
            }
        };

        let total_speech: f64 = prep.segs.iter().map(|s| (s.end - s.start) as f64).sum::<f64>() / 16000.0;
        eprintln!(
            " segmented {}: {} segments, {:.1}s of speech",
            media_path.display(),
            prep.segs.len(),
            total_speech
        );

        if cli.dry_run {
            for (i, s) in prep.segs.iter().enumerate() {
                eprintln!(
                    "   #{:<3} {:7.2} -> {:7.2} ({:5.1}s)",
                    i + 1,
                    s.start as f64 / 16000.0,
                    s.end as f64 / 16000.0,
                    (s.end - s.start) as f64 / 16000.0
                );
            }
            pb.inc(1);
            ok += 1;
            continue;
        }

        if prep.segs.is_empty() {
            eprintln!("⚠️  {}: no speech detected; skipping", media_path.display());
            pb.inc(1);
            continue;
        }

        // ---- transcribe segments concurrently ----
        let lang = resolve_language(&cli, media_path);
        let asr = AsrClient::new(cli.server.clone(), cli.model.clone(), lang.clone(), cli.max_completion_tokens)?;
        let texts = match transcribe_segments(&prep, asr, cli.concurrency).await {
            Ok(t) => t,
            Err(e) => {
                eprintln!("❌ {}: transcription failed: {e}", media_path.display());
                failed.push(media_path.clone());
                pb.inc(1);
                continue;
            }
        };

        let entries: Vec<srt::SrtEntry> = texts
            .iter()
            .map(|(idx, text)| srt::SrtEntry {
                start: prep.segs[*idx].start as f64 / 16000.0,
                end: prep.segs[*idx].end as f64 / 16000.0,
                text: text.clone(),
            })
            .collect();

        srt::write_srt(&srt_path, &entries)?;
        let nonempty = entries.iter().filter(|e| !e.text.trim().is_empty()).count();
        eprintln!("✅ {}: {} entries -> {}", media_path.display(), nonempty, srt_path.display());
        pb.inc(1);
        ok += 1;
    }

    pb.finish_with_message(format!("{}/{} done", ok, files.len()));
    if !failed.is_empty() {
        eprintln!("\n{} file(s) failed:", failed.len());
        for f in &failed {
            eprintln!("  {}", f.display());
        }
        std::process::exit(1);
    }
    Ok(())
}

async fn transcribe_segments(
    prep: &Prep,
    asr: AsrClient,
    concurrency: usize,
) -> Result<Vec<(usize, String)>> {
    let conc = concurrency.max(1);
    let sem = Arc::new(Semaphore::new(conc));
    let mut futs = Vec::new();

    for (idx, seg) in prep.segs.iter().enumerate() {
        let start = seg.start;
        let end = seg.end.min(prep.samples.len());
        if end <= start {
            continue;
        }
        let bytes = wav::encode_pcm16_mono(&prep.samples[start..end], 16000);
        let asr = asr.clone();
        let sem = sem.clone();
        futs.push(tokio::spawn(async move {
            let _permit = sem.acquire_owned().await?;
            let txt = asr.transcribe(bytes).await?;
            Ok::<(usize, String), anyhow::Error>((idx, txt))
        }));
    }

    let mut out = Vec::with_capacity(futs.len());
    for f in futs {
        match f.await {
            Ok(Ok(v)) => out.push(v),
            Ok(Err(e)) => return Err(e),
            Err(e) => bail!("task panicked: {e}"),
        }
    }
    out.sort_by_key(|(i, _)| *i);
    Ok(out)
}

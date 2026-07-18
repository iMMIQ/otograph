//! Launch and tear down the Qwen3-ASR vLLM Docker container.
//!
//! [`ContainerGuard::launch`] starts the container and blocks until its
//! `/health` endpoint is ready. When the guard is dropped (at the end of
//! `main`, including on error via `?`) the container is removed, so the ASR
//! deployment is always shut down after transcription.
//!
//! Mirrors epub-translator's `container.rs`, adapted to the Qwen3-ASR
//! container's real config (from `docker inspect` of the working deployment):
//! four per-engine cache mounts (triton/flashinfer/torchinductor/vllm),
//! `--shm-size 1g`, native bf16 KV cache (no `--kv-cache-dtype`), and
//! `--enforce-eager` (CUDA-graph capture is pathological ~121s on this Jetson).
//!
//! Cold start is retried (`launch_retries`): on this Jetson's shared unified
//! memory, vLLM v1's GPU-memory profiling occasionally trips a strict assertion
//! ("Initial free memory X, current free memory Y") when another container's
//! CUDA memory shifts during the profiling window. The race is transient, so a
//! fresh launch attempt succeeds within a few tries.

use anyhow::{bail, Context, Result};
use std::process::{Command, Stdio};
use std::time::Duration;

pub struct ServeSpec {
    pub image: String,
    /// Host directory containing the model files (mounted read-only at
    /// `container_model`).
    pub host_model_dir: String,
    /// Host base dir whose `{triton,flashinfer,torchinductor,vllm}` subdirs
    /// are persisted as the container's JIT/compile caches. The FlashInfer/
    /// Triton kernels JIT-compiled on first launch are reused on subsequent
    /// launches (avoids minutes of recompilation each cold start).
    pub host_cache_dir: String,
    /// Path inside the container (also the served model id).
    pub container_model: String,
    pub host_port: u16,
    pub container_name: String,
    pub dtype: String,
    /// `None` = omit `--kv-cache-dtype` (use the model's native dtype, bf16).
    /// ASR uses native; the LLM (epub) presets use fp8.
    pub kv_cache_dtype: Option<String>,
    pub gpu_memory_utilization: f32,
    pub max_model_len: u32,
    /// vLLM admission cap (`--max-num-seqs`); `None` = vLLM default (256).
    pub max_num_seqs: Option<u32>,
    /// Host-side multimodal preprocessing cache size in GiB. Transcription
    /// segments are unique, so the default disables this 4 GiB vLLM cache.
    pub mm_processor_cache_gb: f32,
    /// Pass `--enforce-eager` (no torch.compile/CUDA graphs). ASR wants eager:
    /// graph capture is pathological (~121s) on this Jetson.
    pub enforce_eager: bool,
    /// Seconds to wait for the server to become healthy (per attempt).
    pub health_timeout: u64,
    /// Cold-start attempts; each failure (profiling race / container exit) is
    /// retried after tearing down the dead container. 1 = no retry.
    pub launch_retries: u32,
}

pub struct ContainerGuard {
    name: String,
    endpoint: String,
    active: bool,
}

impl ContainerGuard {
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Start the container and wait until it is healthy, retrying transient
    /// cold-start failures. Returns a guard whose `Drop` removes the container.
    pub async fn launch(spec: &ServeSpec) -> Result<Self> {
        let retries = spec.launch_retries.max(1);
        let mut last: Option<anyhow::Error> = None;
        for attempt in 1..=retries {
            match launch_once(spec).await {
                Ok(g) => return Ok(g),
                Err(e) => {
                    // Tear down the dead container so the next attempt gets a
                    // clean window (the profiling race is triggered by GPU-mem
                    // movement during init; a fresh start clears it).
                    let _ = Command::new("docker")
                        .args(["rm", "-f", &spec.container_name])
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .status();
                    let is_last = attempt == retries;
                    if is_last {
                        // Surface the logs of the final (failed) attempt so the
                        // failure is diagnosable.
                        let _ = Command::new("docker")
                            .args(["logs", &spec.container_name])
                            .status();
                        last = Some(e);
                    } else {
                        eprintln!(
                            "launch attempt {}/{} failed: {}; retrying in 8s…",
                            attempt, retries, e
                        );
                        tokio::time::sleep(Duration::from_secs(8)).await;
                    }
                }
            }
        }
        Err(last.unwrap_or_else(|| anyhow::anyhow!("launch failed with no attempts")))
    }
}

/// One cold-start attempt: remove any stale container, run, wait for health.
async fn launch_once(spec: &ServeSpec) -> Result<ContainerGuard> {
    // Remove any stale container of the same name.
    let _ = Command::new("docker")
        .args(["rm", "-f", &spec.container_name])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    let model_mount = format!("{}:{}:ro", spec.host_model_dir, spec.container_model);
    // Per-engine JIT/compile caches, each in its own subdir so the ASR and
    // any co-deployed LLM containers don't collide.
    let cb = spec.host_cache_dir.trim_end_matches('/');
    let cache_mounts: Vec<String> = vec![
        format!("{cb}/triton:/root/.triton"),
        format!("{cb}/flashinfer:/root/.cache/flashinfer"),
        format!("{cb}/torchinductor:/root/.cache/torchinductor"),
        format!("{cb}/vllm:/root/.cache/vllm"),
    ];

    // All `docker run` flags come before the image; all `vllm serve` flags
    // after it. Mixing the two would feed vllm args to docker (or vice versa).
    let mut args: Vec<String> = vec![
        "run".into(),
        "-d".into(),
        "--runtime=nvidia".into(),
        "--name".into(),
        spec.container_name.clone(),
        "-p".into(),
        format!("{}:8000", spec.host_port),
        "--shm-size".into(),
        "1g".into(),
        "-v".into(),
        model_mount,
    ];
    for m in &cache_mounts {
        args.push("-v".into());
        args.push(m.clone());
    }
    args.extend([
        "-e".into(),
        "VLLM_CONFIGURE_LOGGING=0".into(),
        "-e".into(),
        "VLLM_DO_NOT_TRACK=1".into(),
        "-e".into(),
        "VLLM_NO_USAGE_STATS=1".into(),
        "-e".into(),
        "VLLM_CACHE_ROOT=/root/.cache/vllm".into(),
        "-e".into(),
        "TRITON_CACHE_DIR=/root/.triton".into(),
        "-e".into(),
        "TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor".into(),
        "--entrypoint".into(),
        "vllm".into(),
        spec.image.clone(),
        "serve".into(),
        spec.container_model.clone(),
        "--host".into(),
        "0.0.0.0".into(),
        "--port".into(),
        "8000".into(),
        "--dtype".into(),
        spec.dtype.clone(),
        "--max-model-len".into(),
        spec.max_model_len.to_string(),
        "--gpu-memory-utilization".into(),
        spec.gpu_memory_utilization.to_string(),
        "--mm-processor-cache-gb".into(),
        spec.mm_processor_cache_gb.to_string(),
        "--enable-prefix-caching".into(),
    ]);
    if spec.enforce_eager {
        args.push("--enforce-eager".into());
    }
    if let Some(mns) = spec.max_num_seqs {
        args.push("--max-num-seqs".into());
        args.push(mns.to_string());
    }
    if let Some(kv) = &spec.kv_cache_dtype {
        args.push("--kv-cache-dtype".into());
        args.push(kv.clone());
    }

    eprintln!(
        "starting Qwen3-ASR container '{}' (image {}, model {})…",
        spec.container_name, spec.image, spec.container_model
    );
    let out = Command::new("docker")
        .args(&args)
        .output()
        .context("failed to invoke docker")?;
    if !out.status.success() {
        bail!(
            "docker run failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }

    let endpoint = format!("http://127.0.0.1:{}", spec.host_port);
    if !wait_health(&endpoint, &spec.container_name, spec.health_timeout).await {
        bail!(
            "ASR container '{}' did not become healthy within {}s",
            spec.container_name, spec.health_timeout
        );
    }
    eprintln!("Qwen3-ASR healthy at {}", endpoint);

    Ok(ContainerGuard {
        name: spec.container_name.clone(),
        endpoint,
        active: true,
    })
}

impl Drop for ContainerGuard {
    fn drop(&mut self) {
        if self.active {
            eprintln!("stopping container '{}'…", self.name);
            let _ = Command::new("docker")
                .args(["rm", "-f", &self.name])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }
}

async fn wait_health(endpoint: &str, name: &str, timeout: u64) -> bool {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout);
    loop {
        if !is_running(name) {
            eprintln!("container '{}' exited before becoming healthy", name);
            return false;
        }
        if let Ok(resp) = client.get(format!("{}/health", endpoint)).send().await {
            if resp.status().is_success() {
                return true;
            }
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_secs(4)).await;
    }
}

fn is_running(name: &str) -> bool {
    Command::new("docker")
        .args(["inspect", "-f", "{{.State.Running}}", name])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "true")
        .unwrap_or(false)
}

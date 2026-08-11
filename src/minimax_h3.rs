//! Experimental MiniMax-H3 video lane.
//!
//! H3 is not an autoregressive GGUF model. Its smallest useful local bundle is
//! a diffusion transformer, a truncated Qwen3-VL text encoder, and video/audio
//! VAEs. Camelid therefore delegates this lane to a capability-checked
//! `stable-diffusion.cpp` `sd-cli` process instead of admitting the files to the
//! text-generation engine.

use std::{
    ffi::{OsStr, OsString},
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const COMMUNITY_LICENSE_URL: &str =
    "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE";
pub const BACKEND_SOURCE_URL: &str = "https://github.com/leejet/stable-diffusion.cpp";
/// First stable-diffusion.cpp revision checked while adding the Camelid bridge.
/// Runtime preflight remains capability-based because released `sd-cli` builds
/// do not expose a machine-readable source revision.
pub const CHECKED_BACKEND_REVISION: &str = "c6beeef35526c6dc94b74a7fb69f9d2e6a2a7a12";
const BACKEND_PREFLIGHT_TIMEOUT: Duration = Duration::from_secs(5);

const MODEL_REPO: &str = "unsloth/MiniMax-H3-GGUF";
const VAE_REPO: &str = "Comfy-Org/MiniMax-H3";

const FL2VA_MODEL: Artifact = Artifact {
    role: "diffusion_model",
    repo_id: MODEL_REPO,
    remote_path: "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf",
    local_filename: "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf",
    size_bytes: 8_063_029_344,
    sha256: "cfe0795c00ab6e6ebf8c64fe4574f45a828e8a93e0876bca704e055662a9d7b8",
};

const REF2VA_MODEL: Artifact = Artifact {
    role: "diffusion_model",
    repo_id: MODEL_REPO,
    remote_path: "minimax_h3_ref2va_pruned-Q2_K.gguf",
    local_filename: "minimax_h3_ref2va_pruned-Q2_K.gguf",
    size_bytes: 6_678_171_744,
    sha256: "12089d0a9935b3616c19e430a2e9e0e14e4b391f773363a167619ad245c1ab6f",
};

const TEXT_ENCODER: Artifact = Artifact {
    role: "text_encoder",
    repo_id: MODEL_REPO,
    remote_path: "qwen3vl_32b_minimax_h3-Q2_K_M.gguf",
    local_filename: "qwen3vl_32b_minimax_h3-Q2_K_M.gguf",
    size_bytes: 13_102_161_024,
    sha256: "a8ccadccd57ef34c838ffb8a7da8368bb554721b2760274a1d3b0df63960b997",
};

const VIDEO_VAE: Artifact = Artifact {
    role: "video_vae",
    repo_id: VAE_REPO,
    remote_path: "vae/minimax_h3_video_vae_fp16.safetensors",
    local_filename: "minimax_h3_video_vae_fp16.safetensors",
    size_bytes: 5_207_808_496,
    sha256: "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
};

const AUDIO_VAE: Artifact = Artifact {
    role: "audio_vae",
    repo_id: VAE_REPO,
    remote_path: "vae/minimax_h3_audio_vae_fp32.safetensors",
    local_filename: "minimax_h3_audio_vae_fp32.safetensors",
    size_bytes: 605_254_808,
    sha256: "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum H3Variant {
    /// Text-to-video, first-frame, and first/last-frame generation.
    Fl2va,
    /// Image/video/audio reference-conditioned generation.
    Ref2va,
}

impl H3Variant {
    fn diffusion_artifact(self) -> Artifact {
        match self {
            Self::Fl2va => FL2VA_MODEL,
            Self::Ref2va => REF2VA_MODEL,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Artifact {
    pub role: &'static str,
    pub repo_id: &'static str,
    pub remote_path: &'static str,
    pub local_filename: &'static str,
    pub size_bytes: u64,
    pub sha256: &'static str,
}

pub fn bundle_artifacts(variant: H3Variant, include_audio: bool) -> Vec<Artifact> {
    let mut artifacts = vec![variant.diffusion_artifact(), TEXT_ENCODER, VIDEO_VAE];
    if include_audio {
        artifacts.push(AUDIO_VAE);
    }
    artifacts
}

#[derive(Clone, Debug)]
pub struct H3Bundle {
    pub variant: H3Variant,
    pub diffusion_model: PathBuf,
    pub text_encoder: PathBuf,
    pub video_vae: PathBuf,
    pub audio_vae: Option<PathBuf>,
}

impl H3Bundle {
    pub fn from_dir(dir: &Path, variant: H3Variant, include_audio: bool) -> Self {
        Self {
            variant,
            diffusion_model: dir.join(variant.diffusion_artifact().local_filename),
            text_encoder: dir.join(TEXT_ENCODER.local_filename),
            video_vae: dir.join(VIDEO_VAE.local_filename),
            audio_vae: include_audio.then(|| dir.join(AUDIO_VAE.local_filename)),
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        validate_file(&self.diffusion_model, "MiniMax-H3 diffusion model")?;
        validate_file(&self.text_encoder, "MiniMax-H3 Qwen3-VL text encoder")?;
        validate_file(&self.video_vae, "MiniMax-H3 video VAE")?;
        if let Some(audio_vae) = &self.audio_vae {
            validate_file(audio_vae, "MiniMax-H3 audio VAE")?;
        }

        let diffusion_name = lowercase_filename(&self.diffusion_model)?;
        let expected_marker = match self.variant {
            H3Variant::Fl2va => "minimax_h3_fl2va",
            H3Variant::Ref2va => "minimax_h3_ref2va",
        };
        anyhow::ensure!(
            diffusion_name.contains(expected_marker) && diffusion_name.ends_with(".gguf"),
            "the {:?} lane requires a {expected_marker} GGUF, got {}",
            self.variant,
            self.diffusion_model.display()
        );

        let encoder_name = lowercase_filename(&self.text_encoder)?;
        anyhow::ensure!(
            encoder_name.contains("qwen3vl_32b_minimax_h3") && encoder_name.ends_with(".gguf"),
            "the text encoder must be the MiniMax-H3 Qwen3-VL-32B GGUF, got {}",
            self.text_encoder.display()
        );
        anyhow::ensure!(
            lowercase_filename(&self.video_vae)? == "minimax_h3_video_vae_fp16.safetensors",
            "the video VAE must be minimax_h3_video_vae_fp16.safetensors"
        );
        if let Some(audio_vae) = &self.audio_vae {
            anyhow::ensure!(
                lowercase_filename(audio_vae)? == "minimax_h3_audio_vae_fp32.safetensors",
                "the audio VAE must be minimax_h3_audio_vae_fp32.safetensors"
            );
        }
        Ok(())
    }
}

fn validate_file(path: &Path, label: &str) -> anyhow::Result<()> {
    let metadata = std::fs::metadata(path)
        .map_err(|err| anyhow::anyhow!("{label} is unavailable at {}: {err}", path.display()))?;
    anyhow::ensure!(
        metadata.is_file(),
        "{label} is not a file: {}",
        path.display()
    );
    anyhow::ensure!(metadata.len() > 0, "{label} is empty: {}", path.display());
    Ok(())
}

fn lowercase_filename(path: &Path) -> anyhow::Result<String> {
    path.file_name()
        .and_then(OsStr::to_str)
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| anyhow::anyhow!("path has no UTF-8 filename: {}", path.display()))
}

#[derive(Clone, Debug)]
pub struct GenerateRequest {
    pub prompt: String,
    pub output: PathBuf,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub steps: u32,
    pub seed: i64,
    pub init_image: Option<PathBuf>,
    pub end_image: Option<PathBuf>,
    pub reference_images: Vec<PathBuf>,
    pub reference_videos: Vec<PathBuf>,
    pub reference_audios: Vec<PathBuf>,
    pub offload_to_cpu: bool,
}

impl GenerateRequest {
    pub fn validate(&self, variant: H3Variant) -> anyhow::Result<()> {
        anyhow::ensure!(!self.prompt.trim().is_empty(), "prompt must not be empty");
        anyhow::ensure!(self.prompt.len() <= 32 * 1024, "prompt exceeds 32 KiB");
        anyhow::ensure!(
            (64..=4096).contains(&self.width) && self.width.is_multiple_of(32),
            "width must be a multiple of 32 between 64 and 4096"
        );
        anyhow::ensure!(
            (64..=4096).contains(&self.height) && self.height.is_multiple_of(32),
            "height must be a multiple of 32 between 64 and 4096"
        );
        anyhow::ensure!(
            (5..=360).contains(&self.frames),
            "frames must be between 5 and 360 (15 seconds at 24 fps)"
        );
        anyhow::ensure!(
            (1..=100).contains(&self.steps),
            "steps must be between 1 and 100"
        );
        anyhow::ensure!(
            matches!(
                self.output
                    .extension()
                    .and_then(OsStr::to_str)
                    .map(str::to_ascii_lowercase)
                    .as_deref(),
                Some("webm" | "avi")
            ),
            "output must end in .webm or .avi; WebM is recommended for native stereo audio"
        );

        if let Some(init) = &self.init_image {
            validate_file(init, "initial frame")?;
        }
        if let Some(end) = &self.end_image {
            anyhow::ensure!(
                self.init_image.is_some(),
                "--end-image requires --init-image for first/last-frame generation"
            );
            validate_file(end, "ending frame")?;
        }
        for path in &self.reference_images {
            validate_file(path, "reference image")?;
        }
        for path in &self.reference_videos {
            anyhow::ensure!(
                path.is_dir(),
                "stable-diffusion.cpp reference videos must be frame directories: {}",
                path.display()
            );
        }
        for path in &self.reference_audios {
            validate_file(path, "reference audio")?;
        }

        let has_references = !self.reference_images.is_empty()
            || !self.reference_videos.is_empty()
            || !self.reference_audios.is_empty();
        match variant {
            H3Variant::Fl2va => anyhow::ensure!(
                !has_references,
                "reference media requires --variant ref2va; fl2va accepts text and optional first/last frames"
            ),
            H3Variant::Ref2va => {
                anyhow::ensure!(
                    self.init_image.is_none() && self.end_image.is_none(),
                    "ref2va reference conditioning cannot be combined with first/last frames"
                );
                anyhow::ensure!(has_references, "ref2va requires at least one reference input");
            }
        }
        Ok(())
    }

    pub fn effective_frames(&self) -> u32 {
        align_h3_frames(self.frames)
    }
}

/// MiniMax-H3 uses the temporal grid `17k + 5`; stable-diffusion.cpp rounds up.
pub fn align_h3_frames(requested: u32) -> u32 {
    if requested <= 5 {
        5
    } else {
        ((requested - 5).div_ceil(17) * 17) + 5
    }
}

#[derive(Debug, Serialize)]
pub struct GenerationPlan {
    pub lane: &'static str,
    pub support_status: &'static str,
    pub variant: H3Variant,
    pub backend: String,
    pub checked_backend_revision: &'static str,
    pub requested_frames: u32,
    pub effective_frames: u32,
    pub fps: u32,
    pub audio_enabled: bool,
    pub output: String,
    pub argv: Vec<String>,
}

pub fn build_generation_plan(
    sd_cli: &Path,
    bundle: &H3Bundle,
    request: &GenerateRequest,
) -> anyhow::Result<(GenerationPlan, Vec<OsString>)> {
    bundle.validate()?;
    request.validate(bundle.variant)?;
    anyhow::ensure!(
        request.reference_audios.is_empty() || bundle.audio_vae.is_some(),
        "MiniMax-H3 reference audio requires the audio VAE; remove --no-audio or pass --audio-vae"
    );

    let mut args: Vec<OsString> = vec![
        "--mode".into(),
        "vid_gen".into(),
        "--diffusion-model".into(),
        bundle.diffusion_model.as_os_str().to_owned(),
        "--llm".into(),
        bundle.text_encoder.as_os_str().to_owned(),
        "--vae".into(),
        bundle.video_vae.as_os_str().to_owned(),
    ];
    if let Some(audio_vae) = &bundle.audio_vae {
        args.push("--audio-vae".into());
        args.push(audio_vae.as_os_str().to_owned());
    }
    args.extend([
        "--prompt".into(),
        request.prompt.as_str().into(),
        "--width".into(),
        request.width.to_string().into(),
        "--height".into(),
        request.height.to_string().into(),
        "--video-frames".into(),
        request.frames.to_string().into(),
        "--fps".into(),
        "24".into(),
        "--steps".into(),
        request.steps.to_string().into(),
        "--cfg-scale".into(),
        "1.0".into(),
        "--seed".into(),
        request.seed.to_string().into(),
        "--backend".into(),
        "te=cpu".into(),
        "--diffusion-fa".into(),
        "--rng".into(),
        "cpu".into(),
    ]);
    if request.offload_to_cpu {
        args.push("--offload-to-cpu".into());
    }
    if let Some(path) = &request.init_image {
        args.push("--init-img".into());
        args.push(path.as_os_str().to_owned());
    }
    if let Some(path) = &request.end_image {
        args.push("--end-img".into());
        args.push(path.as_os_str().to_owned());
    }
    for path in &request.reference_images {
        args.push("--ref-image".into());
        args.push(path.as_os_str().to_owned());
    }
    for path in &request.reference_videos {
        args.push("--ref-video".into());
        args.push(path.as_os_str().to_owned());
    }
    for path in &request.reference_audios {
        args.push("--ref-audio".into());
        args.push(path.as_os_str().to_owned());
    }
    args.push("--output".into());
    args.push(request.output.as_os_str().to_owned());

    let plan = GenerationPlan {
        lane: "minimax_h3_video_via_sd_cli",
        support_status: "experimental_backend_bridge",
        variant: bundle.variant,
        backend: sd_cli.display().to_string(),
        checked_backend_revision: CHECKED_BACKEND_REVISION,
        requested_frames: request.frames,
        effective_frames: request.effective_frames(),
        fps: 24,
        audio_enabled: bundle.audio_vae.is_some(),
        output: request.output.display().to_string(),
        argv: args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect(),
    };
    Ok((plan, args))
}

pub fn resolve_sd_cli(explicit: Option<&Path>) -> PathBuf {
    if let Some(path) = explicit {
        return path.to_path_buf();
    }
    if let Some(path) = std::env::var_os("CAMELID_SD_CLI") {
        return PathBuf::from(path);
    }
    if let Ok(executable) = std::env::current_exe() {
        let name = if cfg!(windows) {
            "sd-cli.exe"
        } else {
            "sd-cli"
        };
        if let Some(sibling) = executable.parent().map(|dir| dir.join(name)) {
            if sibling.is_file() {
                return sibling;
            }
        }
    }
    #[cfg(target_os = "macos")]
    if let Ok(volumes) = std::fs::read_dir("/Volumes") {
        let name = "sd-cli";
        for volume in volumes.flatten() {
            let candidate = volume.path().join("Camelid").join("bin").join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    PathBuf::from(if cfg!(windows) {
        "sd-cli.exe"
    } else {
        "sd-cli"
    })
}

pub fn preflight_sd_cli(sd_cli: &Path) -> anyhow::Result<()> {
    let mut child = Command::new(sd_cli)
        .arg("--help")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| {
            anyhow::anyhow!(
                "could not run MiniMax-H3 backend {}: {err}; install a current stable-diffusion.cpp sd-cli or pass --sd-cli",
                sd_cli.display()
            )
        })?;
    let deadline = Instant::now() + BACKEND_PREFLIGHT_TIMEOUT;
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|err| {
            anyhow::anyhow!(
                "could not poll MiniMax-H3 backend {}: {err}",
                sd_cli.display()
            )
        })? {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!(
                "{} --help did not finish within {} seconds; on macOS, approve Camelid's removable-volume access or bundle sd-cli with the desktop app",
                sd_cli.display(),
                BACKEND_PREFLIGHT_TIMEOUT.as_secs()
            );
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    if let Some(mut pipe) = child.stdout.take() {
        pipe.read_to_end(&mut stdout)?;
    }
    if let Some(mut pipe) = child.stderr.take() {
        pipe.read_to_end(&mut stderr)?;
    }
    anyhow::ensure!(
        status.success(),
        "{} --help exited {}; use a current stable-diffusion.cpp build from {BACKEND_SOURCE_URL}",
        sd_cli.display(),
        status
    );
    let help = format!(
        "{}\n{}",
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr)
    );
    for required in [
        "--audio-vae",
        "--diffusion-model",
        "--video-frames",
        "--llm",
        "--diffusion-fa",
    ] {
        anyhow::ensure!(
            help.contains(required),
            "{} lacks required H3 capability flag {required}; build stable-diffusion.cpp at or after checked revision {CHECKED_BACKEND_REVISION}",
            sd_cli.display()
        );
    }
    let binary = locate_executable(sd_cli).ok_or_else(|| {
        anyhow::anyhow!(
            "{} ran but its executable file could not be resolved for the H3 capability check",
            sd_cli.display()
        )
    })?;
    anyhow::ensure!(
        file_contains_any(&binary, &[b"MiniMax-H3", b"minimax_h3"])
            .map_err(|err| anyhow::anyhow!("could not inspect {}: {err}", binary.display()))?,
        "{} does not contain the MiniMax-H3 runtime marker; build stable-diffusion.cpp at or after checked revision {CHECKED_BACKEND_REVISION}",
        binary.display()
    );
    Ok(())
}

fn locate_executable(command: &Path) -> Option<PathBuf> {
    if command.is_file() {
        return Some(command.to_path_buf());
    }
    if command.components().count() > 1 {
        return None;
    }
    let path = std::env::var_os("PATH")?;
    for directory in std::env::split_paths(&path) {
        let candidate = directory.join(command);
        if candidate.is_file() {
            return Some(candidate);
        }
        #[cfg(windows)]
        if candidate.extension().is_none() {
            let executable = candidate.with_extension("exe");
            if executable.is_file() {
                return Some(executable);
            }
        }
    }
    None
}

fn file_contains_any(path: &Path, needles: &[&[u8]]) -> std::io::Result<bool> {
    let mut file = File::open(path)?;
    let longest = needles.iter().map(|needle| needle.len()).max().unwrap_or(0);
    let mut carry = Vec::with_capacity(longest.saturating_sub(1));
    let mut chunk = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut chunk)?;
        if read == 0 {
            return Ok(false);
        }
        carry.extend_from_slice(&chunk[..read]);
        if needles
            .iter()
            .any(|needle| carry.windows(needle.len()).any(|window| window == *needle))
        {
            return Ok(true);
        }
        let retain = longest.saturating_sub(1).min(carry.len());
        carry.drain(..carry.len() - retain);
    }
}

pub fn generate(
    sd_cli: &Path,
    bundle: &H3Bundle,
    request: &GenerateRequest,
    dry_run: bool,
) -> anyhow::Result<GenerationPlan> {
    let (plan, args) = build_generation_plan(sd_cli, bundle, request)?;
    if dry_run {
        return Ok(plan);
    }
    preflight_sd_cli(sd_cli)?;
    if let Some(parent) = request
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let status = Command::new(sd_cli)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|err| anyhow::anyhow!("failed to start {}: {err}", sd_cli.display()))?;
    anyhow::ensure!(status.success(), "MiniMax-H3 backend exited {status}");
    let output = std::fs::metadata(&request.output).map_err(|err| {
        anyhow::anyhow!(
            "backend succeeded but output {} is unavailable: {err}",
            request.output.display()
        )
    })?;
    anyhow::ensure!(output.len() > 0, "backend created an empty output file");
    Ok(plan)
}

#[derive(Debug, Serialize)]
pub struct DoctorArtifact {
    pub artifact: Artifact,
    pub path: String,
    pub present: bool,
    pub size_matches: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256_matches: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct DoctorReport {
    pub lane: &'static str,
    pub support_status: &'static str,
    pub variant: H3Variant,
    pub backend: String,
    pub backend_ready: bool,
    pub artifacts_ready: bool,
    pub artifacts: Vec<DoctorArtifact>,
    pub license: &'static str,
}

pub fn doctor(
    dir: &Path,
    variant: H3Variant,
    include_audio: bool,
    sd_cli: &Path,
    verify_sha256: bool,
) -> DoctorReport {
    let artifacts: Vec<DoctorArtifact> = bundle_artifacts(variant, include_audio)
        .into_iter()
        .map(|artifact| {
            let path = dir.join(artifact.local_filename);
            let size_matches = std::fs::metadata(&path)
                .map(|metadata| metadata.is_file() && metadata.len() == artifact.size_bytes)
                .unwrap_or(false);
            let sha256_matches = if verify_sha256 && size_matches {
                Some(
                    sha256_file(&path)
                        .map(|actual| actual == artifact.sha256)
                        .unwrap_or(false),
                )
            } else {
                None
            };
            DoctorArtifact {
                artifact,
                path: path.display().to_string(),
                present: path.is_file(),
                size_matches,
                sha256_matches,
            }
        })
        .collect();
    let artifacts_ready = artifacts
        .iter()
        .all(|item| item.present && item.size_matches && item.sha256_matches.unwrap_or(true));
    DoctorReport {
        lane: "minimax_h3_video_via_sd_cli",
        support_status: "experimental_backend_bridge",
        variant,
        backend: sd_cli.display().to_string(),
        backend_ready: preflight_sd_cli(sd_cli).is_ok(),
        artifacts_ready,
        artifacts,
        license: COMMUNITY_LICENSE_URL,
    }
}

pub fn pull_bundle(
    dir: &Path,
    variant: H3Variant,
    include_audio: bool,
    accepted_license: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        accepted_license,
        "MiniMax H3 uses the MiniMax-H3 Community License Agreement, including territory restrictions. Read {COMMUNITY_LICENSE_URL}, then re-run with --accept-license if it applies to you"
    );
    std::fs::create_dir_all(dir)?;
    for artifact in bundle_artifacts(variant, include_audio) {
        download_artifact(dir, artifact)?;
    }
    Ok(())
}

fn download_artifact(dir: &Path, artifact: Artifact) -> anyhow::Result<()> {
    let destination = dir.join(artifact.local_filename);
    if destination.is_file() {
        let size = std::fs::metadata(&destination)?.len();
        anyhow::ensure!(
            size == artifact.size_bytes,
            "existing {} has {size} bytes, expected {}; move it aside before retrying",
            destination.display(),
            artifact.size_bytes
        );
        anyhow::ensure!(
            sha256_file(&destination)? == artifact.sha256,
            "existing {} failed the pinned SHA-256 check; move it aside before retrying",
            destination.display()
        );
        eprintln!(
            "{} already present and SHA-256 verified",
            artifact.local_filename
        );
        return Ok(());
    }

    let partial = dir.join(format!("{}.part", artifact.local_filename));
    if partial.is_file() && std::fs::metadata(&partial)?.len() > artifact.size_bytes {
        File::create(&partial)?;
    }
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        artifact.repo_id, artifact.remote_path
    );
    eprintln!(
        "Downloading {} ({:.2} GiB) from {}",
        artifact.local_filename,
        artifact.size_bytes as f64 / 1024_f64.powi(3),
        artifact.repo_id
    );
    let status = Command::new("curl")
        .args(["-L", "-C", "-", "--fail", "--output"])
        .arg(&partial)
        .arg(&url)
        .status()
        .map_err(|err| anyhow::anyhow!("could not run curl: {err}"))?;
    let downloaded = std::fs::metadata(&partial)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    anyhow::ensure!(
        status.success() || downloaded == artifact.size_bytes,
        "download failed (curl exited {status}); re-run to resume"
    );
    anyhow::ensure!(
        downloaded == artifact.size_bytes,
        "download incomplete: {} has {downloaded} bytes, expected {}; re-run to resume",
        partial.display(),
        artifact.size_bytes
    );
    eprintln!("Verifying SHA-256 for {}", artifact.local_filename);
    let actual_sha256 = sha256_file(&partial)?;
    if actual_sha256 != artifact.sha256 {
        // The `.part` path is owned by this downloader. Truncate a corrupt full
        // body so the next invocation starts at byte zero instead of repeatedly
        // asking the Hub to resume an invalid complete-length file.
        File::create(&partial)?;
        anyhow::bail!(
            "downloaded {} failed its pinned SHA-256 check and its partial file was discarded; re-run to download it again",
            artifact.local_filename
        );
    }
    std::fs::rename(&partial, &destination)?;
    Ok(())
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(path: &Path) {
        std::fs::write(path, b"test").unwrap();
    }

    fn fixture() -> (tempfile::TempDir, H3Bundle, GenerateRequest) {
        let dir = tempfile::tempdir().unwrap();
        let bundle = H3Bundle::from_dir(dir.path(), H3Variant::Fl2va, true);
        touch(&bundle.diffusion_model);
        touch(&bundle.text_encoder);
        touch(&bundle.video_vae);
        touch(bundle.audio_vae.as_ref().unwrap());
        let request = GenerateRequest {
            prompt: "a red panda walking through mist".into(),
            output: dir.path().join("clip.webm"),
            width: 640,
            height: 384,
            frames: 25,
            steps: 4,
            seed: 11,
            init_image: None,
            end_image: None,
            reference_images: Vec::new(),
            reference_videos: Vec::new(),
            reference_audios: Vec::new(),
            offload_to_cpu: true,
        };
        (dir, bundle, request)
    }

    #[test]
    fn h3_frame_grid_rounds_up_like_the_backend() {
        assert_eq!(align_h3_frames(5), 5);
        assert_eq!(align_h3_frames(6), 22);
        assert_eq!(align_h3_frames(22), 22);
        assert_eq!(align_h3_frames(25), 39);
        assert_eq!(align_h3_frames(360), 362);
    }

    #[test]
    fn fl2va_plan_pins_h3_safe_runtime_flags() {
        let (_dir, bundle, request) = fixture();
        let (plan, args) = build_generation_plan(Path::new("sd-cli"), &bundle, &request).unwrap();
        let args: Vec<String> = args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(plan.support_status, "experimental_backend_bridge");
        assert_eq!(plan.effective_frames, 39);
        assert!(args.windows(2).any(|pair| pair == ["--cfg-scale", "1.0"]));
        assert!(args.windows(2).any(|pair| pair == ["--backend", "te=cpu"]));
        assert!(args.contains(&"--diffusion-fa".to_string()));
        assert!(args.contains(&"--offload-to-cpu".to_string()));
    }

    #[test]
    fn reference_media_fails_closed_on_fl2va() {
        let (dir, bundle, mut request) = fixture();
        let reference = dir.path().join("reference.png");
        touch(&reference);
        request.reference_images.push(reference);
        let error = build_generation_plan(Path::new("sd-cli"), &bundle, &request)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires --variant ref2va"), "{error}");
    }

    #[test]
    fn end_frame_requires_initial_frame() {
        let (dir, bundle, mut request) = fixture();
        let end = dir.path().join("end.png");
        touch(&end);
        request.end_image = Some(end);
        let error = build_generation_plan(Path::new("sd-cli"), &bundle, &request)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires --init-image"), "{error}");
    }

    #[test]
    fn doctor_reports_missing_bundle_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let report = doctor(
            dir.path(),
            H3Variant::Fl2va,
            true,
            Path::new("definitely-not-an-sd-cli"),
            false,
        );
        assert!(!report.backend_ready);
        assert!(!report.artifacts_ready);
        assert_eq!(report.artifacts.len(), 4);
    }

    #[test]
    fn reference_audio_requires_the_audio_vae() {
        let (dir, mut bundle, mut request) = fixture();
        bundle.variant = H3Variant::Ref2va;
        bundle.diffusion_model = dir.path().join(REF2VA_MODEL.local_filename);
        touch(&bundle.diffusion_model);
        bundle.audio_vae = None;
        let reference = dir.path().join("reference.wav");
        touch(&reference);
        request.reference_audios.push(reference);
        let error = build_generation_plan(Path::new("sd-cli"), &bundle, &request)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires the audio VAE"), "{error}");
    }

    #[test]
    fn pull_refuses_to_download_before_license_acknowledgement() {
        let dir = tempfile::tempdir().unwrap();
        let error = pull_bundle(dir.path(), H3Variant::Fl2va, true, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("--accept-license"), "{error}");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn backend_marker_scan_crosses_read_chunk_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sd-cli-fixture");
        let mut bytes = vec![b'x'; 1024 * 1024 - 4];
        bytes.extend_from_slice(b"MiniMax-H3");
        std::fs::write(&path, bytes).unwrap();
        assert!(file_contains_any(&path, &[b"MiniMax-H3", b"minimax_h3"]).unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn backend_preflight_kills_a_stalled_help_process() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sd-cli-stalled");
        std::fs::write(&path, "#!/bin/sh\nsleep 30\n").unwrap();
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&path, permissions).unwrap();

        let started = Instant::now();
        let error = preflight_sd_cli(&path).unwrap_err().to_string();
        assert!(error.contains("did not finish"), "{error}");
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "stalled backend was not bounded"
        );
    }
}

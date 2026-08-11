//! Local asynchronous MiniMax-H3 video jobs for the dashboard.
//!
//! This module deliberately owns only orchestration and media delivery. The
//! model contract, artifact identities, argument construction, and backend
//! capability checks remain in `minimax_h3` so the CLI and UI cannot drift.

use std::{
    collections::HashMap,
    ffi::OsString,
    fs::OpenOptions,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    extract::{Path as AxumPath, Query, State},
    http::{
        header::{
            ACCEPT_RANGES, CACHE_CONTROL, CONTENT_LENGTH, CONTENT_RANGE, CONTENT_TYPE, RANGE,
        },
        HeaderMap, HeaderValue, StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::{
    io::{AsyncReadExt, AsyncSeekExt},
    sync::{RwLock, Semaphore},
};
use tokio_util::io::ReaderStream;

use super::AppState;
use crate::minimax_h3::{
    self, build_generation_plan, bundle_artifacts, configure_backend_command, preflight_sd_cli,
    GenerateRequest, H3Bundle, H3Variant,
};

const MAX_RETAINED_JOBS: usize = 64;

#[derive(Clone)]
pub(super) struct VideoJobManager {
    jobs: Arc<RwLock<HashMap<String, VideoJobRecord>>>,
    generation_slot: Arc<Semaphore>,
}

impl Default for VideoJobManager {
    fn default() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            // H3's working set is large enough that concurrent local jobs are
            // unsafe on the machines this bridge targets.
            generation_slot: Arc::new(Semaphore::new(1)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum VideoJobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Canceled,
}

impl VideoJobStatus {
    fn terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Canceled)
    }
}

#[derive(Clone)]
struct VideoJobRecord {
    id: String,
    status: VideoJobStatus,
    prompt: String,
    variant: H3Variant,
    width: u32,
    height: u32,
    requested_frames: u32,
    effective_frames: u32,
    steps: u32,
    seed: i64,
    audio_enabled: bool,
    models_dir: String,
    created_at: u64,
    updated_at: u64,
    error: Option<String>,
    output: PathBuf,
    log: PathBuf,
    cancel: Arc<AtomicBool>,
}

#[derive(Serialize)]
struct VideoJobView {
    id: String,
    status: VideoJobStatus,
    prompt: String,
    variant: H3Variant,
    width: u32,
    height: u32,
    requested_frames: u32,
    effective_frames: u32,
    fps: u32,
    steps: u32,
    seed: i64,
    audio_enabled: bool,
    models_dir: String,
    created_at: u64,
    updated_at: u64,
    error: Option<String>,
    content_url: Option<String>,
    log_url: String,
}

impl From<&VideoJobRecord> for VideoJobView {
    fn from(job: &VideoJobRecord) -> Self {
        let content_url = (job.status == VideoJobStatus::Succeeded)
            .then(|| format!("/api/video/jobs/{}/content", job.id));
        Self {
            id: job.id.clone(),
            status: job.status,
            prompt: job.prompt.clone(),
            variant: job.variant,
            width: job.width,
            height: job.height,
            requested_frames: job.requested_frames,
            effective_frames: job.effective_frames,
            fps: 24,
            steps: job.steps,
            seed: job.seed,
            audio_enabled: job.audio_enabled,
            models_dir: job.models_dir.clone(),
            created_at: job.created_at,
            updated_at: job.updated_at,
            error: job.error.clone(),
            content_url,
            log_url: format!("/api/video/jobs/{}/log", job.id),
        }
    }
}

#[derive(Deserialize)]
pub(super) struct VideoCapabilityQuery {
    models_dir: Option<String>,
    sd_cli: Option<String>,
    variant: Option<H3Variant>,
    include_audio: Option<bool>,
}

#[derive(Deserialize)]
pub(super) struct CreateVideoJobRequest {
    prompt: String,
    variant: Option<H3Variant>,
    models_dir: Option<String>,
    sd_cli: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    frames: Option<u32>,
    steps: Option<u32>,
    seed: Option<i64>,
    include_audio: Option<bool>,
    init_image: Option<String>,
    end_image: Option<String>,
    reference_images: Option<Vec<String>>,
    reference_videos: Option<Vec<String>>,
    reference_audios: Option<Vec<String>>,
    offload_to_cpu: Option<bool>,
}

pub(super) async fn capabilities(
    State(state): State<AppState>,
    Query(query): Query<VideoCapabilityQuery>,
) -> Response {
    if let Some(response) = require_loopback(&state) {
        return response;
    }
    let models_dir = discover_models_dir(query.models_dir.as_deref(), &state.models_dir);
    let sd_cli = minimax_h3::resolve_sd_cli(query.sd_cli.as_deref().map(Path::new));
    let variant = query.variant.unwrap_or(H3Variant::Fl2va);
    let include_audio = query.include_audio.unwrap_or(true);
    let output_dir = discover_output_dir(&models_dir);

    let artifacts = bundle_artifacts(variant, include_audio)
        .into_iter()
        .map(|artifact| {
            let final_path = models_dir.join(artifact.local_filename);
            let partial_path = models_dir.join(format!("{}.part", artifact.local_filename));
            let final_bytes = file_len(&final_path);
            let partial_bytes = file_len(&partial_path);
            let downloaded_bytes = final_bytes.max(partial_bytes).min(artifact.size_bytes);
            let stage = if final_bytes == artifact.size_bytes {
                "ready"
            } else if partial_bytes == artifact.size_bytes {
                "verifying"
            } else if partial_bytes > 0 {
                "downloading"
            } else {
                "missing"
            };
            json!({
                "role": artifact.role,
                "filename": artifact.local_filename,
                "expected_bytes": artifact.size_bytes,
                "downloaded_bytes": downloaded_bytes,
                "size_matches": final_bytes == artifact.size_bytes,
                "downloading": partial_bytes > 0 && final_bytes != artifact.size_bytes,
                "stage": stage,
            })
        })
        .collect::<Vec<_>>();
    let expected_bytes = artifacts
        .iter()
        .filter_map(|artifact| artifact["expected_bytes"].as_u64())
        .sum::<u64>();
    let downloaded_bytes = artifacts
        .iter()
        .filter_map(|artifact| artifact["downloaded_bytes"].as_u64())
        .sum::<u64>();
    let artifacts_ready = artifacts
        .iter()
        .all(|artifact| artifact["size_matches"].as_bool() == Some(true));

    let backend = sd_cli.clone();
    let backend_check = tokio::task::spawn_blocking(move || preflight_sd_cli(&backend))
        .await
        .map_err(|error| format!("backend check could not run: {error}"))
        .and_then(|result| result.map_err(|error| error.to_string()));

    Json(json!({
        "lane": "minimax_h3_video_via_sd_cli",
        "support_status": "experimental_backend_bridge",
        "variant": variant,
        "models_dir": models_dir.display().to_string(),
        "output_dir": output_dir.display().to_string(),
        "backend": sd_cli.display().to_string(),
        "backend_ready": backend_check.is_ok(),
        "backend_error": backend_check.err(),
        "artifacts_ready": artifacts_ready,
        "expected_bytes": expected_bytes,
        "downloaded_bytes": downloaded_bytes,
        "artifacts": artifacts,
        "license": minimax_h3::COMMUNITY_LICENSE_URL,
        "checked_backend_revision": minimax_h3::CHECKED_BACKEND_REVISION,
        "variants": ["fl2va", "ref2va"],
    }))
    .into_response()
}

pub(super) async fn list_jobs(State(state): State<AppState>) -> Response {
    if let Some(response) = require_loopback(&state) {
        return response;
    }
    let jobs = state.video_jobs.jobs.read().await;
    let mut values = jobs.values().collect::<Vec<_>>();
    values.sort_by_key(|job| std::cmp::Reverse(job.created_at));
    Json(json!({
        "data": values.into_iter().map(VideoJobView::from).collect::<Vec<_>>()
    }))
    .into_response()
}

pub(super) async fn get_job(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Some(response) = require_loopback(&state) {
        return response;
    }
    let jobs = state.video_jobs.jobs.read().await;
    match jobs.get(&id) {
        Some(job) => Json(VideoJobView::from(job)).into_response(),
        None => api_error(StatusCode::NOT_FOUND, "video job was not found"),
    }
}

pub(super) async fn create_job(
    State(state): State<AppState>,
    Json(input): Json<CreateVideoJobRequest>,
) -> Response {
    if let Some(response) = require_loopback(&state) {
        return response;
    }
    let variant = input.variant.unwrap_or(H3Variant::Fl2va);
    let include_audio = input.include_audio.unwrap_or(true);
    let models_dir = discover_models_dir(input.models_dir.as_deref(), &state.models_dir);
    let sd_cli = minimax_h3::resolve_sd_cli(input.sd_cli.as_deref().map(Path::new));
    let output_dir = discover_output_dir(&models_dir);
    if let Err(error) = std::fs::create_dir_all(&output_dir) {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("could not create video output directory: {error}"),
        );
    }

    let id = uuid::Uuid::new_v4().to_string();
    let output = output_dir.join(format!("minimax-h3-{id}.webm"));
    let log = output_dir.join(format!("minimax-h3-{id}.log"));
    let request = GenerateRequest {
        prompt: input.prompt,
        output: output.clone(),
        width: input.width.unwrap_or(640),
        height: input.height.unwrap_or(384),
        frames: input.frames.unwrap_or(25),
        steps: input.steps.unwrap_or(4),
        seed: input.seed.unwrap_or(11),
        init_image: optional_path(input.init_image),
        end_image: optional_path(input.end_image),
        reference_images: path_list(input.reference_images),
        reference_videos: path_list(input.reference_videos),
        reference_audios: path_list(input.reference_audios),
        offload_to_cpu: input.offload_to_cpu.unwrap_or(true),
    };
    let bundle = H3Bundle::from_dir(&models_dir, variant, include_audio);
    let (plan, args) = match build_generation_plan(&sd_cli, &bundle, &request) {
        Ok(plan) => plan,
        Err(error) => return api_error(StatusCode::CONFLICT, error.to_string()),
    };

    let preflight_binary = sd_cli.clone();
    match tokio::task::spawn_blocking(move || preflight_sd_cli(&preflight_binary)).await {
        Ok(Ok(())) => {}
        Ok(Err(error)) => return api_error(StatusCode::CONFLICT, error.to_string()),
        Err(error) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("backend check could not run: {error}"),
            )
        }
    }

    let now = unix_timestamp();
    let record = VideoJobRecord {
        id: id.clone(),
        status: VideoJobStatus::Queued,
        prompt: request.prompt.clone(),
        variant,
        width: request.width,
        height: request.height,
        requested_frames: request.frames,
        effective_frames: plan.effective_frames,
        steps: request.steps,
        seed: request.seed,
        audio_enabled: include_audio,
        models_dir: models_dir.display().to_string(),
        created_at: now,
        updated_at: now,
        error: None,
        output,
        log,
        cancel: Arc::new(AtomicBool::new(false)),
    };
    let view = VideoJobView::from(&record);
    state.video_jobs.insert(record).await;
    state.video_jobs.spawn(id, sd_cli, args);

    (StatusCode::ACCEPTED, Json(view)).into_response()
}

pub(super) async fn cancel_job(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Some(response) = require_loopback(&state) {
        return response;
    }
    let mut jobs = state.video_jobs.jobs.write().await;
    let Some(job) = jobs.get_mut(&id) else {
        return api_error(StatusCode::NOT_FOUND, "video job was not found");
    };
    if !job.status.terminal() {
        job.cancel.store(true, Ordering::Release);
        if job.status == VideoJobStatus::Queued {
            job.status = VideoJobStatus::Canceled;
            job.updated_at = unix_timestamp();
        }
    }
    Json(VideoJobView::from(&*job)).into_response()
}

pub(super) async fn job_content(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    headers: HeaderMap,
) -> Response {
    if let Some(response) = require_loopback(&state) {
        return response;
    }
    let output = {
        let jobs = state.video_jobs.jobs.read().await;
        let Some(job) = jobs.get(&id) else {
            return api_error(StatusCode::NOT_FOUND, "video job was not found");
        };
        if job.status != VideoJobStatus::Succeeded {
            return api_error(StatusCode::CONFLICT, "video output is not ready");
        }
        job.output.clone()
    };
    let mut file = match tokio::fs::File::open(&output).await {
        Ok(file) => file,
        Err(error) => {
            return api_error(
                StatusCode::NOT_FOUND,
                format!("video output is unavailable: {error}"),
            )
        }
    };
    let size = match file.metadata().await {
        Ok(metadata) => metadata.len(),
        Err(error) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()),
    };

    let requested_range = headers
        .get(RANGE)
        .and_then(|value| value.to_str().ok())
        .map(|value| parse_byte_range(value, size));
    match requested_range {
        Some(Ok((start, end))) => {
            if let Err(error) = file.seek(std::io::SeekFrom::Start(start)).await {
                return api_error(StatusCode::INTERNAL_SERVER_ERROR, error.to_string());
            }
            let length = end - start + 1;
            let body = Body::from_stream(ReaderStream::new(file.take(length)));
            media_response(
                StatusCode::PARTIAL_CONTENT,
                body,
                length,
                Some(format!("bytes {start}-{end}/{size}")),
            )
        }
        Some(Err(message)) => {
            let mut response = api_error(StatusCode::RANGE_NOT_SATISFIABLE, message);
            if let Ok(value) = HeaderValue::from_str(&format!("bytes */{size}")) {
                response.headers_mut().insert(CONTENT_RANGE, value);
            }
            response
        }
        None => media_response(
            StatusCode::OK,
            Body::from_stream(ReaderStream::new(file)),
            size,
            None,
        ),
    }
}

pub(super) async fn job_log(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    if let Some(response) = require_loopback(&state) {
        return response;
    }
    let log = {
        let jobs = state.video_jobs.jobs.read().await;
        let Some(job) = jobs.get(&id) else {
            return api_error(StatusCode::NOT_FOUND, "video job was not found");
        };
        job.log.clone()
    };
    match tokio::fs::read(log).await {
        Ok(bytes) => (
            [
                (CONTENT_TYPE, "text/plain; charset=utf-8"),
                (CACHE_CONTROL, "no-store"),
            ],
            bytes,
        )
            .into_response(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            (StatusCode::OK, "Video job has not written a log yet.\n").into_response()
        }
        Err(error) => api_error(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()),
    }
}

impl VideoJobManager {
    async fn insert(&self, record: VideoJobRecord) {
        let mut jobs = self.jobs.write().await;
        if jobs.len() >= MAX_RETAINED_JOBS {
            let oldest_terminal = jobs
                .values()
                .filter(|job| job.status.terminal())
                .min_by_key(|job| job.created_at)
                .map(|job| job.id.clone());
            if let Some(id) = oldest_terminal {
                jobs.remove(&id);
            }
        }
        jobs.insert(record.id.clone(), record);
    }

    fn spawn(&self, id: String, sd_cli: PathBuf, args: Vec<OsString>) {
        let manager = self.clone();
        tokio::spawn(async move {
            let Ok(_permit) = manager.generation_slot.clone().acquire_owned().await else {
                manager
                    .finish(&id, Err("video generation queue closed".to_string()))
                    .await;
                return;
            };
            let Some((output, log, cancel)) = manager.mark_running(&id).await else {
                return;
            };
            let worker_cancel = cancel.clone();
            let result = tokio::task::spawn_blocking(move || {
                run_backend(&sd_cli, &args, &output, &log, &worker_cancel)
            })
            .await
            .unwrap_or_else(|error| Err(format!("video worker stopped unexpectedly: {error}")));
            manager.finish(&id, result).await;
        });
    }

    async fn mark_running(&self, id: &str) -> Option<(PathBuf, PathBuf, Arc<AtomicBool>)> {
        let mut jobs = self.jobs.write().await;
        let job = jobs.get_mut(id)?;
        if job.cancel.load(Ordering::Acquire) || job.status == VideoJobStatus::Canceled {
            job.status = VideoJobStatus::Canceled;
            job.updated_at = unix_timestamp();
            return None;
        }
        job.status = VideoJobStatus::Running;
        job.updated_at = unix_timestamp();
        Some((job.output.clone(), job.log.clone(), job.cancel.clone()))
    }

    async fn finish(&self, id: &str, result: Result<(), String>) {
        let mut jobs = self.jobs.write().await;
        let Some(job) = jobs.get_mut(id) else {
            return;
        };
        if job.cancel.load(Ordering::Acquire) {
            let _ = std::fs::remove_file(&job.output);
            job.status = VideoJobStatus::Canceled;
            job.error = None;
        } else if let Err(error) = result {
            job.status = VideoJobStatus::Failed;
            job.error = Some(error);
        } else {
            job.status = VideoJobStatus::Succeeded;
            job.error = None;
        }
        job.updated_at = unix_timestamp();
    }
}

fn run_backend(
    sd_cli: &Path,
    args: &[OsString],
    output: &Path,
    log: &Path,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let log_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(log)
        .map_err(|error| format!("could not create generation log: {error}"))?;
    let log_stderr = log_file
        .try_clone()
        .map_err(|error| format!("could not open generation log stream: {error}"))?;
    let mut command = Command::new(sd_cli);
    configure_backend_command(&mut command);
    let mut child = command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_stderr))
        .spawn()
        .map_err(|error| format!("failed to start {}: {error}", sd_cli.display()))?;

    loop {
        if cancel.load(Ordering::Acquire) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(output);
            return Ok(());
        }
        match child.try_wait() {
            Ok(Some(status)) if status.success() => break,
            Ok(Some(status)) => return Err(format!("MiniMax-H3 backend exited {status}")),
            Ok(None) => std::thread::sleep(Duration::from_millis(400)),
            Err(error) => return Err(format!("could not monitor MiniMax-H3 backend: {error}")),
        }
    }

    let metadata = std::fs::metadata(output)
        .map_err(|error| format!("backend succeeded but output is unavailable: {error}"))?;
    if metadata.len() == 0 {
        return Err("backend created an empty video output".to_string());
    }
    Ok(())
}

fn discover_models_dir(explicit: Option<&str>, server_models_dir: &Path) -> PathBuf {
    if let Some(path) = explicit.map(str::trim).filter(|value| !value.is_empty()) {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("CAMELID_H3_MODELS_DIR") {
        return PathBuf::from(path);
    }
    #[cfg(target_os = "macos")]
    if let Ok(volumes) = std::fs::read_dir("/Volumes") {
        for volume in volumes.flatten() {
            let candidate = volume
                .path()
                .join("Camelid")
                .join("models")
                .join("minimax-h3");
            if candidate.is_dir() {
                return candidate;
            }
        }
    }
    h3_models_dir(server_models_dir)
}

/// Keep Video Studio under the server's single resolved model-store authority.
/// This matters in particular for the Windows desktop app, whose Explorer
/// launch directory is not stable and may even be a protected system folder.
fn h3_models_dir(server_models_dir: &Path) -> PathBuf {
    if path_file_name_eq(server_models_dir, "minimax-h3") {
        server_models_dir.to_path_buf()
    } else {
        server_models_dir.join("minimax-h3")
    }
}

fn discover_output_dir(models_dir: &Path) -> PathBuf {
    if let Some(path) = std::env::var_os("CAMELID_H3_OUTPUT_DIR") {
        return PathBuf::from(path);
    }
    if path_file_name_eq(models_dir, "minimax-h3") {
        if let Some(models) = models_dir.parent() {
            if path_file_name_eq(models, "models") {
                if let Some(camelid) = models.parent() {
                    return camelid.join("outputs");
                }
            }
        }
    }
    models_dir.join("outputs")
}

fn path_file_name_eq(path: &Path, expected: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case(expected))
}

fn optional_path(value: Option<String>) -> Option<PathBuf> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn path_list(values: Option<Vec<String>>) -> Vec<PathBuf> {
    values
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| optional_path(Some(value)))
        .collect()
}

fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path)
        .map(|metadata| {
            if metadata.is_file() {
                metadata.len()
            } else {
                0
            }
        })
        .unwrap_or(0)
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn api_error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({ "error": { "message": message.into() } })),
    )
        .into_response()
}

fn require_loopback(state: &AppState) -> Option<Response> {
    (!state.serve_addr.ip().is_loopback()).then(|| {
        api_error(
            StatusCode::FORBIDDEN,
            "local video generation is available only on a loopback listener",
        )
    })
}

fn media_response(
    status: StatusCode,
    body: Body,
    content_length: u64,
    content_range: Option<String>,
) -> Response {
    let mut builder = Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "video/webm")
        .header(CONTENT_LENGTH, content_length)
        .header(ACCEPT_RANGES, "bytes")
        .header(CACHE_CONTROL, "no-store");
    if let Some(range) = content_range {
        builder = builder.header(CONTENT_RANGE, range);
    }
    builder.body(body).unwrap_or_else(|error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("could not serve video: {error}"),
        )
    })
}

fn parse_byte_range(value: &str, size: u64) -> Result<(u64, u64), &'static str> {
    if size == 0 || !value.starts_with("bytes=") || value.contains(',') {
        return Err("unsupported byte range");
    }
    let (start, end) = value[6..].split_once('-').ok_or("invalid byte range")?;
    if start.is_empty() {
        let suffix = end.parse::<u64>().map_err(|_| "invalid byte range")?;
        if suffix == 0 {
            return Err("invalid byte range");
        }
        return Ok((size.saturating_sub(suffix.min(size)), size - 1));
    }
    let start = start.parse::<u64>().map_err(|_| "invalid byte range")?;
    if start >= size {
        return Err("byte range starts beyond the video");
    }
    let end = if end.is_empty() {
        size - 1
    } else {
        end.parse::<u64>()
            .map_err(|_| "invalid byte range")?
            .min(size - 1)
    };
    if end < start {
        return Err("invalid byte range");
    }
    Ok((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use tower::ServiceExt;

    #[test]
    fn range_parser_supports_browser_range_shapes() {
        assert_eq!(parse_byte_range("bytes=0-99", 1_000), Ok((0, 99)));
        assert_eq!(parse_byte_range("bytes=500-", 1_000), Ok((500, 999)));
        assert_eq!(parse_byte_range("bytes=-100", 1_000), Ok((900, 999)));
        assert!(parse_byte_range("bytes=1000-", 1_000).is_err());
        assert!(parse_byte_range("bytes=0-1,4-5", 1_000).is_err());
    }

    #[test]
    fn external_layout_keeps_outputs_outside_the_model_directory() {
        let models = Path::new("/Volumes/External/Camelid/models/minimax-h3");
        assert_eq!(
            discover_output_dir(models),
            Path::new("/Volumes/External/Camelid/outputs")
        );
    }

    #[test]
    fn video_models_follow_the_resolved_server_model_store() {
        let server_models = Path::new("/app/sidecar/models");
        assert_eq!(
            h3_models_dir(server_models),
            server_models.join("minimax-h3")
        );

        let already_scoped = server_models.join("MiniMax-H3");
        assert_eq!(h3_models_dir(&already_scoped), already_scoped);
    }

    #[test]
    fn windows_style_case_does_not_break_the_external_output_layout() {
        let models = Path::new("/External/Camelid/MODELS/MiniMax-H3");
        assert_eq!(
            discover_output_dir(models),
            Path::new("/External/Camelid/outputs")
        );
    }

    #[tokio::test]
    async fn capabilities_reports_resumable_partial_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let partial = dir
            .path()
            .join("minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf.part");
        std::fs::write(partial, b"partial").unwrap();
        let uri = format!(
            "/api/video/capabilities?variant=fl2va&include_audio=false&models_dir={}&sd_cli=/missing/sd-cli",
            dir.path().display()
        );
        let app = super::super::router_with_state(AppState::default());
        let response = app
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload["downloaded_bytes"], 7);
        assert_eq!(payload["artifacts"][0]["downloading"], true);
        assert_eq!(payload["backend_ready"], false);
    }

    #[tokio::test]
    async fn create_job_fails_closed_before_queueing_incomplete_bundle() {
        let dir = tempfile::tempdir().unwrap();
        let app = super::super::router_with_state(AppState::default());
        let body = json!({
            "prompt": "a red panda in mist",
            "models_dir": dir.path().display().to_string(),
            "include_audio": false,
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/video/jobs")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn completed_job_serves_browser_byte_ranges() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("clip.webm");
        std::fs::write(&output, b"0123456789").unwrap();
        let state = AppState::default();
        let record = VideoJobRecord {
            id: "range-test".to_string(),
            status: VideoJobStatus::Succeeded,
            prompt: "test".to_string(),
            variant: H3Variant::Fl2va,
            width: 640,
            height: 384,
            requested_frames: 5,
            effective_frames: 5,
            steps: 4,
            seed: 11,
            audio_enabled: true,
            models_dir: dir.path().display().to_string(),
            created_at: 1,
            updated_at: 1,
            error: None,
            output,
            log: dir.path().join("clip.log"),
            cancel: Arc::new(AtomicBool::new(false)),
        };
        state.video_jobs.insert(record).await;
        let response = super::super::router_with_state(state)
            .oneshot(
                Request::builder()
                    .uri("/api/video/jobs/range-test/content")
                    .header(RANGE, "bytes=2-5")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
        assert_eq!(response.headers()[CONTENT_RANGE], "bytes 2-5/10");
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(&body[..], b"2345");
    }
}

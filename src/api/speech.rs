//! Authenticated speech routes use the same server policy as chat. One blocking
//! speech job per AppState; disconnecting a client cannot release its slot early.
use super::AppState;
use crate::speech;
use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

fn error(status: StatusCode, message: impl ToString) -> Response {
    (
        status,
        Json(json!({"error": {"message": message.to_string()}})),
    )
        .into_response()
}
pub(super) async fn status(State(state): State<AppState>) -> Response {
    Json(json!({
        "installed": speech::installed(&speech::model_dir(&state.models_dir)),
        "busy": state.speech_slot.available_permits() == 0,
        "model": "whisper-tiny.en", "language": "en", "device": "cpu",
        "download_bytes": speech::MODEL_BYTES, "max_seconds": speech::MAX_SECONDS,
    }))
    .into_response()
}
pub(super) async fn install(State(state): State<AppState>) -> Response {
    let Ok(slot) = state.speech_slot.clone().try_acquire_owned() else {
        return error(
            StatusCode::CONFLICT,
            "Speech is busy. Try again when the current operation finishes.",
        );
    };
    let dir = speech::model_dir(&state.models_dir);
    let limit = state.server_limits.max_download_bytes;
    match tokio::task::spawn_blocking(move || {
        let _slot = slot;
        speech::install(&dir, limit)
    })
    .await
    {
        Ok(Ok(())) => Json(json!({"installed": true})).into_response(),
        Ok(Err(e)) => error(StatusCode::BAD_GATEWAY, e),
        Err(e) => {
            tracing::error!(%e, "speech setup worker failed");
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Speech setup failed. Retry the download.",
            )
        }
    }
}
pub(super) async fn transcribe(State(state): State<AppState>, bytes: Bytes) -> Response {
    let Ok(slot) = state.speech_slot.clone().try_acquire_owned() else {
        return error(
            StatusCode::CONFLICT,
            "Speech is busy. Try again when the current operation finishes.",
        );
    };
    let dir = speech::model_dir(&state.models_dir);
    if !speech::installed(&dir) {
        return error(
            StatusCode::PRECONDITION_FAILED,
            "Download the speech model before recording.",
        );
    }
    match tokio::task::spawn_blocking(move || {
        let _slot = slot;
        let samples = speech::decode_audio(&bytes).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
        speech::transcribe(&dir, &samples).map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, e))
    })
    .await
    {
        Ok(Ok(text)) => Json(json!({"text": text})).into_response(),
        Ok(Err((code, e))) => error(code, e),
        Err(e) => {
            tracing::error!(%e, "speech transcription worker failed");
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Transcription failed. Try a shorter recording.",
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
        routing::{get, post},
        Router,
    };
    use tower::ServiceExt;
    #[tokio::test]
    async fn status_missing_model_and_busy_admission() {
        let temp = tempfile::tempdir().unwrap();
        let state = AppState::default().with_models_dir(Some(temp.path().to_path_buf()));
        let app = Router::new()
            .route("/status", get(status))
            .route("/transcribe", post(transcribe))
            .with_state(state.clone());
        let result = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let value: serde_json::Value =
            serde_json::from_slice(&to_bytes(result.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(value["installed"], false);
        let request = || {
            Request::builder()
                .method("POST")
                .uri("/transcribe")
                .body(Body::from("audio"))
                .unwrap()
        };
        assert_eq!(
            app.clone().oneshot(request()).await.unwrap().status(),
            StatusCode::PRECONDITION_FAILED
        );
        let _permit = state.speech_slot.clone().try_acquire_owned().unwrap();
        assert_eq!(
            app.oneshot(request()).await.unwrap().status(),
            StatusCode::CONFLICT
        );
    }
}

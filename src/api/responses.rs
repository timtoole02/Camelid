//! OpenAI Responses API adapter with opt-in durable local state.
//!
//! The generation implementation remains `/v1/chat/completions`; this module
//! owns only request normalization and response/event translation. That keeps
//! model routing, queueing, cancellation, token accounting, tool parsing, and
//! evidence gates on one execution path. Responses storage preserves canonical
//! JSON items in a separate SQLite store rather than flattening tool state into
//! the Workspace text-memory schema.

#![allow(clippy::result_large_err)]

use std::{collections::HashMap, convert::Infallible};

use axum::{
    body::{to_bytes, Body},
    extract::{rejection::JsonRejection, Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{sse::Event, IntoResponse, Response, Sse},
    Json,
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::{
    api_error, chat_completions, malformed_json_error,
    responses_store::{ResponseCommit, StoreError},
    ChatCompletionRequest, ChatMessage, StopSpec,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct ResponsesRequest {
    model: Option<String>,
    input: Option<Value>,
    instructions: Option<Value>,
    stream: Option<bool>,
    max_output_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
    parallel_tool_calls: Option<bool>,
    text: Option<Value>,
    store: Option<bool>,
    metadata: Option<Value>,
    previous_response_id: Option<String>,
    conversation: Option<Value>,
    background: Option<bool>,
    #[serde(flatten)]
    unsupported_fields: HashMap<String, Value>,
}

pub(super) async fn create(
    State(state): State<super::AppState>,
    headers: HeaderMap,
    payload: Result<Json<ResponsesRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(err) => return malformed_json_error(err),
    };
    if let Err(response) = request.validate_boundary_fields() {
        return response;
    }
    let stream = request.stream.unwrap_or(false);
    let store_response = request.store.unwrap_or(false);
    let conversation_id = match request.conversation_id() {
        Ok(id) => id,
        Err(response) => return response,
    };
    let idempotency_key = match idempotency_key(&headers) {
        Ok(key) => key,
        Err(response) => return response,
    };
    if idempotency_key.is_some() && !store_response {
        return api_error(
            StatusCode::BAD_REQUEST,
            "idempotency_requires_storage",
            "Idempotency-Key requires store:true so the completed response can be replayed"
                .to_string(),
            Some("store"),
        );
    }
    let request_hash = request_fingerprint(&request);
    let mut lock_keys = Vec::with_capacity(2);
    if let Some(id) = conversation_id.as_ref() {
        lock_keys.push(format!("conversation:{id}"));
    }
    if let Some(key) = idempotency_key.as_ref() {
        lock_keys.push(format!("idempotency:{key}"));
    }
    lock_keys.sort_unstable();
    let mut persistence_guards = Vec::with_capacity(lock_keys.len());
    for key in &lock_keys {
        persistence_guards.push(state.responses_locks.for_key(key).lock_owned().await);
    }

    if let Some(key) = idempotency_key.as_deref() {
        match state.responses_store.get_response_by_idempotency_key(key) {
            Ok(Some(stored)) if stored.request_hash == request_hash => {
                return if stream {
                    replay_stream(stored.response)
                } else {
                    Json(stored.response).into_response()
                };
            }
            Ok(Some(_)) => {
                return api_error(
                    StatusCode::CONFLICT,
                    "idempotency_key_conflict",
                    "Idempotency-Key was already used with a different Responses request"
                        .to_string(),
                    None,
                )
            }
            Ok(None) => {}
            Err(error) => return store_error(error, None),
        }
    }

    let current_input = match request.canonical_input_items() {
        Ok(items) => items,
        Err(response) => return response,
    };
    let mut context = if let Some(previous_id) = request.previous_response_id.as_deref() {
        match state.responses_store.get_response(previous_id) {
            Ok(stored) => stored.context,
            Err(error) => return store_error(error, Some("previous_response_id")),
        }
    } else if let Some(conversation_id) = conversation_id.as_deref() {
        match state.responses_store.get_conversation(conversation_id) {
            Ok(snapshot) => snapshot.items,
            Err(error) => return store_error(error, Some("conversation")),
        }
    } else {
        Vec::new()
    };
    context.extend(current_input.clone());
    if let Err(error) = super::responses_store::validate_context(&context) {
        return store_error(error, Some("input"));
    }

    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let created_at = super::unix_secs();
    let model = request
        .model
        .clone()
        .unwrap_or_else(|| "camelid-active".to_string());
    let chat_request = match request.to_chat_request(&Value::Array(context.clone())) {
        Ok(request) => request,
        Err(response) => return response,
    };
    let properties = ResponseProperties::from_request(&request, conversation_id.clone());
    let persistence = (store_response || conversation_id.is_some()).then(|| ResponsePersistence {
        store: state.responses_store.clone(),
        response_id: response_id.clone(),
        created_at,
        previous_response_id: request.previous_response_id.clone(),
        conversation_id,
        request_hash,
        idempotency_key,
        current_input,
        context,
        store_response,
        _guards: persistence_guards,
    });

    let upstream = chat_completions(State(state), Ok(Json(chat_request))).await;
    if !upstream.status().is_success() {
        return upstream;
    }
    if stream {
        return translate_stream(
            upstream,
            response_id,
            model,
            created_at,
            properties,
            persistence,
        );
    }
    let response =
        match translate_nonstreaming(upstream, response_id, model, created_at, &properties).await {
            Ok(response) => response,
            Err(response) => return response,
        };
    if let Some(persistence) = persistence {
        if let Err(error) = persistence.commit(&response) {
            return store_error(error, None);
        }
    }
    Json(response).into_response()
}

impl ResponsesRequest {
    fn to_chat_request(&self, input: &Value) -> Result<ChatCompletionRequest, Response> {
        let mut messages = Vec::new();
        if let Some(instructions) = self.instructions.as_ref() {
            let text = value_text(instructions, "instructions")?;
            if !text.is_empty() {
                messages.push(text_message("system", text));
            }
        }
        messages.extend(input_messages(input)?);
        if messages.is_empty() {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "empty_response_input",
                "Responses input must contain at least one text or function item".to_string(),
                Some("input"),
            ));
        }

        let tools = responses_tools_to_chat(self.tools.as_deref())?;
        let response_format = responses_text_format(self.text.as_ref())?;
        Ok(ChatCompletionRequest {
            model: self.model.clone(),
            camelid_expected_gguf_sha256: None,
            messages: Some(messages),
            stream: self.stream,
            max_tokens: self.max_output_tokens,
            temperature: self.temperature,
            top_k: None,
            top_p: self.top_p,
            seed: None,
            presence_penalty: None,
            frequency_penalty: None,
            min_p: None,
            repeat_penalty: None,
            penalty_last_n: None,
            typical_p: None,
            top_n_sigma: None,
            min_keep: None,
            logit_bias: None,
            stop: None::<StopSpec>,
            n: None,
            logprobs: None,
            top_logprobs: None,
            camelid_logit_token_ids: None,
            camelid_dense_diagnostics: None,
            camelid_dense_diagnostic_generated_index: None,
            camelid_context_budget_tokens: None,
            camelid_receipt: None,
            camelid_enable_thinking: None,
            camelid_image_min_tokens: None,
            camelid_image_max_tokens: None,
            tools,
            tool_choice: self.tool_choice.clone(),
            parallel_tool_calls: self.parallel_tool_calls,
            response_format,
            json_schema: None,
            grammar: None,
            stream_options: self
                .stream
                .unwrap_or(false)
                .then(|| json!({"include_usage": true})),
            unsupported_fields: HashMap::new(),
        })
    }

    fn validate_boundary_fields(&self) -> Result<(), Response> {
        if self.previous_response_id.is_some() && self.conversation.is_some() {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "mutually_exclusive_parameters",
                "previous_response_id and conversation cannot be used together".to_string(),
                Some("conversation"),
            ));
        }
        if self.background.unwrap_or(false) {
            return Err(unsupported_parameter(
                "background",
                "background Responses jobs are not implemented by the local runtime",
            ));
        }
        if !self.unsupported_fields.is_empty() {
            let mut fields = self
                .unsupported_fields
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>();
            fields.sort_unstable();
            return Err(unsupported_parameter(
                "request",
                &format!(
                    "unsupported Responses request field(s): {}",
                    fields.join(", ")
                ),
            ));
        }
        if let Some(metadata) = self.metadata.as_ref() {
            if !metadata.is_object() {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_metadata",
                    "Responses metadata must be a JSON object".to_string(),
                    Some("metadata"),
                ));
            }
        }
        Ok(())
    }

    fn conversation_id(&self) -> Result<Option<String>, Response> {
        let Some(conversation) = self.conversation.as_ref() else {
            return Ok(None);
        };
        let id = match conversation {
            Value::String(id) => Some(id.as_str()),
            Value::Object(object) => object.get("id").and_then(Value::as_str),
            _ => None,
        }
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                "invalid_conversation",
                "conversation must be a conversation id or an object containing id".to_string(),
                Some("conversation"),
            )
        })?;
        Ok(Some(id.to_string()))
    }

    fn canonical_input_items(&self) -> Result<Vec<Value>, Response> {
        let input = self.input.as_ref().ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                "missing_response_input",
                "Responses requests require an input string or input-item array".to_string(),
                Some("input"),
            )
        })?;
        match input {
            Value::String(text) => Ok(vec![json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            })]),
            Value::Array(items) => {
                // Validate through the same conversion used to build the model
                // prompt before any item is admitted to durable state.
                let _ = input_messages(input)?;
                Ok(items.clone())
            }
            _ => Err(api_error(
                StatusCode::BAD_REQUEST,
                "invalid_response_input",
                "Responses input must be a string or an array of input items".to_string(),
                Some("input"),
            )),
        }
    }
}

#[derive(Clone)]
struct ResponseProperties {
    previous_response_id: Option<String>,
    conversation_id: Option<String>,
    store: bool,
    metadata: Value,
    instructions: Option<Value>,
    max_output_tokens: Option<u32>,
    temperature: Option<f32>,
    text: Value,
    tool_choice: Value,
    tools: Vec<Value>,
    top_p: Option<f32>,
    parallel_tool_calls: bool,
}

impl ResponseProperties {
    fn from_request(request: &ResponsesRequest, conversation_id: Option<String>) -> Self {
        Self {
            previous_response_id: request.previous_response_id.clone(),
            conversation_id,
            store: request.store.unwrap_or(false),
            metadata: request.metadata.clone().unwrap_or_else(|| json!({})),
            instructions: request.instructions.clone(),
            max_output_tokens: request.max_output_tokens,
            temperature: request.temperature,
            text: request
                .text
                .clone()
                .unwrap_or_else(|| json!({"format": {"type": "text"}})),
            tool_choice: request.tool_choice.clone().unwrap_or_else(|| json!("auto")),
            tools: request.tools.clone().unwrap_or_default(),
            top_p: request.top_p,
            parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
        }
    }
}

struct ResponsePersistence {
    store: super::responses_store::ResponsesStore,
    response_id: String,
    created_at: u64,
    previous_response_id: Option<String>,
    conversation_id: Option<String>,
    request_hash: String,
    idempotency_key: Option<String>,
    current_input: Vec<Value>,
    context: Vec<Value>,
    store_response: bool,
    _guards: Vec<tokio::sync::OwnedMutexGuard<()>>,
}

impl ResponsePersistence {
    fn commit(self, response: &Value) -> Result<(), StoreError> {
        let output = response
            .get("output")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut completed_context = self.context.clone();
        completed_context.extend(output.clone());
        self.store.commit_response(ResponseCommit {
            id: &self.response_id,
            created_at: self.created_at,
            conversation_id: self.conversation_id.as_deref(),
            previous_response_id: self.previous_response_id.as_deref(),
            request_hash: &self.request_hash,
            idempotency_key: self.idempotency_key.as_deref(),
            input: &self.current_input,
            output: &output,
            context: &completed_context,
            response,
            store_response: self.store_response,
        })
    }
}

fn request_fingerprint(request: &ResponsesRequest) -> String {
    let bytes = serde_json::to_vec(request).unwrap_or_default();
    format!("{:x}", Sha256::digest(bytes))
}

fn idempotency_key(headers: &HeaderMap) -> Result<Option<String>, Response> {
    let Some(value) = headers.get("idempotency-key") else {
        return Ok(None);
    };
    let key = value.to_str().map_err(|_| {
        api_error(
            StatusCode::BAD_REQUEST,
            "invalid_idempotency_key",
            "Idempotency-Key must contain visible ASCII text".to_string(),
            None,
        )
    })?;
    if key.is_empty() || key.len() > 255 || key.chars().any(char::is_control) {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_idempotency_key",
            "Idempotency-Key must contain 1 to 255 visible characters".to_string(),
            None,
        ));
    }
    Ok(Some(key.to_string()))
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct ConversationCreateRequest {
    metadata: Option<Value>,
    items: Option<Vec<Value>>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ConversationUpdateRequest {
    metadata: Value,
}

#[derive(Debug, Deserialize)]
pub(super) struct ConversationItemsRequest {
    items: Vec<Value>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ConversationItemsQuery {
    after: Option<String>,
    limit: Option<usize>,
    order: Option<String>,
}

pub(super) async fn create_conversation(
    State(state): State<super::AppState>,
    payload: Result<Json<ConversationCreateRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(error) => return malformed_json_error(error),
    };
    let id = format!("conv_{}", uuid::Uuid::new_v4().simple());
    let created_at = super::unix_secs();
    let metadata = request.metadata.unwrap_or_else(|| json!({}));
    let items = request.items.unwrap_or_default();
    if let Err(response) = input_messages(&Value::Array(items.clone())) {
        return response;
    }
    match state
        .responses_store
        .create_conversation(&id, created_at, &metadata, &items)
    {
        Ok(snapshot) => Json(snapshot.object).into_response(),
        Err(error) => store_error(error, None),
    }
}

pub(super) async fn get_conversation(
    State(state): State<super::AppState>,
    Path(id): Path<String>,
) -> Response {
    match state.responses_store.get_conversation(&id) {
        Ok(snapshot) => Json(snapshot.object).into_response(),
        Err(error) => store_error(error, Some("conversation")),
    }
}

pub(super) async fn update_conversation(
    State(state): State<super::AppState>,
    Path(id): Path<String>,
    payload: Result<Json<ConversationUpdateRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(error) => return malformed_json_error(error),
    };
    let lock = state.responses_locks.for_key(&format!("conversation:{id}"));
    let _guard = lock.lock().await;
    match state
        .responses_store
        .update_conversation(&id, &request.metadata, super::unix_secs())
    {
        Ok(conversation) => Json(conversation).into_response(),
        Err(error) => store_error(error, Some("conversation")),
    }
}

pub(super) async fn delete_conversation(
    State(state): State<super::AppState>,
    Path(id): Path<String>,
) -> Response {
    let lock = state.responses_locks.for_key(&format!("conversation:{id}"));
    let _guard = lock.lock().await;
    match state.responses_store.delete_conversation(&id) {
        Ok(()) => Json(json!({
            "id": id,
            "object": "conversation.deleted",
            "deleted": true,
        }))
        .into_response(),
        Err(error) => store_error(error, Some("conversation")),
    }
}

pub(super) async fn add_conversation_items(
    State(state): State<super::AppState>,
    Path(id): Path<String>,
    payload: Result<Json<ConversationItemsRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(error) => return malformed_json_error(error),
    };
    if let Err(response) = input_messages(&Value::Array(request.items.clone())) {
        return response;
    }
    let lock = state.responses_locks.for_key(&format!("conversation:{id}"));
    let _guard = lock.lock().await;
    match state
        .responses_store
        .add_conversation_items(&id, super::unix_secs(), &request.items)
    {
        Ok(items) => Json(list_object(items, false)).into_response(),
        Err(error) => store_error(error, Some("items")),
    }
}

pub(super) async fn list_conversation_items(
    State(state): State<super::AppState>,
    Path(id): Path<String>,
    Query(query): Query<ConversationItemsQuery>,
) -> Response {
    let items = match state.responses_store.list_conversation_items(&id) {
        Ok(items) => items,
        Err(error) => return store_error(error, Some("conversation")),
    };
    let descending = match query.order.as_deref().unwrap_or("desc") {
        "asc" => false,
        "desc" => true,
        _ => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_order",
                "order must be asc or desc".to_string(),
                Some("order"),
            )
        }
    };
    let limit = query.limit.unwrap_or(20);
    if limit == 0 || limit > 100 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_limit",
            "limit must be between 1 and 100".to_string(),
            Some("limit"),
        );
    }
    let mut ordered = items;
    if descending {
        ordered.reverse();
    }
    if let Some(after) = query.after.as_deref() {
        let Some(index) = ordered
            .iter()
            .position(|item| item.get("id").and_then(Value::as_str) == Some(after))
        else {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_after",
                "after does not identify an item in this conversation".to_string(),
                Some("after"),
            );
        };
        ordered.drain(..=index);
    }
    let has_more = ordered.len() > limit;
    ordered.truncate(limit);
    Json(list_object(ordered, has_more)).into_response()
}

pub(super) async fn get_conversation_item(
    State(state): State<super::AppState>,
    Path((id, item_id)): Path<(String, String)>,
) -> Response {
    match state.responses_store.get_conversation_item(&id, &item_id) {
        Ok(item) => Json(item).into_response(),
        Err(error) => store_error(error, Some("item_id")),
    }
}

pub(super) async fn delete_conversation_item(
    State(state): State<super::AppState>,
    Path((id, item_id)): Path<(String, String)>,
) -> Response {
    let lock = state.responses_locks.for_key(&format!("conversation:{id}"));
    let _guard = lock.lock().await;
    match state
        .responses_store
        .delete_conversation_item(&id, &item_id, super::unix_secs())
    {
        Ok(()) => Json(json!({
            "id": item_id,
            "object": "conversation.item.deleted",
            "deleted": true,
        }))
        .into_response(),
        Err(error) => store_error(error, Some("item_id")),
    }
}

pub(super) async fn get_response(
    State(state): State<super::AppState>,
    Path(id): Path<String>,
) -> Response {
    match state.responses_store.get_response(&id) {
        Ok(stored) => Json(stored.response).into_response(),
        Err(error) => store_error(error, Some("response_id")),
    }
}

pub(super) async fn delete_response(
    State(state): State<super::AppState>,
    Path(id): Path<String>,
) -> Response {
    match state.responses_store.delete_response(&id) {
        Ok(()) => Json(json!({
            "id": id,
            "object": "response.deleted",
            "deleted": true,
        }))
        .into_response(),
        Err(error) => store_error(error, Some("response_id")),
    }
}

fn list_object(items: Vec<Value>, has_more: bool) -> Value {
    let first_id = items
        .first()
        .and_then(|item| item.get("id"))
        .cloned()
        .unwrap_or(Value::Null);
    let last_id = items
        .last()
        .and_then(|item| item.get("id"))
        .cloned()
        .unwrap_or(Value::Null);
    json!({
        "object": "list",
        "data": items,
        "first_id": first_id,
        "last_id": last_id,
        "has_more": has_more,
    })
}

fn store_error(error: StoreError, param: Option<&'static str>) -> Response {
    match error {
        StoreError::NotFound(message) => api_error(
            StatusCode::NOT_FOUND,
            "resource_not_found",
            message.to_string(),
            param,
        ),
        StoreError::Conflict(message) => api_error(
            StatusCode::CONFLICT,
            "responses_store_conflict",
            message.to_string(),
            param,
        ),
        StoreError::Limit(message) => api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "conversation_context_limit_exceeded",
            message.to_string(),
            param,
        ),
        StoreError::Invalid(message) => api_error(
            StatusCode::BAD_REQUEST,
            "invalid_conversation_item",
            message.to_string(),
            param,
        ),
        StoreError::Database(message) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "responses_store_failed",
            message,
            param,
        ),
    }
}

fn unsupported_parameter(param: &'static str, message: &str) -> Response {
    api_error(
        StatusCode::BAD_REQUEST,
        "unsupported_parameter",
        message.to_string(),
        Some(param),
    )
}

fn text_message(role: &str, content: String) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content,
        image_urls: Vec::new(),
        unsupported_content_parts: Vec::new(),
    }
}

fn input_messages(input: &Value) -> Result<Vec<ChatMessage>, Response> {
    match input {
        Value::String(text) => Ok(vec![text_message("user", text.clone())]),
        Value::Array(items) => {
            let mut messages = Vec::new();
            for item in items {
                messages.extend(input_item_messages(item)?);
            }
            Ok(messages)
        }
        _ => Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_response_input",
            "Responses input must be a string or an array of input items".to_string(),
            Some("input"),
        )),
    }
}

fn input_item_messages(item: &Value) -> Result<Vec<ChatMessage>, Response> {
    if let Value::String(text) = item {
        return Ok(vec![text_message("user", text.clone())]);
    }
    let object = item.as_object().ok_or_else(|| {
        api_error(
            StatusCode::BAD_REQUEST,
            "invalid_response_input_item",
            "each Responses input item must be a string or object".to_string(),
            Some("input"),
        )
    })?;
    let item_type = object.get("type").and_then(Value::as_str);
    match item_type {
        Some("function_call") => {
            let name = required_string(object.get("name"), "input.name")?;
            let arguments = required_json_string(object.get("arguments"), "input.arguments")?;
            let parsed_arguments: Value = serde_json::from_str(&arguments).map_err(|err| {
                api_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_function_call_arguments",
                    format!("function_call arguments must contain valid JSON: {err}"),
                    Some("input"),
                )
            })?;
            Ok(vec![text_message(
                "assistant",
                json!({"name": name, "parameters": parsed_arguments}).to_string(),
            )])
        }
        Some("function_call_output") => {
            let _call_id = required_string(object.get("call_id"), "input.call_id")?;
            let output = object.get("output").ok_or_else(|| {
                api_error(
                    StatusCode::BAD_REQUEST,
                    "missing_function_call_output",
                    "function_call_output requires output".to_string(),
                    Some("input"),
                )
            })?;
            Ok(vec![text_message(
                "tool",
                response_content_text(output, "input.output")?,
            )])
        }
        Some("message") | None if object.contains_key("role") => {
            let mut role = required_string(object.get("role"), "input.role")?;
            if role == "developer" {
                role = "system".to_string();
            }
            let content = object.get("content").ok_or_else(|| {
                api_error(
                    StatusCode::BAD_REQUEST,
                    "missing_response_message_content",
                    "Responses message items require content".to_string(),
                    Some("input"),
                )
            })?;
            Ok(vec![text_message(
                &role,
                response_content_text(content, "input.content")?,
            )])
        }
        Some("input_text") => Ok(vec![text_message(
            "user",
            required_string(object.get("text"), "input.text")?,
        )]),
        Some(other) => Err(api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_response_input_item",
            format!(
                "Responses input item type {other:?} is unsupported; Camelid accepts message, input_text, function_call, and function_call_output items"
            ),
            Some("input"),
        )),
        None => Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_response_input_item",
            "Responses input objects require a type or role".to_string(),
            Some("input"),
        )),
    }
}

fn response_content_text(value: &Value, param: &'static str) -> Result<String, Response> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                if let Value::String(value) = part {
                    text.push_str(value);
                    continue;
                }
                let object = part.as_object().ok_or_else(|| {
                    api_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_response_content",
                        "Responses content parts must be strings or typed objects".to_string(),
                        Some(param),
                    )
                })?;
                match object.get("type").and_then(Value::as_str) {
                    Some("input_text" | "output_text" | "text") => {
                        text.push_str(
                            object
                                .get("text")
                                .and_then(Value::as_str)
                                .unwrap_or_default(),
                        );
                    }
                    Some(other) => {
                        return Err(api_error(
                            StatusCode::BAD_REQUEST,
                            "unsupported_multimodal_content",
                            format!(
                                "Responses content part type {other:?} is unsupported; Camelid accepts text input only"
                            ),
                            Some(param),
                        ))
                    }
                    None => {
                        return Err(api_error(
                            StatusCode::BAD_REQUEST,
                            "invalid_response_content",
                            "Responses content part objects require a type".to_string(),
                            Some(param),
                        ))
                    }
                }
            }
            Ok(text)
        }
        other => Ok(other
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| other.to_string())),
    }
}

fn value_text(value: &Value, param: &'static str) -> Result<String, Response> {
    response_content_text(value, param)
}

fn required_string(value: Option<&Value>, field: &'static str) -> Result<String, Response> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            api_error(
                StatusCode::BAD_REQUEST,
                "invalid_response_input_item",
                format!("{field} must be a non-empty string"),
                Some("input"),
            )
        })
}

fn required_json_string(value: Option<&Value>, field: &'static str) -> Result<String, Response> {
    match value {
        Some(Value::String(value)) => Ok(value.clone()),
        Some(value) => Ok(value.to_string()),
        None => Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_response_input_item",
            format!("{field} is required"),
            Some("input"),
        )),
    }
}

fn responses_tools_to_chat(tools: Option<&[Value]>) -> Result<Option<Vec<Value>>, Response> {
    let Some(tools) = tools else {
        return Ok(None);
    };
    let mut normalized = Vec::with_capacity(tools.len());
    for tool in tools {
        let object = tool.as_object().ok_or_else(|| {
            unsupported_parameter("tools", "Responses tools must be typed objects")
        })?;
        if object.get("type").and_then(Value::as_str) != Some("function") {
            let kind = object
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            return Err(unsupported_parameter(
                "tools",
                &format!(
                    "Responses tool type {kind:?} is unsupported; only local function tools are available"
                ),
            ));
        }
        if let Some(function) = object.get("function") {
            // Accept Chat-Completions-shaped tools as a local interoperability
            // convenience, while emitting the canonical nested shape below.
            normalized.push(json!({"type": "function", "function": function}));
            continue;
        }
        let name = required_string(object.get("name"), "tools.name")?;
        let mut function = serde_json::Map::new();
        function.insert("name".to_string(), Value::String(name));
        for field in ["description", "parameters", "strict"] {
            if let Some(value) = object.get(field) {
                function.insert(field.to_string(), value.clone());
            }
        }
        normalized.push(json!({"type": "function", "function": function}));
    }
    Ok(Some(normalized))
}

fn responses_text_format(text: Option<&Value>) -> Result<Option<Value>, Response> {
    let Some(text) = text else {
        return Ok(None);
    };
    let format = text.get("format").unwrap_or(text);
    match format.get("type").and_then(Value::as_str) {
        None | Some("text") => Ok(None),
        Some("json_object") => Ok(Some(json!({"type": "json_object"}))),
        Some("json_schema") => {
            let schema = format.get("schema").cloned().ok_or_else(|| {
                api_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "Responses text.format json_schema requires schema".to_string(),
                    Some("text"),
                )
            })?;
            let mut envelope = serde_json::Map::new();
            envelope.insert("schema".to_string(), schema);
            for field in ["name", "description", "strict"] {
                if let Some(value) = format.get(field) {
                    envelope.insert(field.to_string(), value.clone());
                }
            }
            Ok(Some(json!({
                "type": "json_schema",
                "json_schema": envelope,
            })))
        }
        Some(other) => Err(unsupported_parameter(
            "text",
            &format!(
                "Responses text format {other:?} is unsupported; use text, json_object, or json_schema"
            ),
        )),
    }
}

async fn translate_nonstreaming(
    upstream: Response,
    response_id: String,
    requested_model: String,
    created_at: u64,
    properties: &ResponseProperties,
) -> Result<Value, Response> {
    let (parts, body) = upstream.into_parts();
    let bytes = match to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(err) => {
            return Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "responses_adapter_failed",
                format!("could not read the chat completion response: {err}"),
                None,
            ))
        }
    };
    let chat: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(_) => return Err(Response::from_parts(parts, Body::from(bytes))),
    };
    let model = chat
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&requested_model);
    let message = &chat["choices"][0]["message"];
    let output = response_output_items(message);
    let usage = responses_usage(chat.get("usage"));
    let incomplete = chat["choices"][0]["finish_reason"] == "length";
    Ok(response_object(
        &response_id,
        model,
        created_at,
        if incomplete {
            "incomplete"
        } else {
            "completed"
        },
        output,
        usage,
        incomplete.then_some("max_output_tokens"),
        properties,
    ))
}

fn response_output_items(message: &Value) -> Vec<Value> {
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        return calls
            .iter()
            .map(|call| {
                let call_id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("call_camelid");
                json!({
                    "id": format!("fc_{}", uuid::Uuid::new_v4().simple()),
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                })
            })
            .collect();
    }
    let text = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    vec![message_output_item(
        &format!("msg_{}", uuid::Uuid::new_v4().simple()),
        "completed",
        text,
    )]
}

fn message_output_item(id: &str, status: &str, text: &str) -> Value {
    json!({
        "id": id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": [],
            "logprobs": [],
        }],
    })
}

fn responses_usage(chat_usage: Option<&Value>) -> Value {
    let input = chat_usage
        .and_then(|usage| usage.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = chat_usage
        .and_then(|usage| usage.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    json!({
        "input_tokens": input,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input + output,
    })
}

#[allow(clippy::too_many_arguments)]
fn response_object(
    id: &str,
    model: &str,
    created_at: u64,
    status: &str,
    output: Vec<Value>,
    usage: Value,
    incomplete_reason: Option<&str>,
    properties: &ResponseProperties,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "completed_at": (status == "completed").then_some(super::unix_secs()),
        "error": null,
        "incomplete_details": incomplete_reason.map(|reason| json!({"reason": reason})),
        "instructions": properties.instructions,
        "max_output_tokens": properties.max_output_tokens,
        "model": model,
        "output": output,
        "parallel_tool_calls": properties.parallel_tool_calls,
        "previous_response_id": properties.previous_response_id,
        "conversation": properties.conversation_id.as_ref().map(|id| json!({"id": id})),
        "store": properties.store,
        "temperature": properties.temperature,
        "text": properties.text,
        "tool_choice": properties.tool_choice,
        "tools": properties.tools,
        "top_p": properties.top_p,
        "truncation": "disabled",
        "usage": usage,
        "metadata": properties.metadata,
    })
}

fn translate_stream(
    upstream: Response,
    response_id: String,
    model: String,
    created_at: u64,
    properties: ResponseProperties,
    persistence: Option<ResponsePersistence>,
) -> Response {
    let (_, body) = upstream.into_parts();
    let mut upstream = body.into_data_stream();
    let events = async_stream::stream! {
        let mut sequence = 0u64;
        let mut state = ResponsesStreamState::new(response_id, model, created_at, properties);
        let mut persistence = persistence;
        for event in state.start_events(&mut sequence) {
            yield response_event(event, &mut sequence);
        }
        let mut buffer = String::new();
        while let Some(chunk) = upstream.next().await {
            match chunk {
                Ok(bytes) => buffer.push_str(&String::from_utf8_lossy(&bytes)),
                Err(err) => {
                    yield response_event(json!({
                        "type": "error",
                        "code": "responses_adapter_stream_failed",
                        "message": err.to_string(),
                        "param": null,
                    }), &mut sequence);
                    return;
                }
            }
            while let Some(frame_end) = buffer.find("\n\n") {
                let frame = buffer[..frame_end].to_string();
                buffer.drain(..frame_end + 2);
                let Some(data) = sse_data(&frame) else {
                    continue;
                };
                if data == "[DONE]" {
                    let terminal_events = state.finish_events(&mut sequence);
                    if let Some(persistence) = persistence.take() {
                        let response = terminal_events
                            .last()
                            .and_then(|event| event.get("response"))
                            .cloned()
                            .unwrap_or(Value::Null);
                        if let Err(error) = persistence.commit(&response) {
                            yield response_event(json!({
                                "type": "error",
                                "code": "responses_store_failed",
                                "message": error.to_string(),
                                "param": null,
                            }), &mut sequence);
                            return;
                        }
                    }
                    for event in terminal_events {
                        yield response_event(event, &mut sequence);
                    }
                    return;
                }
                let Ok(chunk): Result<Value, _> = serde_json::from_str(&data) else {
                    continue;
                };
                if let Some(error) = chunk.get("error") {
                    yield response_event(json!({
                        "type": "error",
                        "code": error.get("code").and_then(Value::as_str).unwrap_or("generation_error"),
                        "message": error.get("message").and_then(Value::as_str).unwrap_or("generation failed"),
                        "param": error.get("param").cloned().unwrap_or(Value::Null),
                    }), &mut sequence);
                    return;
                }
                for event in state.accept_chat_chunk(&chunk, &mut sequence) {
                    yield response_event(event, &mut sequence);
                }
            }
        }
        let terminal_events = state.finish_events(&mut sequence);
        if let Some(persistence) = persistence.take() {
            let response = terminal_events
                .last()
                .and_then(|event| event.get("response"))
                .cloned()
                .unwrap_or(Value::Null);
            if let Err(error) = persistence.commit(&response) {
                yield response_event(json!({
                    "type": "error",
                    "code": "responses_store_failed",
                    "message": error.to_string(),
                    "param": null,
                }), &mut sequence);
                return;
            }
        }
        for event in terminal_events {
            yield response_event(event, &mut sequence);
        }
    };
    Sse::new(events).into_response()
}

fn replay_stream(response: Value) -> Response {
    let events = async_stream::stream! {
        let mut sequence = 0u64;
        let mut created = response.clone();
        created["status"] = json!("in_progress");
        created["completed_at"] = Value::Null;
        yield response_event(
            json!({"type": "response.created", "response": created}),
            &mut sequence,
        );
        let event_type = if response["status"] == "incomplete" {
            "response.incomplete"
        } else {
            "response.completed"
        };
        yield response_event(
            json!({"type": event_type, "response": response}),
            &mut sequence,
        );
    };
    Sse::new(events).into_response()
}

fn sse_data(frame: &str) -> Option<String> {
    let data = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>();
    (!data.is_empty()).then(|| data.join("\n"))
}

fn response_event(mut event: Value, sequence: &mut u64) -> Result<Event, Infallible> {
    if event.get("sequence_number").is_none() {
        event["sequence_number"] = json!(*sequence);
    }
    *sequence += 1;
    let event_type = event
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("response.event")
        .to_string();
    Ok(Event::default().event(event_type).data(event.to_string()))
}

struct ResponsesStreamState {
    response_id: String,
    model: String,
    created_at: u64,
    message_id: String,
    message_started: bool,
    message_finished: bool,
    text: String,
    output: Vec<Value>,
    usage: Value,
    finish_reason: Option<String>,
    properties: ResponseProperties,
}

impl ResponsesStreamState {
    fn new(
        response_id: String,
        model: String,
        created_at: u64,
        properties: ResponseProperties,
    ) -> Self {
        Self {
            response_id,
            model,
            created_at,
            message_id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            message_started: false,
            message_finished: false,
            text: String::new(),
            output: Vec::new(),
            usage: responses_usage(None),
            finish_reason: None,
            properties,
        }
    }

    fn start_events(&self, _sequence: &mut u64) -> Vec<Value> {
        let response = response_object(
            &self.response_id,
            &self.model,
            self.created_at,
            "in_progress",
            Vec::new(),
            Value::Null,
            None,
            &self.properties,
        );
        vec![
            json!({"type": "response.created", "response": response}),
            json!({"type": "response.in_progress", "response": response_object(
                &self.response_id,
                &self.model,
                self.created_at,
                "in_progress",
                Vec::new(),
                Value::Null,
                None,
                &self.properties,
            )}),
        ]
    }

    fn ensure_message_started(&mut self, events: &mut Vec<Value>) {
        if self.message_started {
            return;
        }
        self.message_started = true;
        events.push(json!({
            "type": "response.output_item.added",
            "output_index": self.output.len(),
            "item": message_output_item(&self.message_id, "in_progress", ""),
        }));
        events.push(json!({
            "type": "response.content_part.added",
            "item_id": self.message_id,
            "output_index": self.output.len(),
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
        }));
    }

    fn accept_chat_chunk(&mut self, chunk: &Value, _sequence: &mut u64) -> Vec<Value> {
        if let Some(usage) = chunk.get("usage").filter(|usage| !usage.is_null()) {
            self.usage = responses_usage(Some(usage));
        }
        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|v| v.first())
        else {
            return Vec::new();
        };
        let mut events = Vec::new();
        let delta = &choice["delta"];
        if let Some(text) = delta.get("content").and_then(Value::as_str) {
            self.ensure_message_started(&mut events);
            self.text.push_str(text);
            events.push(json!({
                "type": "response.output_text.delta",
                "item_id": self.message_id,
                "output_index": self.output.len(),
                "content_index": 0,
                "delta": text,
                "logprobs": [],
            }));
        }
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                let call_id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("call_camelid")
                    .to_string();
                let item_id = format!("fc_{}", uuid::Uuid::new_v4().simple());
                let name = call["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let arguments = call["function"]["arguments"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let output_index = self.output.len();
                let item = json!({
                    "id": item_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": call_id,
                    "name": name,
                    "arguments": "",
                });
                events.push(json!({
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": item,
                }));
                events.push(json!({
                    "type": "response.function_call_arguments.delta",
                    "item_id": item_id,
                    "output_index": output_index,
                    "delta": arguments,
                }));
                events.push(json!({
                    "type": "response.function_call_arguments.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "name": name,
                    "arguments": arguments,
                }));
                let completed = json!({
                    "id": item_id,
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                });
                events.push(json!({
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": completed,
                }));
                self.output.push(completed);
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(reason.to_string());
        }
        events
    }

    fn finish_message(&mut self, events: &mut Vec<Value>) {
        if !self.message_started || self.message_finished {
            return;
        }
        self.message_finished = true;
        let output_index = self.output.len();
        events.push(json!({
            "type": "response.output_text.done",
            "item_id": self.message_id,
            "output_index": output_index,
            "content_index": 0,
            "text": self.text,
            "logprobs": [],
        }));
        events.push(json!({
            "type": "response.content_part.done",
            "item_id": self.message_id,
            "output_index": output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": self.text, "annotations": [], "logprobs": []},
        }));
        let item = message_output_item(&self.message_id, "completed", &self.text);
        events.push(json!({
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": item,
        }));
        self.output.push(item);
    }

    fn finish_events(&mut self, _sequence: &mut u64) -> Vec<Value> {
        let mut events = Vec::new();
        if !self.message_started && self.output.is_empty() {
            self.ensure_message_started(&mut events);
        }
        self.finish_message(&mut events);
        let incomplete = matches!(self.finish_reason.as_deref(), Some("length"));
        let status = if incomplete {
            "incomplete"
        } else {
            "completed"
        };
        let reason = incomplete.then_some("max_output_tokens");
        events.push(json!({
            "type": if incomplete { "response.incomplete" } else { "response.completed" },
            "response": response_object(
                &self.response_id,
                &self.model,
                self.created_at,
                status,
                self.output.clone(),
                self.usage.clone(),
                reason,
                &self.properties,
            ),
        }));
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_function_call_continuation_and_flat_tools() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "local",
            "input": [
                {"role": "user", "content": "weather?"},
                {"type": "function_call", "call_id": "call_1", "name": "weather", "arguments": "{\"city\":\"Paris\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "{\"temp\":25}"}
            ],
            "tools": [{
                "type": "function",
                "name": "weather",
                "description": "Get weather",
                "parameters": {"type": "object"}
            }]
        }))
        .unwrap();
        let input = request.input.clone().unwrap();
        let chat = request.to_chat_request(&input).unwrap();
        let messages = chat.messages.unwrap();
        assert_eq!(messages[1].role, "assistant");
        assert!(messages[1].content.contains("\"name\":\"weather\""));
        assert_eq!(messages[2].role, "tool");
        assert_eq!(chat.tools.unwrap()[0]["function"]["name"], "weather");
    }

    #[test]
    fn stream_accumulator_emits_text_and_function_events() {
        let properties = ResponseProperties::from_request(
            &serde_json::from_value(json!({"input":"hello"})).unwrap(),
            None,
        );
        let mut state =
            ResponsesStreamState::new("resp_test".into(), "local".into(), 1, properties.clone());
        let mut sequence = 0;
        let text_events = state.accept_chat_chunk(
            &json!({"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}),
            &mut sequence,
        );
        assert!(text_events
            .iter()
            .any(|event| event["type"] == "response.output_text.delta"));

        let mut tool_state =
            ResponsesStreamState::new("resp_tool".into(), "local".into(), 1, properties);
        let tool_events = tool_state.accept_chat_chunk(
            &json!({"choices":[{"delta":{"tool_calls":[{
                "index":0,
                "id":"call_1",
                "type":"function",
                "function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}
            }]},"finish_reason":null}]}),
            &mut sequence,
        );
        assert!(tool_events
            .iter()
            .any(|event| event["type"] == "response.function_call_arguments.delta"));
        assert_eq!(tool_state.output[0]["call_id"], "call_1");
    }

    #[test]
    fn accepts_stateful_previous_response_id_at_the_request_boundary() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "input": "hello",
            "previous_response_id": "resp_prior"
        }))
        .unwrap();
        assert!(request.validate_boundary_fields().is_ok());
        assert_eq!(request.previous_response_id.as_deref(), Some("resp_prior"));
    }

    #[test]
    fn rejects_previous_response_and_conversation_together() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "input": "hello",
            "previous_response_id": "resp_prior",
            "conversation": "conv_test"
        }))
        .unwrap();
        let response = request.validate_boundary_fields().unwrap_err();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}

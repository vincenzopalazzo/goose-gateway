//! OpenAI-compatible gateway in front of the goose SDK.
//!
//! A browser app cannot link `goose`, a Rust crate, so this process sits between them: it
//! accepts `POST /v1/chat/completions` in the OpenAI wire format most chat clients already
//! speak, and answers through goose's own `xai_oauth` provider. That provider owns the X
//! subscription credential and refreshes it itself, which a page cannot do because
//! `auth.x.ai` sends no CORS headers.
//!
//! With `"stream": true` the reply comes back as OpenAI-style server-sent events, fed straight
//! from goose's own message stream, so text reaches the client as the model writes it.
//!
//! Nothing here holds a secret: it reads the token cache `goose configure` wrote, under the
//! directory `GOOSE_PATH_ROOT` points at (`<root>/config/xai_oauth/tokens.json`).

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::{header, HeaderValue, Method, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use goose::conversation::message::{Message, MessageContentBlock};
use goose::providers::base::{Provider, ProviderUsage};
use goose_provider_types::model::ModelConfig;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ContentBlock, ErrorCode, ErrorData, Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::Mutex;
use tower_http::cors::{AllowOrigin, CorsLayer};

const PROVIDER: &str = "xai_oauth";
const DEFAULT_MODEL: &str = "grok-4.7";

// ---------------------------------------------------------------------------
// OpenAI wire format (only what a chat client with tools actually sends).
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<InMessage>,
    #[serde(default)]
    tools: Vec<InTool>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
}

#[derive(Deserialize, Default)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
struct InMessage {
    role: String,
    #[serde(default)]
    content: Option<Value>,
    #[serde(default)]
    tool_calls: Vec<InToolCall>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

#[derive(Deserialize)]
struct InToolCall {
    id: String,
    function: InFunction,
}

#[derive(Deserialize)]
struct InFunction {
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize)]
struct InTool {
    function: InToolDef,
}

#[derive(Deserialize)]
struct InToolDef {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    parameters: Option<Value>,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: u32,
    message: OutMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct OutMessage {
    role: &'static str,
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OutToolCall>,
}

#[derive(Serialize)]
struct OutToolCall {
    id: String,
    r#type: &'static str,
    function: OutFunction,
}

#[derive(Serialize)]
struct OutFunction {
    name: String,
    arguments: String,
}

#[derive(Serialize, Default, Clone, Copy)]
struct Usage {
    prompt_tokens: i64,
    completion_tokens: i64,
    total_tokens: i64,
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    r#type: &'static str,
}

// ---------------------------------------------------------------------------
// Conversion into goose's types. Mirrors what goose-sdk's bindings do.
// ---------------------------------------------------------------------------

/// Text content arrives as a string or as `[{type:"text", text}]` parts.
fn text_of(content: &Option<Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn json_object(v: &str) -> Map<String, Value> {
    match serde_json::from_str::<Value>(if v.is_empty() { "{}" } else { v }) {
        Ok(Value::Object(m)) => m,
        _ => Map::new(),
    }
}

/// Split the OpenAI transcript into (system prompt, goose messages).
///
/// OpenAI represents a tool result as a standalone `tool` role message keyed by
/// `tool_call_id`; goose attaches it to a `user` message as a tool response block.
/// Consecutive tool results are folded into one such message.
fn to_goose(messages: Vec<InMessage>) -> (String, Vec<Message>) {
    let mut system = String::new();
    let mut out: Vec<Message> = Vec::new();

    for m in messages {
        match m.role.as_str() {
            "system" => {
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(&text_of(&m.content));
            }
            "user" => out.push(Message::user().with_text(text_of(&m.content))),
            "assistant" => {
                let mut msg = Message::assistant();
                let text = text_of(&m.content);
                if !text.is_empty() {
                    msg = msg.with_text(text);
                }
                for call in m.tool_calls {
                    let params = CallToolRequestParams::new(call.function.name)
                        .with_arguments(json_object(&call.function.arguments));
                    msg = msg.with_tool_request(call.id, Ok(params));
                }
                out.push(msg);
            }
            "tool" => {
                let id = m.tool_call_id.unwrap_or_default();
                let text = text_of(&m.content);
                let result = if text.starts_with("Error:") {
                    Err(ErrorData::new(ErrorCode::INTERNAL_ERROR, text, None))
                } else {
                    Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
                };
                // Fold into the previous user-role tool-response message when there is one.
                match out.last_mut() {
                    Some(last)
                        if last.role == rmcp::model::Role::User
                            && last
                                .content
                                .iter()
                                .all(|c| matches!(c, MessageContentBlock::ToolResponse(_))) =>
                    {
                        let taken = std::mem::replace(last, Message::user());
                        *last = taken.with_tool_response(id, result);
                    }
                    _ => out.push(Message::user().with_tool_response(id, result)),
                }
            }
            _ => {}
        }
    }
    (system, out)
}

fn to_tools(tools: Vec<InTool>) -> Vec<Tool> {
    tools
        .into_iter()
        .map(|t| {
            let schema = match t.function.parameters {
                Some(Value::Object(m)) => m,
                _ => {
                    let mut m = Map::new();
                    m.insert("type".into(), Value::String("object".into()));
                    m
                }
            };
            Tool::new(t.function.name, t.function.description, schema)
        })
        .collect()
}

/// A goose tool request as an OpenAI tool call; `None` when goose could not parse the call.
fn out_tool_call(req: goose::conversation::message::ToolRequest) -> Option<OutToolCall> {
    let call = req.tool_call.ok()?;
    Some(OutToolCall {
        id: req.id,
        r#type: "function",
        function: OutFunction {
            name: call.name.to_string(),
            arguments: serde_json::to_string(&call.arguments.unwrap_or_default())
                .unwrap_or_else(|_| "{}".into()),
        },
    })
}

fn out_usage(usage: &ProviderUsage) -> Usage {
    let u = &usage.usage;
    Usage {
        prompt_tokens: u.input_tokens.unwrap_or_default() as i64,
        completion_tokens: u.output_tokens.unwrap_or_default() as i64,
        total_tokens: u.total_tokens.unwrap_or_default() as i64,
    }
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or_default()
}

/// Pull text and tool calls out of goose's reply.
fn from_goose(reply: Message) -> (Option<String>, Vec<OutToolCall>) {
    let mut text_parts = Vec::new();
    let mut calls = Vec::new();
    for block in reply.content {
        match block {
            MessageContentBlock::Text(t) if !t.text.is_empty() => text_parts.push(t.text),
            MessageContentBlock::ToolRequest(req) => calls.extend(out_tool_call(req)),
            _ => {}
        }
    }
    let text = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join("\n"))
    };
    (text, calls)
}

/// Turns goose's stream items into OpenAI `chat.completion.chunk` objects.
///
/// goose yields text in small pieces but tool calls only once complete, so each tool call
/// goes out whole in a single delta, with the `index` OpenAI clients key their assembly on.
struct ChunkWriter {
    id: String,
    created: i64,
    model: String,
    include_usage: bool,
    /// The first delta carries `role`, as OpenAI's does.
    started: bool,
    tool_calls: usize,
    usage: Usage,
}

impl ChunkWriter {
    fn new(model: String, include_usage: bool) -> Self {
        let created = unix_now();
        Self {
            id: format!("chatcmpl-{created}"),
            created,
            model,
            include_usage,
            started: false,
            tool_calls: 0,
            usage: Usage::default(),
        }
    }

    fn chunk(&self, choices: Value) -> Value {
        serde_json::json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": choices,
        })
    }

    fn delta(&mut self, mut delta: Value) -> Value {
        if !self.started {
            self.started = true;
            delta["role"] = Value::from("assistant");
        }
        self.chunk(serde_json::json!([{ "index": 0, "delta": delta, "finish_reason": null }]))
    }

    /// Chunks for one item of goose's stream.
    fn item(&mut self, message: Option<Message>, usage: Option<ProviderUsage>) -> Vec<Value> {
        if let Some(u) = usage {
            self.usage = out_usage(&u);
        }
        let Some(message) = message else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for block in message.content {
            match block {
                MessageContentBlock::Text(t) if !t.text.is_empty() => {
                    out.push(self.delta(serde_json::json!({ "content": t.text })));
                }
                MessageContentBlock::ToolRequest(req) => {
                    if let Some(call) = out_tool_call(req) {
                        let mut call = serde_json::to_value(call).unwrap_or_default();
                        call["index"] = Value::from(self.tool_calls);
                        self.tool_calls += 1;
                        out.push(self.delta(serde_json::json!({ "tool_calls": [call] })));
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// The closing chunk with the finish reason, plus the usage chunk when asked for.
    fn finish(&mut self) -> Vec<Value> {
        let reason = if self.tool_calls > 0 {
            "tool_calls"
        } else {
            "stop"
        };
        let mut out =
            vec![self
                .chunk(serde_json::json!([{ "index": 0, "delta": {}, "finish_reason": reason }]))];
        if self.include_usage {
            let mut usage = self.chunk(serde_json::json!([]));
            usage["usage"] = serde_json::to_value(self.usage).unwrap_or_default();
            out.push(usage);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Which browser origins may use the gateway
// ---------------------------------------------------------------------------

/// Browser origins allowed to call the gateway.
///
/// The gateway spends the operator's subscription and has no login of its own, so it must not
/// answer whatever site the operator happens to have open. Local development origins
/// (`localhost`, `127.0.0.1`, `[::1]`, any port) are always allowed; others are listed in
/// `GOOSE_GATEWAY_ALLOWED_ORIGINS`, comma separated, or `*` to allow every origin.
///
/// Requests without an `Origin` header (curl, server-side clients) are not browser
/// cross-site requests and pass: anything that can make them on this machine could read the
/// token file directly anyway.
#[derive(Clone, Default)]
struct Origins {
    any: bool,
    extra: Vec<String>,
}

impl Origins {
    fn parse(list: &str) -> Self {
        let mut origins = Origins::default();
        for entry in list.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            if entry == "*" {
                origins.any = true;
            } else {
                origins
                    .extra
                    .push(entry.trim_end_matches('/').to_ascii_lowercase());
            }
        }
        origins
    }

    fn allows(&self, origin: &str) -> bool {
        let origin = origin.to_ascii_lowercase();
        self.any || is_local_origin(&origin) || self.extra.contains(&origin)
    }
}

/// `http(s)://localhost`, `127.0.0.1` or `[::1]`, with or without a port.
fn is_local_origin(origin: &str) -> bool {
    let Some(rest) = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
    else {
        return false;
    };
    let host = if rest.starts_with('[') {
        rest.split_inclusive(']').next().unwrap_or(rest)
    } else {
        rest.split(':').next().unwrap_or(rest)
    };
    let port = &rest[host.len()..];
    let port_ok = port.is_empty()
        || port
            .strip_prefix(':')
            .is_some_and(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
    matches!(host, "localhost" | "127.0.0.1" | "[::1]") && port_ok
}

/// Refuse browser requests from origins not on the list.
///
/// CORS alone only stops a page from reading the answer. This also stops the request from
/// running, which matters for a DNS-rebinding page that the browser treats as same-origin.
async fn check_origin(
    State(origins): State<Arc<Origins>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if let Some(origin) = request.headers().get(header::ORIGIN) {
        let allowed = origin.to_str().is_ok_and(|o| origins.allows(o));
        if !allowed {
            let shown = origin.to_str().unwrap_or("?").to_string();
            tracing::warn!("refused request from origin {shown}");
            return error(
                StatusCode::FORBIDDEN,
                format!(
                    "Origin {shown} may not use this gateway. Add it to GOOSE_GATEWAY_ALLOWED_ORIGINS."
                ),
                "permission_error",
            );
        }
    }
    next.run(request).await
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

struct AppState {
    /// Built lazily on first use so a missing sign-in is reported per request, not at boot.
    provider: Mutex<Option<Arc<dyn Provider>>>,
}

async fn provider(state: &AppState) -> Result<Arc<dyn Provider>> {
    let mut slot = state.provider.lock().await;
    if let Some(p) = slot.as_ref() {
        return Ok(p.clone());
    }
    let p = goose::providers::create(PROVIDER, Vec::new())
        .await
        .context(
            "building goose's xai_oauth provider; run `goose configure` and choose xai_oauth first",
        )?;
    *slot = Some(p.clone());
    Ok(p)
}

/// Where goose's `xai_oauth` provider keeps the X subscription tokens.
fn token_cache_path() -> std::path::PathBuf {
    goose::config::paths::Paths::in_config_dir("xai_oauth/tokens.json")
}

/// True when a usable sign-in exists: the cache parses and holds a refresh token.
///
/// Building the provider proves nothing about this: goose constructs `xai_oauth` without
/// looking for a token and only fails on the first request. The access token's expiry does
/// not matter here, because goose renews it from the refresh token on use.
fn credential_present_at(path: &std::path::Path) -> bool {
    let Ok(raw) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<Value>(&raw) else {
        return false;
    };
    value
        .get("refresh_token")
        .and_then(Value::as_str)
        .is_some_and(|t| !t.trim().is_empty())
}

const NOT_SIGNED_IN: &str =
    "goose is not signed in to xAI. On the host, run `goose configure` and choose xai_oauth.";

fn error(status: StatusCode, message: impl Into<String>, kind: &'static str) -> Response {
    (
        status,
        Json(ErrorBody {
            error: ErrorDetail {
                message: message.into(),
                r#type: kind,
            },
        }),
    )
        .into_response()
}

fn looks_like_auth_failure(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    [
        "401",
        "403",
        "unauthenticated",
        "authentication",
        "credential",
        "incorrect api key",
        "expired",
        "not configured",
    ]
    .iter()
    .any(|k| m.contains(k))
}

/// Map a goose failure to an OpenAI error, as a status plus body.
///
/// A stale provider instance keeps a stale token, so an auth failure drops the cached one and
/// the next call rebuilds it.
async fn upstream_failure(state: &AppState, message: String) -> (StatusCode, ErrorBody) {
    let (status, kind) = if looks_like_auth_failure(&message) {
        *state.provider.lock().await = None;
        (StatusCode::UNAUTHORIZED, "authentication_error")
    } else {
        (StatusCode::BAD_GATEWAY, "upstream_error")
    };
    (
        status,
        ErrorBody {
            error: ErrorDetail {
                message,
                r#type: kind,
            },
        },
    )
}

async fn chat(State(state): State<Arc<AppState>>, Json(req): Json<ChatRequest>) -> Response {
    let model_name = req
        .model
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let stream = req.stream;
    let include_usage = req.stream_options.unwrap_or_default().include_usage;
    let (system, messages) = to_goose(req.messages);
    let tools = to_tools(req.tools);
    let model = ModelConfig::new(&model_name);

    if !credential_present_at(&token_cache_path()) {
        return error(
            StatusCode::UNAUTHORIZED,
            NOT_SIGNED_IN,
            "authentication_error",
        );
    }

    let provider = match provider(&state).await {
        Ok(p) => p,
        Err(e) => {
            return error(
                StatusCode::UNAUTHORIZED,
                format!("{e:#}"),
                "authentication_error",
            )
        }
    };

    if stream {
        return stream_chat(
            state,
            provider,
            model,
            model_name,
            system,
            messages,
            tools,
            include_usage,
        )
        .await;
    }

    let (reply, usage) = match provider.complete(&model, &system, &messages, &tools).await {
        Ok(v) => v,
        Err(e) => {
            let (status, body) = upstream_failure(&state, e.to_string()).await;
            return (status, Json(body)).into_response();
        }
    };

    let (content, tool_calls) = from_goose(reply);
    let finish_reason = if tool_calls.is_empty() {
        "stop"
    } else {
        "tool_calls"
    };
    let created = unix_now();

    Json(ChatResponse {
        id: format!("chatcmpl-{created}"),
        object: "chat.completion",
        created,
        model: model_name,
        choices: vec![Choice {
            index: 0,
            message: OutMessage {
                role: "assistant",
                content,
                tool_calls,
            },
            finish_reason,
        }],
        usage: out_usage(&usage),
    })
    .into_response()
}

fn sse_json(value: &impl Serialize) -> Event {
    Event::default().data(serde_json::to_string(value).unwrap_or_else(|_| "{}".into()))
}

/// Answer as server-sent events straight from goose's message stream.
#[allow(clippy::too_many_arguments)]
async fn stream_chat(
    state: Arc<AppState>,
    provider: Arc<dyn Provider>,
    model: ModelConfig,
    model_name: String,
    system: String,
    messages: Vec<Message>,
    tools: Vec<Tool>,
    include_usage: bool,
) -> Response {
    let mut upstream = match provider.stream(&model, &system, &messages, &tools).await {
        Ok(s) => s,
        Err(e) => {
            let (status, body) = upstream_failure(&state, e.to_string()).await;
            return (status, Json(body)).into_response();
        }
    };
    // Wait for the first item before committing to a 200: a sign-in or upstream failure that
    // surfaces there still gets a real status code, which is what clients branch on.
    let first = match upstream.next().await {
        Some(Err(e)) => {
            let (status, body) = upstream_failure(&state, e.to_string()).await;
            return (status, Json(body)).into_response();
        }
        other => other,
    };

    // A task drives goose and the response drains the channel. When the client goes away the
    // send fails, the task returns, and dropping goose's stream cancels the upstream request.
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);
    tokio::spawn(async move {
        let mut writer = ChunkWriter::new(model_name, include_usage);
        let mut next = first;
        while let Some(item) = next {
            match item {
                Ok((message, usage)) => {
                    for chunk in writer.item(message, usage) {
                        if tx.send(sse_json(&chunk)).await.is_err() {
                            return;
                        }
                    }
                }
                Err(e) => {
                    // Headers are gone by now, so the failure travels in-band, the way
                    // OpenAI reports mid-stream errors. No [DONE]: the reply is incomplete.
                    let (_, body) = upstream_failure(&state, e.to_string()).await;
                    let _ = tx.send(sse_json(&body)).await;
                    return;
                }
            }
            next = upstream.next().await;
        }
        for chunk in writer.finish() {
            if tx.send(sse_json(&chunk)).await.is_err() {
                return;
            }
        }
        let _ = tx.send(Event::default().data("[DONE]")).await;
    });

    let events = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|event| (Ok::<_, Infallible>(event), rx))
    });
    Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    let credential = credential_present_at(&token_cache_path());
    // Only worth building the provider when there is something to build it with.
    let provider_ok = credential && provider(&state).await.is_ok();
    Json(serde_json::json!({
        "ok": true,
        "provider": PROVIDER,
        "credential": credential && provider_ok,
    }))
    .into_response()
}

async fn models() -> Response {
    let data: Vec<Value> = [
        "grok-4.7",
        "grok-4.6",
        "grok-4.5",
        "grok-code-fast-1",
        "grok-3",
        "grok-3-mini",
    ]
    .iter()
    .map(|id| serde_json::json!({ "id": id, "object": "model", "owned_by": "xai" }))
    .collect();
    Json(serde_json::json!({ "object": "list", "data": data })).into_response()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?),
        )
        .init();

    let host = std::env::var("GOOSE_GATEWAY_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("GOOSE_GATEWAY_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8791);
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .context("bad listen address")?;

    let origins = Arc::new(Origins::parse(
        &std::env::var("GOOSE_GATEWAY_ALLOWED_ORIGINS").unwrap_or_default(),
    ));
    let cors_origins = origins.clone();
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(move |origin, _| {
            origin.to_str().is_ok_and(|o| cors_origins.allows(o))
        }))
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);

    let state = Arc::new(AppState {
        provider: Mutex::new(None),
    });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .with_state(state)
        .layer(axum::middleware::from_fn_with_state(
            origins.clone(),
            check_origin,
        ))
        .layer(cors)
        .layer(axum::middleware::map_response(
            |mut r: Response| async move {
                r.headers_mut()
                    .insert("x-goose-gateway", HeaderValue::from_static("1"));
                r
            },
        ));

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    tracing::info!("goose-gateway listening on http://{addr} (provider {PROVIDER})");
    if origins.any {
        tracing::warn!("GOOSE_GATEWAY_ALLOWED_ORIGINS=*: any website can use this gateway");
    } else if !origins.extra.is_empty() {
        tracing::info!(
            "allowed origins beyond localhost: {}",
            origins.extra.join(", ")
        );
    }
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(v: serde_json::Value) -> Vec<InMessage> {
        serde_json::from_value(v).expect("test fixture deserializes")
    }

    #[test]
    fn text_accepts_string_and_parts() {
        assert_eq!(text_of(&Some(Value::String("hi".into()))), "hi");
        let parts =
            serde_json::json!([{ "type": "text", "text": "a" }, { "type": "text", "text": "b" }]);
        assert_eq!(text_of(&Some(parts)), "a\nb");
        assert_eq!(text_of(&None), "");
    }

    #[test]
    fn malformed_arguments_become_empty_object() {
        assert!(json_object("").is_empty());
        assert!(json_object("{oops").is_empty());
        assert!(json_object("\"a string\"").is_empty());
        assert_eq!(json_object(r#"{"a":1}"#).get("a"), Some(&Value::from(1)));
    }

    #[test]
    fn system_is_split_out_and_roles_map() {
        let (system, out) = to_goose(msgs(serde_json::json!([
            { "role": "system", "content": "sys" },
            { "role": "user", "content": "hi" },
            { "role": "assistant", "content": "hello" }
        ])));
        assert_eq!(system, "sys");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, rmcp::model::Role::User);
        assert_eq!(out[1].role, rmcp::model::Role::Assistant);
    }

    #[test]
    fn assistant_tool_calls_and_tool_results_round_trip() {
        // OpenAI shape: assistant with two tool_calls, then two standalone tool messages.
        let (_, out) = to_goose(msgs(serde_json::json!([
            { "role": "user", "content": "do it" },
            { "role": "assistant", "content": null, "tool_calls": [
                { "id": "c1", "type": "function", "function": { "name": "a", "arguments": "{\"x\":1}" } },
                { "id": "c2", "type": "function", "function": { "name": "b", "arguments": "" } }
            ]},
            { "role": "tool", "tool_call_id": "c1", "content": "ok" },
            { "role": "tool", "tool_call_id": "c2", "content": "Error: no route" }
        ])));
        // user, assistant(with 2 requests), ONE user message holding both responses.
        assert_eq!(
            out.len(),
            3,
            "consecutive tool results fold into one message"
        );
        let requests = out[1]
            .content
            .iter()
            .filter(|c| matches!(c, MessageContentBlock::ToolRequest(_)))
            .count();
        assert_eq!(requests, 2);
        let responses: Vec<_> = out[2]
            .content
            .iter()
            .filter_map(|c| match c {
                MessageContentBlock::ToolResponse(r) => Some(r),
                _ => None,
            })
            .collect();
        assert_eq!(responses.len(), 2);
        assert!(responses[0].tool_result.is_ok(), "plain text is a success");
        assert!(
            responses[1].tool_result.is_err(),
            "an Error: prefix is a failure"
        );
    }

    #[test]
    fn reply_text_and_tool_calls_are_extracted() {
        let mut args = Map::new();
        args.insert("amount_sats".into(), Value::from(50000));
        let reply = Message::assistant()
            .with_text("doing it")
            .with_tool_request(
                "call-1",
                Ok(CallToolRequestParams::new("create_invoice").with_arguments(args)),
            );
        let (text, calls) = from_goose(reply);
        assert_eq!(text.as_deref(), Some("doing it"));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call-1");
        assert_eq!(calls[0].function.name, "create_invoice");
        // Arguments go out as a JSON string, which is what OpenAI clients parse.
        assert_eq!(calls[0].function.arguments, r#"{"amount_sats":50000}"#);
    }

    fn deltas(chunks: &[Value]) -> Vec<&Value> {
        chunks.iter().map(|c| &c["choices"][0]["delta"]).collect()
    }

    #[test]
    fn stream_text_goes_out_as_content_deltas_then_stop() {
        let mut w = ChunkWriter::new("grok-4.7".into(), false);
        let mut out = w.item(Some(Message::assistant().with_text("Hel")), None);
        out.extend(w.item(Some(Message::assistant().with_text("lo")), None));
        out.extend(w.item(None, None)); // usage-only items produce nothing
        let d = deltas(&out);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0]["role"], "assistant", "the first delta names the role");
        assert_eq!(d[0]["content"], "Hel");
        assert!(d[1].get("role").is_none());
        assert_eq!(d[1]["content"], "lo");
        assert_eq!(out[0]["object"], "chat.completion.chunk");

        let end = w.finish();
        assert_eq!(end.len(), 1, "no usage chunk unless asked for");
        assert_eq!(end[0]["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn stream_tool_calls_are_whole_and_indexed() {
        let mut w = ChunkWriter::new("grok-4.7".into(), false);
        let call = |id: &str, name: &str| {
            Message::assistant()
                .with_tool_request(id, Ok(CallToolRequestParams::new(name.to_string())))
        };
        let mut out = w.item(Some(call("c1", "get_balances")), None);
        out.extend(w.item(Some(call("c2", "list_channels")), None));
        let d = deltas(&out);
        assert_eq!(d[0]["tool_calls"][0]["index"], 0);
        assert_eq!(d[0]["tool_calls"][0]["id"], "c1");
        assert_eq!(d[0]["tool_calls"][0]["function"]["name"], "get_balances");
        assert_eq!(d[0]["tool_calls"][0]["function"]["arguments"], "{}");
        assert_eq!(d[1]["tool_calls"][0]["index"], 1);
        assert_eq!(w.finish()[0]["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn stream_usage_chunk_only_when_requested() {
        let mut w = ChunkWriter::new("grok-4.7".into(), true);
        let end = w.finish();
        assert_eq!(end.len(), 2);
        assert_eq!(end[1]["choices"], serde_json::json!([]));
        assert_eq!(end[1]["usage"]["total_tokens"], 0);
    }

    #[test]
    fn stream_flag_is_read_from_the_request() {
        let req: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages": [], "stream": true, "stream_options": { "include_usage": true }
        }))
        .unwrap();
        assert!(req.stream);
        assert!(req.stream_options.unwrap().include_usage);
        let plain: ChatRequest =
            serde_json::from_value(serde_json::json!({ "messages": [] })).unwrap();
        assert!(!plain.stream, "non-streaming stays the default");
    }

    #[test]
    fn local_origins_are_allowed_by_default() {
        let o = Origins::parse("");
        for ok in [
            "http://localhost:5173",
            "http://127.0.0.1:4321",
            "http://localhost",
            "https://localhost:8443",
            "http://[::1]:3000",
            "HTTP://LOCALHOST:5173",
        ] {
            assert!(o.allows(ok), "{ok}");
        }
        for bad in [
            "https://evil.example",
            "http://localhost.evil.example",
            "http://127.0.0.1.evil.example",
            "http://localhost:80evil",
            "http://localhost:",
            "null",
            "file://",
        ] {
            assert!(!o.allows(bad), "{bad}");
        }
    }

    #[test]
    fn listed_origins_and_wildcard() {
        let o = Origins::parse(" https://ldk-server-manager.pages.dev/ , https://b.example ");
        assert!(o.allows("https://ldk-server-manager.pages.dev"));
        assert!(o.allows("https://b.example"));
        assert!(
            !o.allows("https://ldk-server-manager.pages.dev.evil.example"),
            "exact match only"
        );
        assert!(Origins::parse("*").allows("https://anything.example"));
    }

    #[test]
    fn tool_schemas_default_to_an_object() {
        let tools = to_tools(vec![
            serde_json::from_value(serde_json::json!({ "type": "function", "function": { "name": "t", "description": "d" } })).unwrap(),
        ]);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "t");
    }

    fn temp_file(name: &str, contents: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("goose-gateway-test-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokens.json");
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn credential_requires_a_refresh_token() {
        let good = temp_file(
            "good",
            r#"{"access_token":"a","refresh_token":"r","expires_at":"2020-01-01T00:00:00Z"}"#,
        );
        assert!(
            credential_present_at(&good),
            "an expired access token is fine: goose renews it"
        );
        let empty = temp_file("empty", r#"{"access_token":"a","refresh_token":"  "}"#);
        assert!(!credential_present_at(&empty));
        let garbage = temp_file("garbage", "not json");
        assert!(!credential_present_at(&garbage));
        assert!(!credential_present_at(std::path::Path::new(
            "/definitely/not/here/tokens.json"
        )));
    }

    #[test]
    fn not_configured_is_an_auth_failure() {
        assert!(looks_like_auth_failure("Provider is not configured"));
    }

    #[test]
    fn auth_failures_are_recognised_in_both_shapes_xai_uses() {
        assert!(looks_like_auth_failure(
            "403 Forbidden: unauthenticated:bad-credentials"
        ));
        assert!(looks_like_auth_failure(
            "Bad request (400): Incorrect API key provided"
        ));
        assert!(!looks_like_auth_failure("rate limit exceeded"));
    }
}

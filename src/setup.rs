//! Provider setup from a client: list what goose can use, save or remove its settings, and
//! run a subscription sign-in.
//!
//! Mirrors what goose's own desktop server does (`acp/server/providers.rs`), over plain HTTP:
//! the same catalog, the same config and secret storage, and goose's own `configure_oauth`
//! for sign-ins, so nothing here re-implements a provider's authentication.
//!
//! Sign-ins come in two shapes:
//!
//! - **Device code** (GitHub Copilot, Kimi): goose reports the code through
//!   `with_device_code_announce`, and it is streamed to the client to show.
//! - **Browser redirect** (xAI, ChatGPT, Gemini, Hugging Face): goose opens a sign-in page and
//!   waits for the redirect on a localhost port. Natively that opens the operator's browser and
//!   completes on its own. Headless (Docker) there is no browser, so goose logs the page's
//!   address instead; the log tap below hands it to the client. The redirect then lands on the
//!   client's machine, where nothing listens, so the client pastes that address back and the
//!   gateway replays it to goose's listener.
//!
//! goose reports some of this only in its log (the address above, and xAI's device-code
//! fallback), so while a sign-in runs a tracing layer forwards those lines to it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use goose::config::Config;
use goose::providers::catalog::{ProviderSetupCategory, ProviderSetupMethod};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{error, AppState};

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct ProviderView {
    id: String,
    name: String,
    description: String,
    /// How goose sets the provider up, e.g. `oauth_browser`, `single_api_key`, `local`.
    method: ProviderSetupMethod,
    /// goose's own "show by default" grouping; the rest are niche or enterprise.
    featured: bool,
    docs_url: Option<String>,
    /// Signs in through `/sign-in` rather than with fields.
    sign_in: bool,
    fields: Vec<FieldView>,
    configured: bool,
    default_model: String,
    models: Vec<String>,
}

#[derive(Serialize)]
struct FieldView {
    key: String,
    label: String,
    secret: bool,
    required: bool,
    placeholder: Option<String>,
    default: Option<String>,
    /// Whether a value is stored. Secret values are never returned.
    set: bool,
    /// The stored value, for non-secret fields only (a host URL, say).
    value: Option<String>,
}

/// Whether goose considers the provider usable. Can touch the filesystem or keychain, so it
/// runs off the async threads.
pub async fn is_configured(id: &str) -> bool {
    let Ok(entry) = goose::providers::get_from_registry(id).await else {
        return false;
    };
    tokio::task::spawn_blocking(move || entry.inventory_configured())
        .await
        .unwrap_or(false)
}

fn field_value(key: &str) -> Option<String> {
    match Config::global().get_param::<Value>(key) {
        Ok(Value::String(s)) if !s.is_empty() => Some(s),
        Ok(Value::Null) | Err(_) => None,
        Ok(other) => Some(other.to_string()),
    }
}

async fn view(id: &str) -> Option<ProviderView> {
    let entries = goose::providers::catalog::get_setup_catalog_entries().await;
    let entry = entries.into_iter().find(|e| e.provider_id == id)?;
    let registry = goose::providers::get_from_registry(id).await.ok()?;
    let metadata = registry.metadata().clone();
    let sign_in = metadata.config_keys.iter().any(|k| k.oauth_flow);

    let config = Config::global();
    let fields = entry
        .fields
        .into_iter()
        .map(|f| {
            let (set, value) = if f.secret {
                (config.get_secret::<String>(&f.key).is_ok(), None)
            } else {
                let value = field_value(&f.key);
                (value.is_some(), value)
            };
            FieldView {
                key: f.key,
                label: f.label,
                secret: f.secret,
                required: f.required,
                placeholder: f.placeholder,
                default: f.default_value,
                set,
                value,
            }
        })
        .collect();

    Some(ProviderView {
        id: entry.provider_id,
        name: entry.display_name,
        description: entry.description,
        method: entry.setup_method,
        featured: matches!(
            entry.group,
            goose::providers::catalog::ProviderSetupGroup::Default
        ),
        docs_url: entry.docs_url,
        sign_in,
        fields,
        configured: is_configured(id).await,
        default_model: metadata.default_model.clone(),
        models: metadata
            .known_models
            .iter()
            .map(|m| m.name.clone())
            .collect(),
    })
}

/// True for providers a chat client can use: goose's "model" providers. The "agent" ones
/// (Claude Code, Codex CLI…) run their own tools and do not fit a tool-calling client.
pub async fn is_model_provider(id: &str) -> bool {
    goose::providers::catalog::get_setup_catalog_entries()
        .await
        .iter()
        .any(|e| e.provider_id == id && matches!(e.category, ProviderSetupCategory::Model))
}

pub async fn list(State(_): State<Arc<AppState>>) -> Response {
    let ids: Vec<String> = goose::providers::catalog::get_setup_catalog_entries()
        .await
        .into_iter()
        .filter(|e| matches!(e.category, ProviderSetupCategory::Model))
        .map(|e| e.provider_id)
        .collect();
    let mut providers = Vec::with_capacity(ids.len());
    for id in ids {
        providers.extend(view(&id).await);
    }
    providers.sort_by_key(|p| p.name.to_lowercase());
    Json(serde_json::json!({ "providers": providers })).into_response()
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct SaveRequest {
    fields: HashMap<String, String>,
}

/// Store fields the provider declares. Secrets go to goose's secret storage, the rest to its
/// config file, exactly as `goose configure` would.
pub async fn save(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<SaveRequest>,
) -> Response {
    let Ok(entry) = goose::providers::get_from_registry(&id).await else {
        return unknown(&id);
    };
    let keys = entry.metadata().config_keys.clone();
    let config = Config::global();
    let mut params = Vec::new();
    let mut secrets = Vec::new();
    for (key, value) in req.fields {
        let Some(declared) = keys.iter().find(|k| k.name == key) else {
            return error(
                StatusCode::BAD_REQUEST,
                format!("{id} has no setting called {key}"),
                "invalid_request_error",
            );
        };
        let value = value.trim().to_string();
        if value.is_empty() {
            return error(
                StatusCode::BAD_REQUEST,
                format!("{key} cannot be empty; remove the provider's settings instead"),
                "invalid_request_error",
            );
        }
        if declared.secret {
            secrets.push((key, Value::String(value)));
        } else {
            params.push((key, value));
        }
    }
    for (key, value) in params {
        if let Err(e) = config.set_param(&key, &value) {
            return storage_error(e);
        }
    }
    if let Err(e) = config.set_secret_values(&secrets) {
        return storage_error(e);
    }
    config.invalidate_secrets_cache();
    state.forget_provider(&id).await;
    tracing::info!("saved settings for {id}");
    respond_with_view(&id).await
}

/// Remove a provider's settings and sign-in, like goose's own "remove".
pub async fn remove(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let Ok(entry) = goose::providers::get_from_registry(&id).await else {
        return unknown(&id);
    };
    let config = Config::global();
    let mut secret_keys = Vec::new();
    for key in &entry.metadata().config_keys {
        if key.secret {
            secret_keys.push(key.name.clone());
        } else if let Err(e) = config.delete(&key.name) {
            return storage_error(e);
        }
    }
    if let Err(e) = config.delete_secret_values(&secret_keys) {
        return storage_error(e);
    }
    if let Err(e) = goose::providers::cleanup_provider(&id).await {
        return storage_error(e);
    }
    config.invalidate_secrets_cache();
    state.forget_provider(&id).await;
    tracing::info!("removed settings for {id}");
    respond_with_view(&id).await
}

async fn respond_with_view(id: &str) -> Response {
    match view(id).await {
        Some(v) => Json(v).into_response(),
        None => unknown(id),
    }
}

fn unknown(id: &str) -> Response {
    error(
        StatusCode::NOT_FOUND,
        format!("goose has no provider called {id}"),
        "invalid_request_error",
    )
}

fn storage_error(e: impl std::fmt::Display) -> Response {
    error(
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("could not update goose's configuration: {e}"),
        "server_error",
    )
}

// ---------------------------------------------------------------------------
// Sign-in
// ---------------------------------------------------------------------------

/// The one sign-in that may run at a time. goose's redirect flows each bind a fixed port, and
/// two concurrent attempts would fight over it.
pub struct ActiveSignIn {
    serial: u64,
    provider: String,
    task: tokio::task::AbortHandle,
}

static SERIAL: AtomicU64 = AtomicU64::new(1);

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SignInEvent {
    Started {
        provider: String,
    },
    /// Show this code and link; goose is polling for the result.
    DeviceCode {
        user_code: String,
        verification_uri: String,
        expires_in: u64,
    },
    /// goose wants this page opened. If the page it redirects to does not load, paste that
    /// address to `/sign-in/callback`.
    OpenUrl {
        url: String,
    },
    Done {
        provider: Box<ProviderView>,
    },
    Error {
        message: String,
    },
}

fn sse(event: &SignInEvent) -> Event {
    Event::default().data(serde_json::to_string(event).unwrap_or_else(|_| "{}".into()))
}

/// Clears the active sign-in when the stream ends, which also aborts it if the client left.
struct SignInGuard {
    state: Arc<AppState>,
    serial: u64,
}

impl Drop for SignInGuard {
    fn drop(&mut self) {
        let mut slot = self.state.sign_in.lock().unwrap_or_else(|p| p.into_inner());
        if slot.as_ref().is_some_and(|s| s.serial == self.serial) {
            if let Some(active) = slot.take() {
                active.task.abort();
                set_log_tap(None);
            }
        }
    }
}

/// Start a sign-in and stream its progress as server-sent events.
pub async fn sign_in(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let Ok(entry) = goose::providers::get_from_registry(&id).await else {
        return unknown(&id);
    };
    if !entry.metadata().config_keys.iter().any(|k| k.oauth_flow) {
        return error(
            StatusCode::BAD_REQUEST,
            format!("{id} does not sign in with an account; save its settings instead"),
            "invalid_request_error",
        );
    }
    if let Some(active) = state
        .sign_in
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
    {
        return error(
            StatusCode::CONFLICT,
            format!(
                "A sign-in to {} is already in progress; finish or cancel it first",
                active.provider
            ),
            "conflict_error",
        );
    }
    let provider = match entry.create_with_default_model(Vec::new()).await {
        Ok(p) => p,
        Err(e) => {
            return error(
                StatusCode::BAD_GATEWAY,
                format!("goose could not prepare {id}: {e:#}"),
                "upstream_error",
            )
        }
    };

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SignInEvent>();
    let _ = tx.send(SignInEvent::Started {
        provider: id.clone(),
    });

    let serial = SERIAL.fetch_add(1, Ordering::Relaxed);
    let task_state = state.clone();
    let task_id = id.clone();
    let task_tx = tx.clone();
    let task = tokio::spawn(async move {
        let announce_tx = task_tx.clone();
        let announce: Box<dyn Fn(String, String, u64) + Send + Sync> =
            Box::new(move |user_code, verification_uri, expires_in| {
                let _ = announce_tx.send(SignInEvent::DeviceCode {
                    user_code,
                    verification_uri,
                    expires_in,
                });
            });
        set_log_tap(Some(task_tx.clone()));
        let result = goose::providers::oauth_device_flow::with_device_code_announce(
            announce,
            provider.configure_oauth(),
        )
        .await;
        set_log_tap(None);

        Config::global().invalidate_secrets_cache();
        task_state.forget_provider(&task_id).await;
        let event = match result {
            Ok(()) => match view(&task_id).await {
                Some(v) => {
                    tracing::info!("signed in to {task_id}");
                    SignInEvent::Done {
                        provider: Box::new(v),
                    }
                }
                None => SignInEvent::Error {
                    message: format!("{task_id} disappeared from goose's catalog"),
                },
            },
            Err(e) => SignInEvent::Error {
                message: e.to_string(),
            },
        };
        let _ = task_tx.send(event);
    });

    *state.sign_in.lock().unwrap_or_else(|p| p.into_inner()) = Some(ActiveSignIn {
        serial,
        provider: id,
        task: task.abort_handle(),
    });
    drop(tx);

    // The guard lives as long as the stream: when the client goes away, the sign-in stops.
    let guard = SignInGuard { state, serial };
    let events =
        futures::stream::unfold((rx, guard, false), |(mut rx, guard, finished)| async move {
            if finished {
                return None;
            }
            let event = rx.recv().await?;
            let last = matches!(event, SignInEvent::Done { .. } | SignInEvent::Error { .. });
            Some((
                Ok::<_, std::convert::Infallible>(sse(&event)),
                (rx, guard, last),
            ))
        });
    Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[derive(Deserialize)]
pub struct CallbackRequest {
    url: String,
}

/// Deliver the address a redirect sign-in landed on to goose's listener.
///
/// Only while a sign-in to this provider runs, and only to a loopback address: the gateway
/// replays the request on its own machine, which is the whole point, so it must not become a
/// way to reach anything else.
pub async fn sign_in_callback(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<CallbackRequest>,
) -> Response {
    let active = state
        .sign_in
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
        .is_some_and(|s| s.provider == id);
    if !active {
        return error(
            StatusCode::CONFLICT,
            format!("No sign-in to {id} is waiting; start it again"),
            "conflict_error",
        );
    }
    let target = match loopback_callback(&req.url) {
        Ok(t) => t,
        Err(msg) => return error(StatusCode::BAD_REQUEST, msg, "invalid_request_error"),
    };
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .redirect(reqwest::redirect::Policy::none())
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                e.to_string(),
                "server_error",
            )
        }
    };
    match client.get(&target).send().await {
        Ok(res) if res.status().is_success() => {
            Json(serde_json::json!({ "ok": true })).into_response()
        }
        Ok(res) => error(
            StatusCode::BAD_REQUEST,
            format!(
                "goose did not accept that address (HTTP {}). Copy the whole address from the page you landed on.",
                res.status().as_u16()
            ),
            "invalid_request_error",
        ),
        Err(e) => error(
            StatusCode::BAD_GATEWAY,
            format!("goose is not waiting at that address any more ({e}); start the sign-in again"),
            "upstream_error",
        ),
    }
}

/// `http://localhost:1455/auth/callback?code=…` -> the same URL on 127.0.0.1, or why not.
pub fn loopback_callback(raw: &str) -> Result<String, String> {
    let url = url::Url::parse(raw.trim())
        .map_err(|_| "That is not a web address. Copy it from the browser's address bar.")?;
    if url.scheme() != "http" {
        return Err("A sign-in redirect address starts with http://".into());
    }
    let loopback = match url.host() {
        Some(url::Host::Domain(d)) => d.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        None => false,
    };
    if !loopback {
        return Err("That is not the sign-in redirect: it should start with http://localhost or http://127.0.0.1".into());
    }
    let port = url
        .port()
        .ok_or("The address is missing its port, e.g. http://127.0.0.1:56121/…")?;
    if url.query().is_none_or(str::is_empty) {
        return Err("The address is missing the part after '?'; copy the whole address".into());
    }
    let mut target = url.clone();
    target
        .set_host(Some("127.0.0.1"))
        .map_err(|_| "Could not rewrite the address".to_string())?;
    target
        .set_port(Some(port))
        .map_err(|_| "Could not rewrite the address".to_string())?;
    Ok(target.to_string())
}

// ---------------------------------------------------------------------------
// Forwarding goose's sign-in log lines
// ---------------------------------------------------------------------------

/// Where goose's sign-in log lines go while a sign-in runs.
static LOG_TAP: std::sync::Mutex<Option<tokio::sync::mpsc::UnboundedSender<SignInEvent>>> =
    std::sync::Mutex::new(None);

fn set_log_tap(tx: Option<tokio::sync::mpsc::UnboundedSender<SignInEvent>>) {
    *LOG_TAP.lock().unwrap_or_else(|p| p.into_inner()) = tx;
}

/// A tracing layer that forwards goose's "open this page" and "enter this code" lines to the
/// sign-in in progress. Installed next to the log formatter in `main`.
pub struct SignInLogTap;

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for SignInLogTap {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if !event.metadata().target().starts_with("goose") {
            return;
        }
        let tap = LOG_TAP.lock().unwrap_or_else(|p| p.into_inner());
        let Some(tx) = tap.as_ref() else { return };
        let mut message = MessageText::default();
        event.record(&mut message);
        if let Some(found) = parse_sign_in_line(&message.0) {
            let _ = tx.send(found);
        }
    }
}

#[derive(Default)]
struct MessageText(String);

impl tracing::field::Visit for MessageText {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0 = format!("{value:?}");
        }
    }
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.0 = value.to_string();
        }
    }
}

/// Recognise goose's two sign-in log lines:
///
/// - `Please open this URL in your browser…:\n<url>` (a redirect flow that found no browser)
/// - `… open <uri> and enter code <code>` (xAI's device-code fallback)
fn parse_sign_in_line(message: &str) -> Option<SignInEvent> {
    if message.contains("open this URL in your browser") {
        let url = message.lines().last()?.trim();
        if url.starts_with("https://") {
            return Some(SignInEvent::OpenUrl {
                url: url.to_string(),
            });
        }
    }
    let (_, rest) = message.split_once(" open ")?;
    let (uri, code) = rest.split_once(" and enter code ")?;
    let (uri, code) = (uri.trim(), code.trim());
    if uri.starts_with("https://") && !code.is_empty() && !code.contains(' ') {
        return Some(SignInEvent::DeviceCode {
            user_code: code.to_string(),
            verification_uri: uri.to_string(),
            // goose does not log the lifetime; the client shows no countdown for 0.
            expires_in: 0,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_accepts_loopback_redirects_only() {
        assert_eq!(
            loopback_callback("http://localhost:1455/auth/callback?code=abc&state=x").unwrap(),
            "http://127.0.0.1:1455/auth/callback?code=abc&state=x"
        );
        assert_eq!(
            loopback_callback(" http://127.0.0.1:56121/callback?code=1 ").unwrap(),
            "http://127.0.0.1:56121/callback?code=1"
        );
        for bad in [
            "https://localhost:1455/cb?code=1",
            "http://evil.example:1455/cb?code=1",
            "http://localhost.evil.example:1455/cb?code=1",
            "http://10.0.0.1:1455/cb?code=1",
            "http://localhost/cb?code=1",
            "http://localhost:1455/cb",
            "not a url",
        ] {
            assert!(loopback_callback(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn sign_in_log_lines_are_recognised() {
        match parse_sign_in_line(
            "Please open this URL in your browser to authorize goose with xAI:\nhttps://auth.x.ai/oauth2/authorize?x=1",
        ) {
            Some(SignInEvent::OpenUrl { url }) => {
                assert_eq!(url, "https://auth.x.ai/oauth2/authorize?x=1")
            }
            _ => panic!("expected an address"),
        }
        match parse_sign_in_line(
            "xAI device authorization: open https://accounts.x.ai/oauth2/device and enter code ABCD-1234",
        ) {
            Some(SignInEvent::DeviceCode {
                user_code,
                verification_uri,
                ..
            }) => {
                assert_eq!(user_code, "ABCD-1234");
                assert_eq!(verification_uri, "https://accounts.x.ai/oauth2/device");
            }
            _ => panic!("expected a device code"),
        }
        assert!(parse_sign_in_line("Token refreshed successfully").is_none());
        assert!(parse_sign_in_line("Please open this URL in your browser:\nhttp://evil").is_none());
    }
}

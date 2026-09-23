//! Tools from an MCP server, served over HTTP for a client-side agent.
//!
//! With `GOOSE_GATEWAY_MCP_COMMAND` set, the gateway runs that command as a stdio MCP server
//! (the command inherits the gateway's environment) and serves its tools: `GET /v1/tools` lists
//! them in the shape chat requests take, and `POST /v1/tools/{name}` calls one with a JSON
//! object of arguments. The model never runs a tool through the gateway: the client passes the
//! tools to `/v1/chat/completions`, gets tool calls back, decides which to run (asking its user
//! first for anything that is not read-only) and calls them here.
//!
//! The gateway does no approval of its own, so whatever can reach it can call every tool. When
//! the tools move money, put the gateway behind a login before turning this on.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use rmcp::model::{CallToolRequestParams, CallToolResponse, CallToolResult, Tool};
use rmcp::service::{RoleClient, RunningService, ServiceExt};
use rmcp::transport::TokioChildProcess;
use rmcp::{Peer, ServiceError};
use serde::Serialize;
use serde_json::{Map, Value};
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::{error, AppState};

/// How long a tool call may take. A payment can take a while to settle or fail; past this the
/// client is told the outcome is unknown, not that it failed.
const CALL_TIMEOUT: Duration = Duration::from_secs(120);

/// How long the server gets to start and finish the MCP handshake. Requests queue behind a
/// start, so a server that launches but never answers must not hold them forever.
const START_TIMEOUT: Duration = Duration::from_secs(30);

/// How long listing the tools may take. It has no side effects, so a server that does not
/// answer in time is restarted.
const LIST_TIMEOUT: Duration = Duration::from_secs(30);

/// A started server, numbered so a failure seen on an old one cannot tear down its successor.
struct Session {
    generation: u64,
    running: RunningService<RoleClient, ()>,
}

/// A handle to one session of the server.
struct Handle {
    generation: u64,
    peer: Peer<RoleClient>,
    /// For a tool call: it counts as in flight from the moment the session was chosen, under
    /// the same lock a reset takes, so a reset can never slip in between.
    in_flight: Option<InFlight>,
}

/// One MCP server, started on first use and restarted after its transport closes.
pub struct McpTools {
    program: String,
    args: Vec<String>,
    session: Mutex<Option<Session>>,
    started: AtomicU64,
    start_timeout: Duration,
    /// Tool calls in progress. A slow listing is no reason to restart a server that is busy
    /// running one: that would cut a call (a payment) off halfway.
    calls_in_flight: Arc<AtomicUsize>,
}

/// Counts a tool call as in progress for as long as it lives. It is held by the task running
/// the call, so a call the HTTP handler stopped waiting for still counts until the server
/// answers or goes away.
struct InFlight(Arc<AtomicUsize>);

impl InFlight {
    fn new(counter: &Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Self(Arc::clone(counter))
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A request error that means the connection to the server is gone, rather than the server
/// answering with a JSON-RPC error while still healthy.
fn transport_gone(e: &ServiceError, peer: &Peer<RoleClient>) -> bool {
    matches!(
        e,
        ServiceError::TransportClosed | ServiceError::TransportSend(_)
    ) || peer.is_transport_closed()
}

impl McpTools {
    /// From `GOOSE_GATEWAY_MCP_COMMAND`: a program and its arguments, split on whitespace.
    pub fn from_command(command: &str) -> Option<Self> {
        let mut words = command.split_whitespace().map(str::to_owned);
        let program = words.next()?;
        Some(Self {
            program,
            args: words.collect(),
            session: Mutex::new(None),
            started: AtomicU64::new(0),
            start_timeout: START_TIMEOUT,
            calls_in_flight: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn program(&self) -> &str {
        &self.program
    }

    /// A handle to the running server, starting it if it is not running. `for_call` marks a
    /// tool call in flight before the session lock is released.
    async fn handle(&self, for_call: bool) -> Result<Handle, String> {
        let mut session = self.session.lock().await;
        let in_flight = || for_call.then(|| InFlight::new(&self.calls_in_flight));
        if let Some(s) = session.as_ref() {
            if !s.running.is_transport_closed() {
                return Ok(Handle {
                    generation: s.generation,
                    peer: s.running.peer().clone(),
                    in_flight: in_flight(),
                });
            }
            tracing::warn!("MCP server {} exited; starting it again", self.program);
        }
        let mut command = Command::new(&self.program);
        // A start that times out drops the child: make sure that kills it, rather than leaving
        // one hung process behind per request.
        command.args(&self.args).kill_on_drop(true);
        let transport = TokioChildProcess::new(command)
            .map_err(|e| format!("could not start {}: {e}", self.program))?;
        let running = match tokio::time::timeout(self.start_timeout, ().serve(transport)).await {
            Ok(Ok(running)) => running,
            Ok(Err(e)) => {
                return Err(format!(
                    "{} did not start as an MCP server: {e}",
                    self.program
                ))
            }
            Err(_) => {
                return Err(format!(
                    "{} did not finish the MCP handshake within {}s",
                    self.program,
                    self.start_timeout.as_secs()
                ))
            }
        };
        let generation = self.started.fetch_add(1, Ordering::Relaxed) + 1;
        let peer = running.peer().clone();
        *session = Some(Session {
            generation,
            running,
        });
        Ok(Handle {
            generation,
            peer,
            in_flight: in_flight(),
        })
    }

    /// Forget the given session, if it is still the current one, so the next request starts
    /// a fresh server. A newer session started by another request is left alone.
    async fn reset(&self, generation: u64) {
        let mut session = self.session.lock().await;
        if session.as_ref().is_some_and(|s| s.generation == generation) {
            if let Some(s) = session.take() {
                let _ = s.running.cancel().await;
            }
        }
    }

    /// Like `reset`, but only when no tool call is in flight, decided under the session lock
    /// that calls take to start. Returns false (and leaves the session) when one is running.
    async fn reset_if_idle(&self, generation: u64) -> bool {
        let mut session = self.session.lock().await;
        if self.calls_in_flight.load(Ordering::SeqCst) > 0 {
            return false;
        }
        if session.as_ref().is_some_and(|s| s.generation == generation) {
            if let Some(s) = session.take() {
                let _ = s.running.cancel().await;
            }
        }
        true
    }

    async fn list(&self) -> Result<Vec<Tool>, String> {
        // Listing has no side effects, so a server whose connection died since the last request
        // gets one retry. Calls never do: a call that broke mid-way may still have run.
        // A server that stops answering is treated like one whose connection died.
        for attempt in 0..2 {
            let handle = self.handle(false).await?;
            let failure =
                match tokio::time::timeout(LIST_TIMEOUT, handle.peer.list_all_tools()).await {
                    Ok(Ok(tools)) => return Ok(tools),
                    Ok(Err(e)) if transport_gone(&e, &handle.peer) => {
                        // Dead: no call can finish on it either.
                        self.reset(handle.generation).await;
                        format!("{e}")
                    }
                    Ok(Err(e)) => return Err(format!("listing tools failed: {e}")),
                    // A server that answers one request at a time may just be busy with a call.
                    Err(_) if !self.reset_if_idle(handle.generation).await => return Err(format!(
                        "listing tools failed: no answer within {}s while a tool call is running",
                        LIST_TIMEOUT.as_secs()
                    )),
                    Err(_) => format!("no answer within {}s", LIST_TIMEOUT.as_secs()),
                };
            tracing::warn!("listing MCP tools failed ({failure}), restarted the server");
            if attempt == 1 {
                return Err(format!("listing tools failed: {failure}"));
            }
        }
        unreachable!("the second attempt returns")
    }
}

/// A tool as `GET /v1/tools` returns it: the OpenAI function shape plus MCP's hints.
#[derive(Serialize)]
struct ToolInfo {
    name: String,
    description: String,
    /// JSON Schema for the arguments, as `function.parameters` in a chat request.
    parameters: Value,
    /// MCP's `readOnlyHint`, when the server gives one. Only a hint: the server says so.
    #[serde(skip_serializing_if = "Option::is_none")]
    read_only: Option<bool>,
    /// MCP's `destructiveHint`, when the server gives one.
    #[serde(skip_serializing_if = "Option::is_none")]
    destructive: Option<bool>,
}

impl From<Tool> for ToolInfo {
    fn from(tool: Tool) -> Self {
        let annotations = tool.annotations.as_ref();
        Self {
            name: tool.name.into_owned(),
            description: tool.description.map(|d| d.into_owned()).unwrap_or_default(),
            parameters: Value::Object((*tool.input_schema).clone()),
            read_only: annotations.and_then(|a| a.read_only_hint),
            destructive: annotations.and_then(|a| a.destructive_hint),
        }
    }
}

#[derive(Serialize, Debug, PartialEq)]
struct CallOutput {
    /// The tool's text content, or its structured content as JSON when it has no text.
    output: String,
    /// The server reported the call as failed; `output` says why.
    is_error: bool,
}

impl From<CallToolResult> for CallOutput {
    fn from(result: CallToolResult) -> Self {
        let text: Vec<&str> = result
            .content
            .iter()
            .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
            .collect();
        let output = if !text.is_empty() {
            text.join("\n")
        } else {
            result
                .structured_content
                .as_ref()
                .map(Value::to_string)
                .unwrap_or_default()
        };
        Self {
            output,
            is_error: result.is_error.unwrap_or(false),
        }
    }
}

fn not_configured() -> Response {
    error(
        StatusCode::NOT_FOUND,
        "no tools: set GOOSE_GATEWAY_MCP_COMMAND to serve an MCP server's tools",
        "not_found",
    )
}

/// `GET /v1/tools`.
pub async fn list(State(state): State<std::sync::Arc<AppState>>) -> Response {
    let Some(tools) = state.tools.as_ref() else {
        return not_configured();
    };
    match tools.list().await {
        Ok(list) => {
            let tools: Vec<ToolInfo> = list.into_iter().map(ToolInfo::from).collect();
            Json(serde_json::json!({ "tools": tools })).into_response()
        }
        Err(e) => error(StatusCode::BAD_GATEWAY, e, "upstream_error"),
    }
}

/// `POST /v1/tools/{name}` with the arguments as a JSON object (`{}` for none).
pub async fn call(
    State(state): State<std::sync::Arc<AppState>>,
    Path(name): Path<String>,
    Json(arguments): Json<Map<String, Value>>,
) -> Response {
    let Some(tools) = state.tools.as_ref() else {
        return not_configured();
    };
    let handle = match tools.handle(true).await {
        Ok(h) => h,
        Err(e) => return error(StatusCode::BAD_GATEWAY, e, "upstream_error"),
    };
    tracing::info!("tool call: {name}");
    let params = CallToolRequestParams::new(name.clone()).with_arguments(arguments);
    // The call runs in its own task holding the in-flight guard: past the timeout it keeps
    // going (the 504 says it may still complete), and a slow listing must not restart the
    // server under it.
    let in_flight = handle.in_flight;
    let peer = handle.peer.clone();
    let call = tokio::spawn(async move {
        let _in_flight = in_flight;
        peer.call_tool_once(params).await
    });
    let answer = match tokio::time::timeout(CALL_TIMEOUT, call).await {
        Ok(Ok(answer)) => Ok(answer),
        Ok(Err(join)) => {
            return error(
                StatusCode::BAD_GATEWAY,
                format!("{name} failed and may or may not have run: {join}"),
                "upstream_error",
            )
        }
        Err(elapsed) => Err(elapsed),
    };
    match answer {
        Ok(Ok(CallToolResponse::Complete(result))) => {
            Json(CallOutput::from(result)).into_response()
        }
        Ok(Ok(_)) => error(
            StatusCode::BAD_GATEWAY,
            format!("{name} asked for input or a task; the gateway only takes direct results"),
            "upstream_error",
        ),
        // The server answered with a JSON-RPC error (unknown tool, invalid arguments): it
        // refused the call, so this is a definite failure and the session is healthy.
        Ok(Err(ServiceError::McpError(e))) => Json(CallOutput {
            output: e.message.into_owned(),
            is_error: true,
        })
        .into_response(),
        Ok(Err(e)) => {
            // Only a dead connection earns a restart; never retried here either way.
            if transport_gone(&e, &handle.peer) {
                tools.reset(handle.generation).await;
            }
            error(
                StatusCode::BAD_GATEWAY,
                format!("{name} failed and may or may not have run: {e}"),
                "upstream_error",
            )
        }
        Err(_) => error(
            StatusCode::GATEWAY_TIMEOUT,
            format!(
                "{name} did not answer within {}s; it may still complete",
                CALL_TIMEOUT.as_secs()
            ),
            "timeout",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::{ContentBlock, ToolAnnotations};
    use std::sync::Arc;

    #[test]
    fn command_splits_program_and_arguments() {
        let t = McpTools::from_command("  /usr/local/bin/ldk-server-mcp --verbose ").unwrap();
        assert_eq!(t.program, "/usr/local/bin/ldk-server-mcp");
        assert_eq!(t.args, vec!["--verbose"]);
        assert!(McpTools::from_command("   ").is_none());
    }

    #[tokio::test]
    async fn a_server_that_never_handshakes_times_out_and_frees_the_queue() {
        // An argument no other process on the machine uses, to find ours afterwards.
        let marker = format!("31.{}", std::process::id());
        let mut t = McpTools::from_command(&format!("sleep {marker}")).unwrap();
        t.start_timeout = Duration::from_millis(200);
        let first = tokio::time::timeout(Duration::from_secs(5), t.list()).await;
        let err = first.expect("bounded by the start timeout").unwrap_err();
        assert!(err.contains("did not finish the MCP handshake"), "{err}");
        // The lock was released: the next request gets its own attempt rather than hanging.
        let second = tokio::time::timeout(Duration::from_secs(5), t.list()).await;
        assert!(second.expect("not stuck behind the first").is_err());
        assert!(t.session.lock().await.is_none());
        // Neither timed-out start left its process running.
        tokio::time::sleep(Duration::from_millis(300)).await;
        let left = std::process::Command::new("pgrep")
            .args(["-f", &format!("sleep {marker}")])
            .output()
            .expect("pgrep runs");
        assert!(left.stdout.is_empty(), "left running: {:?}", left.stdout);
    }

    #[test]
    fn in_flight_counts_calls_while_they_live() {
        let t = McpTools::from_command("x").unwrap();
        let a = InFlight::new(&t.calls_in_flight);
        let b = InFlight::new(&t.calls_in_flight);
        assert_eq!(t.calls_in_flight.load(Ordering::SeqCst), 2);
        drop(a);
        drop(b);
        assert_eq!(t.calls_in_flight.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn tool_info_carries_schema_and_hints() {
        let schema: Map<String, Value> = serde_json::from_value(serde_json::json!({
            "type": "object", "properties": { "amount_msat": { "type": "integer" } }
        }))
        .unwrap();
        let mut tool = Tool::new("bolt11_send", "Pay an invoice", Arc::new(schema.clone()));
        tool.annotations = Some(ToolAnnotations::new().read_only(false).destructive(true));
        let info = ToolInfo::from(tool);
        assert_eq!(info.name, "bolt11_send");
        assert_eq!(info.parameters, Value::Object(schema));
        assert_eq!(info.read_only, Some(false));
        assert_eq!(info.destructive, Some(true));

        let bare = ToolInfo::from(Tool::new("get_balances", "Balances", Arc::new(Map::new())));
        let json = serde_json::to_value(&bare).unwrap();
        assert!(json.get("read_only").is_none(), "absent hints stay absent");
    }

    #[test]
    fn call_output_prefers_text_then_structured() {
        let text = CallToolResult::success(vec![ContentBlock::text("a"), ContentBlock::text("b")]);
        assert_eq!(
            CallOutput::from(text),
            CallOutput {
                output: "a\nb".into(),
                is_error: false
            }
        );

        let failed = CallToolResult::error(vec![ContentBlock::text("insufficient funds")]);
        assert!(CallOutput::from(failed).is_error);

        let structured = CallToolResult::structured(serde_json::json!({ "ok": true }));
        let out = CallOutput::from(structured);
        assert!(out.output.contains("\"ok\""), "{}", out.output);
    }
}

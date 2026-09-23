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

use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use rmcp::model::{CallToolRequestParams, CallToolResponse, CallToolResult, Tool};
use rmcp::service::{RoleClient, RunningService, ServiceExt};
use rmcp::transport::TokioChildProcess;
use rmcp::Peer;
use serde::Serialize;
use serde_json::{Map, Value};
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::{error, AppState};

/// How long a tool call may take. A payment can take a while to settle or fail; past this the
/// client is told the outcome is unknown, not that it failed.
const CALL_TIMEOUT: Duration = Duration::from_secs(120);

/// One MCP server, started on first use and restarted after it goes away.
pub struct McpTools {
    program: String,
    args: Vec<String>,
    session: Mutex<Option<RunningService<RoleClient, ()>>>,
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
        })
    }

    pub fn program(&self) -> &str {
        &self.program
    }

    /// A handle to the running server, starting it if it is not running.
    async fn peer(&self) -> Result<Peer<RoleClient>, String> {
        let mut session = self.session.lock().await;
        if let Some(s) = session.as_ref() {
            if !s.is_transport_closed() {
                return Ok(s.peer().clone());
            }
            tracing::warn!("MCP server {} exited; starting it again", self.program);
        }
        let mut command = Command::new(&self.program);
        command.args(&self.args);
        let transport = TokioChildProcess::new(command)
            .map_err(|e| format!("could not start {}: {e}", self.program))?;
        let running = ()
            .serve(transport)
            .await
            .map_err(|e| format!("{} did not start as an MCP server: {e}", self.program))?;
        let peer = running.peer().clone();
        *session = Some(running);
        Ok(peer)
    }

    /// Forget the server so the next request starts a fresh one.
    async fn reset(&self) {
        if let Some(s) = self.session.lock().await.take() {
            let _ = s.cancel().await;
        }
    }

    async fn list(&self) -> Result<Vec<Tool>, String> {
        // Listing has no side effects, so a server that died since the last request gets one
        // retry. Calls never do: a call that broke mid-way may still have run.
        for attempt in 0..2 {
            let peer = self.peer().await?;
            match peer.list_all_tools().await {
                Ok(tools) => return Ok(tools),
                Err(e) if attempt == 0 => {
                    tracing::warn!("listing MCP tools failed, restarting the server: {e}");
                    self.reset().await;
                }
                Err(e) => return Err(format!("listing tools failed: {e}")),
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
    let peer = match tools.peer().await {
        Ok(p) => p,
        Err(e) => return error(StatusCode::BAD_GATEWAY, e, "upstream_error"),
    };
    tracing::info!("tool call: {name}");
    let params = CallToolRequestParams::new(name.clone()).with_arguments(arguments);
    match tokio::time::timeout(CALL_TIMEOUT, peer.call_tool_once(params)).await {
        Ok(Ok(CallToolResponse::Complete(result))) => {
            Json(CallOutput::from(result)).into_response()
        }
        Ok(Ok(_)) => error(
            StatusCode::BAD_GATEWAY,
            format!("{name} asked for input or a task; the gateway only takes direct results"),
            "upstream_error",
        ),
        Ok(Err(e)) => {
            // The server may have gone away; start a fresh one next time. Not retried here.
            tools.reset().await;
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

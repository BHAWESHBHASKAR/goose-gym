//! goose-gym prototype.
//!
//! extract     <session file> <case.json> [max messages]
//!             Convert a goose / Claude Code / Codex / Pi session into a scrubbed case file.
//! record      <case.json>
//!             Run the goose agent loop against the case with a scripted model and replayed
//!             tools, recording every model call with goose's TestProvider.
//! replay      <case.json>
//!             Run the goose agent loop again using only the recorded model calls and the
//!             replayed tools, then compare the resulting conversation with the case.
//! serve-tools <case.json>
//!             Internal: a minimal stdio MCP server that answers tool calls from the case.

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use goose::agents::{
    AgentEvent, ExtensionConfig, GoosePlatform, SessionConfig, StateMachineServices,
    StateMachineServicesConfig,
};
use goose::config::permission::PermissionManager;
use goose::config::GooseMode;
use goose::conversation::message::{ActionRequiredData, Message, MessageContent};
use goose::providers::base::{MessageStream, Provider, ProviderUsage, Usage};
use goose::providers::testprovider::TestProvider;
use goose::session::import_formats::{convert_to_goose_session_json, detect_format};
use goose::session::session_manager::SessionType;
use goose::session::SessionManager;
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use regex::Regex;
use rmcp::model::Tool;
use serde_json::{json, Value};

/// Name of the extension that serves recorded tool results. goose exposes its tools to the
/// model as `replay__<tool>`.
const EXT: &str = "replay";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize| args.get(i).map(String::as_str);
    match (arg(0), arg(1)) {
        (Some("extract"), Some(session)) => {
            let out = arg(2).ok_or_else(|| anyhow!("extract needs an output path"))?;
            let max = arg(3).map(str::parse).transpose()?;
            extract(session, out, max)
        }
        (Some("serve-tools"), Some(case)) => serve_tools(case),
        (Some(mode @ ("record" | "replay")), Some(case)) => {
            let ok = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(run_case(case, mode == "record"))?;
            std::process::exit(if ok { 0 } else { 1 });
        }
        _ => bail!("usage: goose-gym extract|record|replay|serve-tools <file> ..."),
    }
}

// ---------------------------------------------------------------- case file helpers

fn read_case(path: &str) -> Result<Vec<Value>> {
    let case: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    case["messages"]
        .as_array()
        .cloned()
        .ok_or_else(|| anyhow!("{path}: no messages array"))
}

fn blocks(message: &Value) -> impl Iterator<Item = &Value> {
    message["content"].as_array().into_iter().flatten()
}

fn is_tool_response_only(message: &Value) -> bool {
    blocks(message).all(|b| b["type"] == "toolResponse")
}

/// Text of a recorded tool result, and whether it was an error.
fn tool_result_summary(block: &Value) -> (String, bool) {
    let result = &block["toolResult"];
    if result["status"] != "success" {
        return (result["error"].as_str().unwrap_or("error").to_string(), true);
    }
    let text = result["value"]["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| c["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    (text, result["value"]["isError"].as_bool().unwrap_or(false))
}

/// Comparable form of a conversation: one entry per run of same-role messages.
fn comparable(messages: &[Value]) -> Vec<(String, Vec<String>)> {
    let mut out: Vec<(String, Vec<String>)> = Vec::new();
    for message in messages {
        if message["metadata"]["turnContext"] == true {
            continue;
        }
        let role = message["role"].as_str().unwrap_or("?").to_string();
        let mut items: Vec<String> = Vec::new();
        for block in blocks(message) {
            match block["type"].as_str() {
                Some("text") => {
                    let text = block["text"].as_str().unwrap_or("");
                    if text.trim().is_empty() {
                        continue;
                    }
                    // Adjacent text blocks count as one text.
                    match items.last_mut() {
                        Some(last) if last.starts_with("text ") => {
                            last.push('\n');
                            last.push_str(text);
                        }
                        _ => items.push(format!("text {text}")),
                    }
                }
                Some("error") => items.push(format!("error {}", block["message"])),
                Some("toolRequest") => items.push(format!(
                    "call {} {}",
                    block["toolCall"]["value"]["name"].as_str().unwrap_or("?"),
                    block["toolCall"]["value"]["arguments"]
                )),
                Some("toolResponse") => {
                    let (text, is_error) = tool_result_summary(block);
                    items.push(format!("result {} error={is_error} {text}", block["id"]));
                }
                _ => {}
            }
        }
        // Completion order of concurrent tool calls is not part of the comparison.
        if items.iter().all(|item| item.starts_with("result ")) {
            items.sort();
        }
        if items.is_empty() {
            continue;
        }
        match out.last_mut() {
            Some((last_role, last_items)) if *last_role == role => last_items.extend(items),
            _ => out.push((role, items)),
        }
    }
    out
}

// ---------------------------------------------------------------- extract

struct Scrubber {
    rules: Vec<(Regex, &'static str)>,
    hits: usize,
}

impl Scrubber {
    fn new() -> Self {
        let user = regex::escape(&std::env::var("USERNAME").unwrap_or_else(|_| "user".into()));
        let rules = [
            (r"(?:sk|pk|rk)-[A-Za-z0-9_\-]{20,}".to_string(), "SECRET"),
            (r"gh[pousr]_[A-Za-z0-9]{20,}".to_string(), "SECRET"),
            (r"github_pat_[A-Za-z0-9_]{20,}".to_string(), "SECRET"),
            (r"AKIA[0-9A-Z]{16}".to_string(), "SECRET"),
            (r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}".to_string(), "SECRET"),
            (r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+\.[A-Za-z0-9.\-]+".to_string(), "EMAIL"),
            (format!(r"(?i)\b{user}\b"), "USER"),
        ];
        Self {
            rules: rules
                .into_iter()
                .map(|(pattern, with)| (Regex::new(&pattern).unwrap(), with))
                .collect(),
            hits: 0,
        }
    }

    fn scrub(&mut self, value: &mut Value) {
        match value {
            Value::String(text) => {
                for (pattern, with) in &self.rules {
                    self.hits += pattern.find_iter(text).count();
                    *text = pattern.replace_all(text, *with).into_owned();
                }
            }
            Value::Array(items) => items.iter_mut().for_each(|item| self.scrub(item)),
            Value::Object(map) => map.values_mut().for_each(|item| self.scrub(item)),
            _ => {}
        }
    }
}

fn extract(session_path: &str, out_path: &str, max_messages: Option<usize>) -> Result<()> {
    let raw = std::fs::read_to_string(session_path)?;
    let format = detect_format(&raw);
    let session: Value = serde_json::from_str(&convert_to_goose_session_json(&raw)?)?;
    let source = session["conversation"]
        .as_array()
        .ok_or_else(|| anyhow!("converted session has no conversation"))?;

    // Keep what a replay can reproduce: text, tool calls, tool results. Merge runs of
    // same-role messages, because the agent loop produces one message per model call.
    let mut messages: Vec<Value> = Vec::new();
    for message in source {
        let mut message = message.clone();
        let kept: Vec<Value> = blocks(&message)
            .filter(|b| matches!(b["type"].as_str(), Some("text" | "toolRequest" | "toolResponse")))
            .filter(|b| b["type"] != "text" || !b["text"].as_str().unwrap_or("").trim().is_empty())
            .cloned()
            .collect();
        if kept.is_empty() {
            continue;
        }
        let responses_only = kept.iter().all(|b| b["type"] == "toolResponse");
        match messages.last_mut() {
            // Tool results and typed user text stay in separate messages.
            Some(last)
                if last["role"] == message["role"]
                    && (message["role"] == "assistant"
                        || is_tool_response_only(last) == responses_only) =>
            {
                last["content"].as_array_mut().unwrap().extend(kept);
            }
            _ => {
                message["content"] = Value::Array(kept);
                messages.push(message);
            }
        }
    }
    if let Some(max) = max_messages {
        messages.truncate(max);
    }

    // A case must start with a user request and end with an assistant message that has
    // no unanswered tool call.
    while messages.first().is_some_and(|m| m["role"] != "user" || is_tool_response_only(m)) {
        messages.remove(0);
    }
    // An interrupted turn (the user typed while tool calls were open or just answered)
    // cannot be replayed by a scripted model, so the case stops there.
    let has_calls = |m: &Value| blocks(m).any(|b| b["type"] == "toolRequest");
    let interrupted = (1..messages.len()).find(|&i| {
        let (before, now) = (&messages[i - 1], &messages[i]);
        now["role"] == "user"
            && !is_tool_response_only(now)
            && (before["role"] == "user" || has_calls(before))
    });
    if let Some(index) = interrupted {
        println!("note: turn interrupted at message {index}, case cut there");
        messages.truncate(index);
    }
    loop {
        match messages.last_mut() {
            Some(last) if last["role"] == "user" => {
                messages.pop();
            }
            Some(last) if has_calls(last) => {
                // Calls whose results were cut off: keep the text, drop the calls.
                last["content"].as_array_mut().unwrap().retain(|b| b["type"] != "toolRequest");
                if blocks(last).next().is_none() {
                    messages.pop();
                }
            }
            _ => break,
        }
    }
    if messages.is_empty() {
        bail!("nothing replayable in this session");
    }

    // Tools are served by one replay extension, so the model-facing name gets its prefix.
    let mut tool_calls = 0;
    for message in &mut messages {
        for block in message["content"].as_array_mut().unwrap() {
            if block["type"] == "toolRequest" {
                tool_calls += 1;
                block.as_object_mut().unwrap().remove("_meta");
                if let Some(name) = block["toolCall"]["value"]["name"].as_str() {
                    let plain = name.replace("__", "_");
                    block["toolCall"]["value"]["name"] = json!(format!("{EXT}__{plain}"));
                }
            }
        }
    }

    let mut scrubber = Scrubber::new();
    messages.iter_mut().for_each(|m| scrubber.scrub(m));

    let case = json!({
        "version": 1,
        "source_format": format!("{format:?}"),
        "messages": messages,
    });
    std::fs::write(out_path, serde_json::to_string_pretty(&case)?)?;
    println!(
        "extracted {} messages, {} tool calls, {} scrubbed strings ({format:?} session) -> {out_path}",
        case["messages"].as_array().unwrap().len(),
        tool_calls,
        scrubber.hits
    );
    Ok(())
}

// ---------------------------------------------------------------- serve-tools

fn serve_tools(case_path: &str) -> Result<()> {
    let messages = read_case(case_path)?;
    let mut calls: HashMap<&str, (String, Value)> = HashMap::new();
    let mut names: Vec<String> = Vec::new();
    let mut queues: HashMap<String, VecDeque<(Value, Value)>> = HashMap::new();
    for block in messages.iter().flat_map(blocks) {
        let id = block["id"].as_str().unwrap_or("");
        if block["type"] == "toolRequest" {
            let call = &block["toolCall"]["value"];
            let full = call["name"].as_str().unwrap_or("");
            let name = full.strip_prefix(&format!("{EXT}__")).unwrap_or(full).to_string();
            if !names.contains(&name) {
                names.push(name.clone());
            }
            calls.insert(id, (name, call["arguments"].clone()));
        } else if block["type"] == "toolResponse" {
            if let Some((name, arguments)) = calls.get(id) {
                let (text, is_error) = tool_result_summary(block);
                let result = if block["toolResult"]["status"] == "success" {
                    json!({"content": block["toolResult"]["value"]["content"], "isError": is_error})
                } else {
                    json!({"content": [{"type": "text", "text": text}], "isError": true})
                };
                queues.entry(name.clone()).or_default().push_back((arguments.clone(), result));
            }
        }
    }

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(request) = serde_json::from_str::<Value>(&line?) else { continue };
        let Some(id) = request.get("id").cloned() else { continue };
        let params = &request["params"];
        let result = match request["method"].as_str().unwrap_or("") {
            "initialize" => json!({
                "protocolVersion": params["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "goose-gym-replay", "version": "0.0.1"},
            }),
            "ping" => json!({}),
            "tools/list" => json!({"tools": names.iter().map(|name| json!({
                "name": name,
                "description": "Replayed tool from a recorded session",
                "inputSchema": {"type": "object", "additionalProperties": true},
            })).collect::<Vec<_>>()}),
            "tools/call" => {
                let name = params["name"].as_str().unwrap_or("");
                let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
                // goose runs the tool calls of one model message concurrently, so calls are
                // matched by name and arguments, not by arrival order. Repeated identical
                // calls get their recorded results in the recorded order.
                let queue = queues.entry(name.to_string()).or_default();
                match queue.iter().position(|(recorded, _)| *recorded == arguments) {
                    Some(index) => queue.remove(index).unwrap().1,
                    None => json!({"isError": true, "content": [{"type": "text",
                        "text": format!("replay has no recorded result for {name} with {arguments}")}]}),
                }
            }
            other => {
                let error = json!({"jsonrpc": "2.0", "id": id,
                    "error": {"code": -32601, "message": format!("method not found: {other}")}});
                writeln!(stdout, "{error}")?;
                stdout.flush()?;
                continue;
            }
        };
        writeln!(stdout, "{}", json!({"jsonrpc": "2.0", "id": id, "result": result}))?;
        stdout.flush()?;
    }
    Ok(())
}

// ---------------------------------------------------------------- providers

/// Plays the assistant messages of the case back in order. Used only while recording.
struct ScriptedProvider {
    script: Mutex<VecDeque<Message>>,
}

#[async_trait]
impl Provider for ScriptedProvider {
    fn get_name(&self) -> &str {
        "goose-gym-scripted"
    }

    async fn stream(
        &self,
        _model_config: &ModelConfig,
        _system: &str,
        _messages: &[Message],
        _tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let message = self.script.lock().unwrap().pop_front().ok_or_else(|| {
            ProviderError::ExecutionError("scripted provider ran out of messages".into())
        })?;
        let usage = ProviderUsage::new("scripted".to_string(), Usage::default());
        Ok(goose::providers::base::stream_from_single_message(message, usage))
    }
}

/// Sits between the agent and TestProvider. TestProvider matches a recorded call by a hash
/// of the messages, and goose adds a per-turn context message carrying the current time and
/// working directory. That message is dropped here so a recording made at one time and
/// place still matches later.
struct StableInput {
    inner: Arc<TestProvider>,
    calls: Mutex<usize>,
    dropped: Mutex<usize>,
}

#[async_trait]
impl Provider for StableInput {
    fn get_name(&self) -> &str {
        "goose-gym"
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        // By the time a request reaches the provider, goose has folded the turn context
        // into the user message as an extra text block, so it is removed block by block.
        let mut removed = 0;
        let stable: Vec<Message> = messages
            .iter()
            .filter(|m| !m.is_turn_context())
            .cloned()
            .filter_map(|mut m| {
                let before = m.content.len();
                m.content.retain(|block| match block {
                    MessageContent::Text(text) => !text.text.trim_start().starts_with("<turn-context>"),
                    _ => true,
                });
                removed += before - m.content.len();
                // Tool results are stored in completion order, which can change between
                // runs when calls run concurrently. Put them in a fixed order.
                m.content.sort_by_key(|block| match block {
                    MessageContent::ToolResponse(response) => response.id.clone(),
                    _ => String::new(),
                });
                (!m.content.is_empty()).then_some(m)
            })
            .collect();
        *self.calls.lock().unwrap() += 1;
        *self.dropped.lock().unwrap() += removed;
        self.inner.stream(model_config, system, &stable, tools).await
    }
}

// ---------------------------------------------------------------- record / replay

async fn run_case(case_path: &str, record: bool) -> Result<bool> {
    let case_path = std::fs::canonicalize(case_path)?
        .to_string_lossy()
        .trim_start_matches(r"\\?\")
        .to_string();
    let records_path = format!("{case_path}.records.json");
    let case = read_case(&case_path)?;

    // Keep goose's config, data and sessions for this run out of the user's real goose folders.
    let root = tempfile::tempdir()?;
    std::env::set_var("GOOSE_PATH_ROOT", root.path());

    let session_manager = Arc::new(SessionManager::new(root.path().join("sessions")));
    let services = StateMachineServices::with_config(StateMachineServicesConfig::new(
        session_manager.clone(),
        PermissionManager::instance(),
        None,
        true,
        GoosePlatform::GooseCli,
    ));
    let session = session_manager
        .create_session(
            root.path().to_path_buf(),
            "goose-gym".to_string(),
            SessionType::Hidden,
            GooseMode::Auto,
        )
        .await?;

    let test_provider = Arc::new(if record {
        let script = case
            .iter()
            .filter(|m| m["role"] == "assistant")
            .map(|m| serde_json::from_value::<Message>(m.clone()))
            .collect::<Result<VecDeque<_>, _>>()
            .context("case assistant message does not parse as a goose Message")?;
        let scripted = Arc::new(ScriptedProvider { script: Mutex::new(script) });
        TestProvider::new_recording(scripted, records_path.clone())
    } else {
        if !PathBuf::from(&records_path).exists() {
            bail!("{records_path} not found, run record first");
        }
        TestProvider::new_replaying(records_path.clone())?
    });
    let provider = Arc::new(StableInput {
        inner: test_provider.clone(),
        calls: Mutex::new(0),
        dropped: Mutex::new(0),
    });
    services
        .update_provider(provider.clone(), ModelConfig::new("goose-gym-model"), &session.id)
        .await?;

    services
        .extension_manager
        .enable(
            &session.id,
            ExtensionConfig::Stdio {
                name: EXT.to_string(),
                description: "Replays recorded tool results".to_string(),
                cmd: std::env::current_exe()?.to_string_lossy().to_string(),
                args: vec!["serve-tools".to_string(), case_path.clone()],
                envs: Default::default(),
                env_keys: vec![],
                timeout: Some(30),
                cwd: None,
                bundled: Some(false),
                available_tools: vec![],
            },
        )
        .await
        .context("could not start the replay tool server")?;

    // Each user request in the case becomes one reply() call. Tool results are not sent:
    // the agent has to produce them itself by calling the replay extension.
    let mut failure: Option<String> = None;
    let mut user_turns = 0;
    for message in case.iter().filter(|m| m["role"] == "user" && !is_tool_response_only(m)) {
        let text = blocks(message)
            .filter_map(|b| b["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n");
        user_turns += 1;
        let config = SessionConfig { id: session.id.clone(), schedule_id: None, max_turns: Some(1000) };
        let stream = services.reply(Message::user().with_text(text), config, None).await?;
        tokio::pin!(stream);
        while let Some(event) = stream.next().await {
            match event {
                Ok(AgentEvent::Message(reply)) => {
                    for block in &reply.content {
                        if let MessageContent::ActionRequired(action) = block {
                            if let ActionRequiredData::ToolConfirmation { id, tool_name, .. } = &action.data {
                                println!("note: goose asked to confirm {tool_name}, allowed once");
                                services
                                    .submit_tool_confirmation(&session.id, id, goose::permission::Permission::AllowOnce)
                                    .await?;
                            }
                        }
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    failure = Some(format!("agent error in user turn {user_turns}: {error}"));
                    break;
                }
            }
        }
        if failure.is_some() {
            break;
        }
    }

    if record {
        test_provider.save_records()?;
    }

    let produced = session_manager
        .get_session(&session.id, true)
        .await?
        .conversation
        .map(|c| c.messages().to_vec())
        .unwrap_or_default();
    let produced: Vec<Value> = produced.iter().map(serde_json::to_value).collect::<Result<_, _>>()?;
    std::fs::write(
        format!("{case_path}.{}.json", if record { "recorded" } else { "replayed" }),
        serde_json::to_string_pretty(&produced)?,
    )?;

    let want = comparable(&case);
    let got = comparable(&produced);
    let mut same = failure.is_none() && want.len() == got.len();
    println!(
        "{}: {user_turns} user turns, {} model calls ({} recorded), {} turn-context messages dropped before matching",
        if record { "record" } else { "replay" },
        provider.calls.lock().unwrap(),
        test_provider.get_record_count(),
        provider.dropped.lock().unwrap(),
    );
    if let Some(failure) = &failure {
        println!("FAILED: {failure}");
    }
    for index in 0..want.len().max(got.len()) {
        let (w, g) = (want.get(index), got.get(index));
        if w == g {
            continue;
        }
        same = false;
        println!("difference at message group {index}:");
        println!("  case: {}", preview(w));
        println!("  goose: {}", preview(g));
        break;
    }
    println!(
        "{}: case has {} message groups, goose produced {}",
        if same { "MATCH" } else { "MISMATCH" },
        want.len(),
        got.len()
    );
    Ok(same)
}

fn preview(group: Option<&(String, Vec<String>)>) -> String {
    match group {
        None => "(nothing)".to_string(),
        Some((role, items)) => {
            let text: String = format!("{role}: {}", items.join(" | ")).chars().take(400).collect();
            text.replace('\n', " ")
        }
    }
}

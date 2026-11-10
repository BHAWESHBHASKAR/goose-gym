# goose-gym (prototype)

Turns a real agent session into a regression case and replays it through goose's own agent loop with no model and no real tools.

It is a single Rust binary that uses goose as a library (`aaif-goose/goose` at 72b8db8, v1.59.0). Nothing in the goose repo is changed.

## What it does

| Command | What happens |
|---|---|
| `goose-gym extract <session> <case.json> [max messages]` | Reads a goose, Claude Code, Codex or Pi session with goose's own importers, keeps text, tool calls and tool results, scrubs secrets, emails and the user name, writes a case file. |
| `goose-gym record <case.json>` | Runs goose's agent loop on the case. The model is a script of the case's assistant messages. Tools are answered by a small MCP server that serves the recorded results. Every model call is recorded with goose's `TestProvider`. |
| `goose-gym replay <case.json>` | Runs the agent loop again with only the recording and the tool server, then compares the conversation goose produced with the case. Exit code 1 on any difference. |

## Result on 2026-11-10 (Windows 11)

- 13 real sessions (1 Codex, 12 Claude Code) extracted, recorded and replayed with an exact match. The largest: 168 messages, 83 tool calls, 84 model calls.
- The Codex case was replayed 5 times in a row with the same result.
- Changing one recorded tool result in a case makes replay fail at the next model call, so a difference in what the agent sends to the model is caught.

## What had to be handled to make replay exact

1. **Turn context.** goose adds the current time and working directory to the user message on every model call. `TestProvider` finds a recorded answer by a hash of the messages, so a recording never matched on a later run. The prototype removes that text block before the hash is taken.
2. **Concurrent tool calls.** goose runs the tool calls of one model message at the same time and stores results in completion order. The tool server matches calls by name and arguments, and results are put in a fixed order before the hash and before the comparison.
3. **Interrupted turns.** When the user typed while tool calls were open, a scripted model cannot reproduce the turn. The case is cut at that point.

## Limits

- Exact replay checks the harness (agent loop, tool dispatch, message storage). It does not measure whether goose would solve the task better. That needs a live re-run with a real model.
- `TestProvider` does not hash the system prompt or the tool list, so a change to those is not detected.
- Tools are renamed to `replay__<tool>` because goose prefixes tool names with the extension name.
- Thinking blocks and images are dropped from cases.
- The scrubber is a short list of patterns. Case files must be treated as private.
- All cases so far came from Claude Code and Codex sessions through goose's importers. None came from a goose session database yet.
- Tested on Windows only.

## Privacy

Case files are made from your own private sessions and stay on your machine. The `cases` folder is in `.gitignore`. Do not share a case file without reading it first.

## License

Apache-2.0, the same as goose.

## Build and run

Needs Rust 1.99.0 (pinned in `rust-toolchain.toml`). goose is fetched from GitHub at the commit pinned in `Cargo.toml`. The first build compiles goose and takes several minutes.

```
cargo build
goose-gym extract <session.jsonl> cases\one.json
goose-gym record cases\one.json
goose-gym replay cases\one.json
```

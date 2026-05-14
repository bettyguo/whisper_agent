# whisper-agent

A voice-driven local agent loop: STT (Whisper) → local LLM with tool
use → TTS → spoken reply. Runs entirely on your machine.

Status: pre-alpha. The orchestrator, audio I/O, STT/TTS wrappers, MCP
client, wake-word state machine, and CLI surface all have code +
tests, but the live mic → Ollama → speaker integration is not yet
wired end-to-end.

## Install

```
git clone https://github.com/<you>/whisper-agent
cd whisper-agent
pip install -e ".[all]"
```

Python 3.11+. On macOS, `brew install portaudio`; on Debian/Ubuntu,
`apt install portaudio19-dev`.

Pull a default LLM (Ollama):

```
ollama pull qwen2.5:7b-instruct
```

## Usage

```
whisper-agent doctor --verify-offline   # sanity check; exits non-zero if cloud is reachable
whisper-agent talk                      # push-to-talk
whisper-agent listen --wake "hey computer"
whisper-agent listen --continuous
```

The `--ascii` flag on any subcommand drops Rich formatting for
screen-reader compatibility.

## Modes

- **push-to-talk** (default): hold a key, speak, release. Lowest
  friction; no false triggers.
- **wake-word**: openWakeWord listens for a phrase and gates the rest
  of the pipeline.
- **continuous**: mic always on, VAD-gated. Highest power use; useful
  when you can't reliably press a key.

## Privacy

Default mode makes no outbound network calls. `whisper-agent doctor
--verify-offline` exits non-zero if any cloud backend is configured.
Cloud STT/TTS is BYOK opt-in, gated behind `--enable-cloud`, and shows
a persistent banner in the TUI when active.

MCP servers you install yourself are out of scope: they can do
whatever they're configured to do. `whisper-agent doctor` lists them
so you can audit.

## Tool surface

Built-ins: `fs.read`, `fs.write`, `fs.search`, `notes.append`. All
four refuse paths outside a configured workspace root (default:
process cwd at startup; pass `workspace_root=` or `sandbox=False` to
`register_builtin_tools`).

Anything you've configured in `~/.config/whisper-agent/mcp.toml`
becomes a tool the LLM can call as `mcp.<server>.<tool>`.

## Configuration

`~/.config/whisper-agent/config.toml`:

```toml
[stt]
backend = "faster-whisper"
model = "large-v3"
compute_type = "auto"

[tts]
backend = "kokoro"
voice = "default"
speed = 1.0

[llm]
backend = "ollama"
model = "qwen2.5:7b-instruct"
base_url = "http://localhost:11434"
```

## License

Apache 2.0.

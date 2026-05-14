"""parse_ollama_chunk: chunk parsing without an HTTP server."""

from __future__ import annotations

from whisper_agent.llm.ollama import parse_ollama_chunk
from whisper_agent.llm.tool_use import LLMEvent


def test_empty_chunk_yields_nothing() -> None:
    assert parse_ollama_chunk({}) == []


def test_text_only_chunk() -> None:
    out = parse_ollama_chunk({"message": {"content": "hello "}})
    assert out == [LLMEvent(text="hello ")]


def test_chunk_with_done_flag_emits_done() -> None:
    out = parse_ollama_chunk({"message": {"content": ""}, "done": True})
    assert any(e.done for e in out)


def test_text_plus_done() -> None:
    out = parse_ollama_chunk({"message": {"content": "world"}, "done": True})
    assert out[0].text == "world"
    assert out[-1].done is True


def test_tool_call_with_dict_arguments() -> None:
    chunk = {
        "message": {
            "tool_calls": [{"function": {"name": "fs.read", "arguments": {"path": "README.md"}}}],
        }
    }
    out = parse_ollama_chunk(chunk)
    assert len(out) == 1
    assert out[0].tool_call is not None
    assert out[0].tool_call.name == "fs.read"
    assert out[0].tool_call.arguments == {"path": "README.md"}


def test_tool_call_with_string_arguments_parses_json() -> None:
    chunk = {
        "message": {
            "tool_calls": [{"function": {"name": "fs.read", "arguments": '{"path": "x"}'}}],
        }
    }
    out = parse_ollama_chunk(chunk)
    assert out[0].tool_call.arguments == {"path": "x"}


def test_tool_call_with_invalid_string_arguments_is_dropped() -> None:
    chunk = {
        "message": {
            "tool_calls": [{"function": {"name": "fs.read", "arguments": "{not json}"}}],
        }
    }
    out = parse_ollama_chunk(chunk)
    assert out == []


def test_tool_call_with_missing_name_is_dropped() -> None:
    chunk = {
        "message": {"tool_calls": [{"function": {"arguments": {}}}]},
    }
    out = parse_ollama_chunk(chunk)
    assert out == []


def test_multiple_tool_calls_in_one_chunk() -> None:
    chunk = {
        "message": {
            "tool_calls": [
                {"function": {"name": "fs.read", "arguments": {"path": "a"}}},
                {"function": {"name": "fs.read", "arguments": {"path": "b"}}},
            ],
        },
        "done": True,
    }
    out = parse_ollama_chunk(chunk)
    tool_events = [e for e in out if e.tool_call is not None]
    assert len(tool_events) == 2
    assert tool_events[0].tool_call.arguments == {"path": "a"}
    assert tool_events[1].tool_call.arguments == {"path": "b"}

"""Tool-use core: ToolRegistry, ToolSpec, dispatch."""

from __future__ import annotations

import asyncio

import pytest

from whisper_agent.llm.tool_use import (
    LLMEvent,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolSpec,
)


def _spec(name: str, **kw) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="test tool",
        parameters={"type": "object", "properties": {}},
        **kw,
    )


def test_register_and_lookup() -> None:
    r = ToolRegistry()
    r.register(Tool(spec=_spec("noop"), handler=lambda: "ok"))
    assert r.has("noop")
    assert r.names() == ["noop"]
    assert [s.name for s in r.specs()] == ["noop"]


def test_double_register_raises() -> None:
    r = ToolRegistry()
    t = Tool(spec=_spec("x"), handler=lambda: 1)
    r.register(t)
    with pytest.raises(ValueError):
        r.register(t)


def test_call_unknown_tool_yields_error() -> None:
    r = ToolRegistry()
    call = ToolCall(name="missing", arguments={})
    result = asyncio.run(r.call(call))
    assert result.ok is False
    assert "unknown tool" in (result.error or "")


def test_call_sync_handler() -> None:
    r = ToolRegistry()
    r.register(
        Tool(
            spec=_spec("add"),
            handler=lambda a, b: a + b,
        )
    )
    result = asyncio.run(r.call(ToolCall(name="add", arguments={"a": 2, "b": 3})))
    assert result.ok is True
    assert result.result == 5


def test_call_async_handler() -> None:
    r = ToolRegistry()

    async def handler(x: int) -> int:
        await asyncio.sleep(0)
        return x * 2

    r.register(Tool(spec=_spec("double"), handler=handler))
    result = asyncio.run(r.call(ToolCall(name="double", arguments={"x": 5})))
    assert result.ok is True
    assert result.result == 10


def test_call_handler_exception_returns_error() -> None:
    r = ToolRegistry()

    def boom() -> None:
        raise RuntimeError("nope")

    r.register(Tool(spec=_spec("boom"), handler=boom))
    result = asyncio.run(r.call(ToolCall(name="boom", arguments={})))
    assert result.ok is False
    assert "RuntimeError" in (result.error or "")


def test_call_wrong_arg_types_returns_error() -> None:
    r = ToolRegistry()
    r.register(Tool(spec=_spec("expect_x"), handler=lambda x: x))
    result = asyncio.run(r.call(ToolCall(name="expect_x", arguments={"y": 1})))
    assert result.ok is False
    assert "invalid arguments" in (result.error or "")


def test_tool_result_message_content_ok() -> None:
    import json

    from whisper_agent.llm.tool_use import ToolResult

    r = ToolResult(tool_call_id="abc", ok=True, result={"hello": "world"})
    payload = json.loads(r.to_message_content())
    assert payload == {"ok": True, "result": {"hello": "world"}}


def test_tool_result_message_content_error() -> None:
    import json

    from whisper_agent.llm.tool_use import ToolResult

    r = ToolResult(tool_call_id="abc", ok=False, error="boom")
    payload = json.loads(r.to_message_content())
    assert payload == {"ok": False, "error": "boom"}


def test_tool_spec_to_json_schema() -> None:
    s = ToolSpec(
        name="fs.read",
        description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    schema = s.to_json_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "fs.read"
    assert "path" in schema["function"]["parameters"]["properties"]


def test_llm_event_default_done_false() -> None:
    e = LLMEvent(text="hi")
    assert e.text == "hi"
    assert e.tool_call is None
    assert e.done is False


def test_tool_use_protocol_is_async_iterator_shape() -> None:
    """ToolUseBackend.stream must be ``def -> AsyncIterator``, not
    ``async def``. Async generators already return an async iterator
    on call; the Protocol method has to match that callable shape.
    """
    import inspect

    from whisper_agent.llm.tool_use import ToolUseBackend

    fn = ToolUseBackend.stream
    assert not inspect.iscoroutinefunction(fn), (
        "ToolUseBackend.stream is async def; should be plain `def -> AsyncIterator[LLMEvent]`"
    )


def test_tts_protocol_is_async_iterator_shape() -> None:
    """Companion of the above for TTSSynthesizer.synthesize."""
    import inspect

    from whisper_agent.tts.base import TTSSynthesizer

    fn = TTSSynthesizer.synthesize
    assert not inspect.iscoroutinefunction(fn), (
        "TTSSynthesizer.synthesize is async def; should be plain `def -> AsyncIterator[TTSChunk]`"
    )

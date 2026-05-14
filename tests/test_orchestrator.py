"""Orchestrator drives prose + tool-call turns through MockBackend."""

from __future__ import annotations

import asyncio

import pytest

from whisper_agent.llm.tool_use import LLMEvent, Tool, ToolCall, ToolRegistry, ToolSpec
from whisper_agent.orchestrator import AgentState, MockBackend, Orchestrator


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="test",
        parameters={"type": "object", "properties": {}},
    )


def test_prose_only_turn() -> None:
    backend = MockBackend(
        scripts=[
            [
                LLMEvent(text="Hello."),
                LLMEvent(text=" How can I help?"),
                LLMEvent(done=True),
            ]
        ]
    )
    orch = Orchestrator(backend=backend, tools=ToolRegistry())
    trace = asyncio.run(orch.run_turn("hi"))
    assert trace.error is None
    assert trace.reply_text.startswith("Hello.")
    assert AgentState.SPEAKING in trace.states
    assert orch.state is AgentState.IDLE


def test_single_tool_round_then_prose() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            spec=_spec("fs.read"),
            handler=lambda path: {"path": path, "content": "PRETEND"},
        )
    )
    backend = MockBackend(
        scripts=[
            [
                LLMEvent(tool_call=ToolCall(name="fs.read", arguments={"path": "x"}, id="t1")),
                LLMEvent(done=True),
            ],
            [
                LLMEvent(text="Got it."),
                LLMEvent(done=True),
            ],
        ]
    )
    orch = Orchestrator(backend=backend, tools=registry)
    trace = asyncio.run(orch.run_turn("read x"))
    assert trace.error is None
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].name == "fs.read"
    assert trace.reply_text == "Got it."
    # History: system, user, assistant(tool_calls), tool, assistant(reply)
    history = orch.history
    assert history[0]["role"] == "system"
    assert history[1]["role"] == "user"
    assert history[2]["role"] == "assistant"
    assert "tool_calls" in history[2]
    assert history[3]["role"] == "tool"
    assert history[4]["role"] == "assistant"
    assert history[4]["content"] == "Got it."


def test_unknown_tool_returns_error_not_crash() -> None:
    backend = MockBackend(
        scripts=[
            [
                LLMEvent(tool_call=ToolCall(name="ghost.tool", arguments={}, id="t1")),
                LLMEvent(done=True),
            ],
            [
                LLMEvent(text="Sorry, couldn't do that."),
                LLMEvent(done=True),
            ],
        ]
    )
    orch = Orchestrator(backend=backend, tools=ToolRegistry())
    trace = asyncio.run(orch.run_turn("do magic"))
    assert trace.error is None
    assert trace.reply_text == "Sorry, couldn't do that."
    # The tool turn was followed by an error tool-result and another LLM call.
    tool_msgs = [m for m in orch.history if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "unknown tool" in tool_msgs[0]["content"]


def test_tool_round_budget_hits_error() -> None:
    registry = ToolRegistry()
    registry.register(Tool(spec=_spec("loop"), handler=lambda: "ok"))

    # Three rounds in a row that ALL emit tool calls -> exceeds limit of 2.
    def round_with_tool() -> list[LLMEvent]:
        return [
            LLMEvent(tool_call=ToolCall(name="loop", arguments={}, id="x")),
            LLMEvent(done=True),
        ]

    backend = MockBackend(scripts=[round_with_tool() for _ in range(5)])
    orch = Orchestrator(backend=backend, tools=registry)
    trace = asyncio.run(orch.run_turn("loop forever", max_tool_rounds=2))
    assert trace.error is not None
    assert "round limit" in trace.error
    assert orch.state is AgentState.ERROR


def test_speak_hook_receives_chunks() -> None:
    received: list[str] = []

    class FakeTTS:
        sample_rate = 24_000

        async def synthesize(self, text):
            import numpy as np

            from whisper_agent.tts.base import TTSChunk

            received.append(text)
            yield TTSChunk(
                samples=np.zeros(1, dtype=np.float32), sample_rate=self.sample_rate, is_last=True
            )

    spoken = []

    async def speak(chunk):
        spoken.append(chunk)

    backend = MockBackend(
        scripts=[
            [
                LLMEvent(text="One sentence. Two sentences. Three."),
                LLMEvent(done=True),
            ]
        ]
    )
    orch = Orchestrator(
        backend=backend,
        tools=ToolRegistry(),
        tts=FakeTTS(),
        speak=speak,
    )
    asyncio.run(orch.run_turn("speak"))
    # SentenceChunker splits into 3 chunks; TTS is invoked for each.
    assert received == ["One sentence.", "Two sentences.", "Three."]
    assert len(spoken) == 3


@pytest.mark.parametrize("state", list(AgentState))
def test_all_agent_states_are_distinct(state: AgentState) -> None:
    """Catches accidental enum-value collisions."""
    same_value = [s for s in AgentState if s.value == state.value]
    assert same_value == [state]


def test_interleaved_prose_dropped_from_history() -> None:
    """If a single LLM round emits prose AND a tool call, the prose
    must not be concatenated into the next round's history entry.

    The prose still streams through TTS (we can't unspeak it), but it
    does not enter the conversation history.
    """
    registry = ToolRegistry()
    registry.register(Tool(spec=_spec("the_tool"), handler=lambda: "result"))
    backend = MockBackend(
        scripts=[
            # Round 1: prose + tool call (the interleaved case).
            [
                LLMEvent(text="Let me check that "),
                LLMEvent(text="for you."),
                LLMEvent(tool_call=ToolCall(name="the_tool", arguments={}, id="t1")),
                LLMEvent(done=True),
            ],
            # Round 2: prose-only finish.
            [
                LLMEvent(text="It's hello."),
                LLMEvent(done=True),
            ],
        ]
    )
    orch = Orchestrator(backend=backend, tools=registry)
    trace = asyncio.run(orch.run_turn("check"))

    assert trace.error is None
    assert len(trace.tool_calls) == 1

    # The final assistant prose entry in history is round 2 only. The
    # round-1 "Let me check that for you." narration must not leak in.
    history = orch.history
    assistant_prose = [m for m in history if m["role"] == "assistant" and "content" in m]
    assert len(assistant_prose) == 1
    assert assistant_prose[0]["content"] == "It's hello."

    # reply_text reflects only what's in history-bound prose, not the
    # dropped round-1 narration.
    assert trace.reply_text == "It's hello."

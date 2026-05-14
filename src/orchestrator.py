"""Voice -> STT -> LLM (+ tools) -> TTS pipeline.

A small state machine plus the ``run_turn`` driver that turns a
transcribed utterance into a spoken reply, possibly with tool calls
along the way. Testable without audio I/O: pass a mock backend, a mock
TTS, and a list of pre-decided messages.

Interrupt path: the orchestrator owns an ``interrupt`` event that
callers (in production the VAD ``speech_start`` hook) can set to abort
a turn mid-reply. The driver polls between LLM events and after every
chunk; when set, it short-circuits to IDLE, marks the trace
``interrupted``, and returns.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from whisper_agent.llm.tool_use import (
    LLMEvent,
    ToolCall,
    ToolRegistry,
    ToolSpec,
    ToolUseBackend,
)
from whisper_agent.tts.streaming import SentenceChunker

if TYPE_CHECKING:
    from whisper_agent.tts.base import TTSChunk, TTSSynthesizer

log = logging.getLogger(__name__)


class AgentState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class TurnTrace:
    """Observable per-turn record used by tests + the TUI."""

    text_chunks: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tts_chunks: list[TTSChunk] = field(default_factory=list)
    states: list[AgentState] = field(default_factory=list)
    error: str | None = None
    interrupted: bool = False

    @property
    def reply_text(self) -> str:
        return " ".join(self.text_chunks).strip()


SPEAK_HOOK = Callable[["TTSChunk"], Awaitable[None]]


class Orchestrator:
    """Drives one or many conversation turns through the agent loop."""

    def __init__(
        self,
        backend: ToolUseBackend,
        tools: ToolRegistry,
        tts: TTSSynthesizer | None = None,
        *,
        system_prompt: str | None = None,
        speak: SPEAK_HOOK | None = None,
        on_interrupt: Callable[[], None] | None = None,
    ) -> None:
        self.backend = backend
        self.tools = tools
        self.tts = tts
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._speak = speak  # injected for tests; in production this is the Speaker
        self._on_interrupt = on_interrupt  # production: Speaker.interrupt
        self.state: AgentState = AgentState.IDLE
        self._history: list[dict] = []
        self._interrupt = asyncio.Event()

    def reset(self) -> None:
        self.state = AgentState.IDLE
        self._history.clear()
        self._interrupt.clear()

    def request_interrupt(self) -> None:
        """Signal the current turn to abort. Safe to call from any task."""
        self._interrupt.set()
        if self._on_interrupt is not None:
            try:
                self._on_interrupt()
            except Exception:
                log.exception("on_interrupt hook raised; continuing")

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    async def run_turn(
        self,
        user_text: str,
        *,
        max_tool_rounds: int = 4,
    ) -> TurnTrace:
        """Drive one turn: user utterance -> spoken reply.

        Multiple LLM rounds may happen if tool calls fire. Each round
        either streams prose (and is spoken) or emits tool calls (and is
        not).

        The interrupt event is not auto-cleared. If a caller has
        already requested interrupt (e.g. ``request_interrupt()`` fired
        while the previous turn was finishing), the new turn sees it
        set and short-circuits immediately. Call :meth:`reset` or
        :meth:`clear_interrupt` between turns when that's desired.
        """
        trace = TurnTrace()
        try:
            return await self._run_turn_inner(user_text, trace, max_tool_rounds)
        except Exception as e:
            log.exception("orchestrator turn failed")
            self._transition(AgentState.ERROR, trace)
            trace.error = f"{type(e).__name__}: {e}"
            return trace

    def clear_interrupt(self) -> None:
        """Drop a pending interrupt signal without resetting state."""
        self._interrupt.clear()

    async def _run_turn_inner(
        self,
        user_text: str,
        trace: TurnTrace,
        max_tool_rounds: int,
    ) -> TurnTrace:
        if not self._history:
            self._history.append({"role": "system", "content": self.system_prompt})
        self._history.append({"role": "user", "content": user_text})

        for _ in range(max_tool_rounds):
            if self._interrupt.is_set():
                return self._finish_interrupt(trace)
            self._transition(AgentState.THINKING, trace)
            specs = self.tools.specs()
            pending_calls: list[ToolCall] = []
            chunker = SentenceChunker()
            spoke_this_round = False
            done = False
            # Track text appended during this round so we can drop it
            # from history if the round also emits a tool call.
            round_text_start = len(trace.text_chunks)

            async for event in self.backend.stream(self._history, specs):
                if self._interrupt.is_set():
                    return self._finish_interrupt(trace)
                if event.tool_call is not None:
                    pending_calls.append(event.tool_call)
                    trace.tool_calls.append(event.tool_call)
                if event.text:
                    if not spoke_this_round:
                        self._transition(AgentState.SPEAKING, trace)
                        spoke_this_round = True
                    trace.text_chunks.append(event.text)
                    for chunk_text in chunker.feed(event.text):
                        await self._maybe_speak(chunk_text, trace)
                        if self._interrupt.is_set():
                            return self._finish_interrupt(trace)
                if event.done:
                    done = True
                    break

            # Flush any buffered text for this round.
            for chunk_text in chunker.flush():
                if self._interrupt.is_set():
                    return self._finish_interrupt(trace)
                if not spoke_this_round:
                    self._transition(AgentState.SPEAKING, trace)
                    spoke_this_round = True
                await self._maybe_speak(chunk_text, trace)

            if pending_calls:
                # If this round emitted prose AND tool calls, drop the
                # prose from the history-bound trace. The prose already
                # streamed through TTS, but it shouldn't be concatenated
                # into the next prose-only round's history entry. Log so
                # users notice when their model is interleaving.
                if len(trace.text_chunks) > round_text_start:
                    dropped = "".join(trace.text_chunks[round_text_start:])
                    log.warning(
                        "LLM emitted prose and tool_call in the same round; "
                        "prose was spoken but is dropped from history: %r",
                        dropped,
                    )
                    del trace.text_chunks[round_text_start:]
                # Record the assistant's tool-call turn in history.
                self._history.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.name, "arguments": dict(c.arguments)},
                            }
                            for c in pending_calls
                        ],
                    }
                )
                # Execute each call sequentially. No parallel calls in v1.
                for call in pending_calls:
                    result = await self.tools.call(call)
                    self._history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.to_message_content(),
                        }
                    )
                # Next round picks up the tool results.
                continue

            # No tool calls: assistant finished its prose turn.
            if trace.text_chunks:
                self._history.append({"role": "assistant", "content": "".join(trace.text_chunks)})
            if done or not pending_calls:
                self._transition(AgentState.IDLE, trace)
                return trace

        # Tool-round budget exhausted.
        self._transition(AgentState.ERROR, trace)
        trace.error = f"tool-call round limit exceeded ({max_tool_rounds})"
        return trace

    def _transition(self, new_state: AgentState, trace: TurnTrace) -> None:
        if self.state is not new_state:
            self.state = new_state
            trace.states.append(new_state)

    def _finish_interrupt(self, trace: TurnTrace) -> TurnTrace:
        trace.interrupted = True
        self._transition(AgentState.IDLE, trace)
        return trace

    async def _maybe_speak(self, text: str, trace: TurnTrace) -> None:
        if not text or self.tts is None:
            return
        async for tts_chunk in self.tts.synthesize(text):
            trace.tts_chunks.append(tts_chunk)
            if self._speak is not None:
                await self._speak(tts_chunk)


DEFAULT_SYSTEM_PROMPT = """\
You are whisper-agent, a hands-free voice assistant. The user is talking to you,
and your reply will be spoken aloud, so:

- Keep replies short, two to four sentences unless a longer answer is necessary.
- Don't use markdown, code fences, or numbered lists in your reply text.
- When you need to perform an action, call a tool. Don't narrate the call.
- After a tool returns, confirm what you did in plain English.
- If the user says "stop", "cancel", or "nevermind", stop and acknowledge.
"""


class MockBackend:
    """Test double for :class:`ToolUseBackend`.

    Constructed with a list of pre-canned event sequences, one per
    expected ``stream`` call. Useful for testing the orchestrator without
    a real model.
    """

    def __init__(self, scripts: list[list[LLMEvent]]) -> None:
        self._scripts: list[list[LLMEvent]] = list(scripts)
        self.calls: list[tuple[list[dict], list[ToolSpec]]] = []

    async def stream(
        self,
        messages: list[dict],
        tools: list[ToolSpec],
        *,
        temperature: float = 0.2,
    ) -> AsyncIterator[LLMEvent]:
        self.calls.append((list(messages), list(tools)))
        if not self._scripts:
            raise AssertionError("MockBackend ran out of scripted turns")
        for ev in self._scripts.pop(0):
            yield ev
            # Let the orchestrator interleave async work between events.
            await asyncio.sleep(0)

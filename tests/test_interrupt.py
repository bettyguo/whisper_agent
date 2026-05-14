"""Orchestrator interrupt path: setting the interrupt event aborts mid-turn."""

from __future__ import annotations

import asyncio

from whisper_agent.llm.tool_use import LLMEvent, ToolRegistry
from whisper_agent.orchestrator import AgentState, MockBackend, Orchestrator


def test_interrupt_before_first_event_short_circuits() -> None:
    backend = MockBackend(
        scripts=[
            [
                LLMEvent(text="this should not be spoken"),
                LLMEvent(done=True),
            ]
        ]
    )
    orch = Orchestrator(backend=backend, tools=ToolRegistry())

    async def go() -> None:
        orch.request_interrupt()  # before the turn starts
        return await orch.run_turn("hi")

    trace = asyncio.run(go())
    assert trace.interrupted is True
    assert orch.state is AgentState.IDLE


def test_interrupt_during_streaming_stops_reply() -> None:
    received = []

    async def speak(chunk):
        received.append(chunk)

    class FakeTTS:
        sample_rate = 24_000

        async def synthesize(self, text):
            import numpy as np

            from whisper_agent.tts.base import TTSChunk

            received.append(("synth", text))
            yield TTSChunk(
                samples=np.zeros(1, dtype=np.float32), sample_rate=self.sample_rate, is_last=True
            )

    backend = MockBackend(
        scripts=[
            [
                LLMEvent(text="First sentence. "),
                LLMEvent(text="Second sentence. "),
                LLMEvent(text="Third sentence. "),
                LLMEvent(done=True),
            ]
        ]
    )

    interrupt_calls = []
    orch = Orchestrator(
        backend=backend,
        tools=ToolRegistry(),
        tts=FakeTTS(),
        speak=speak,
        on_interrupt=lambda: interrupt_calls.append(1),
    )

    async def go():
        task = asyncio.create_task(orch.run_turn("speak a lot"))
        # Let the first chunk get processed, then interrupt.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        orch.request_interrupt()
        return await task

    trace = asyncio.run(go())
    assert trace.interrupted is True
    assert orch.state is AgentState.IDLE
    assert interrupt_calls == [1]


def test_request_interrupt_safe_when_idle() -> None:
    orch = Orchestrator(backend=MockBackend(scripts=[]), tools=ToolRegistry())
    orch.request_interrupt()
    assert orch.state is AgentState.IDLE

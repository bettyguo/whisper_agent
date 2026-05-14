"""Ollama backend over its HTTP /api/chat endpoint, with tool-use.

Uses ``httpx`` for streaming. Server URL defaults to localhost.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from whisper_agent.llm.tool_use import LLMEvent, ToolCall, ToolSpec

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OllamaConfig:
    """Ollama HTTP client config; all defaults are localhost."""

    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    keep_alive_seconds: int = 600  # keep model warm to protect latency budget


def parse_ollama_chunk(chunk: dict[str, Any]) -> list[LLMEvent]:
    """Pure-function parser for one streamed Ollama chunk.

    Exposed so tests can exercise the parsing logic without an HTTP server.
    Returns a list because a single chunk may yield multiple events
    (text + tool_call + done all possible in the last chunk).
    """
    events: list[LLMEvent] = []
    message = chunk.get("message") or {}
    content = message.get("content")
    if content:
        events.append(LLMEvent(text=content))
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                log.warning("could not parse tool-call arguments: %r", raw_args)
                continue
        else:
            arguments = dict(raw_args)
        events.append(LLMEvent(tool_call=ToolCall(name=name, arguments=arguments)))
    if chunk.get("done"):
        events.append(LLMEvent(done=True))
    return events


class OllamaBackend:
    """Streaming Ollama client implementing :class:`ToolUseBackend`."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()

    async def stream(
        self,
        messages: list[dict],
        tools: list[ToolSpec],
        *,
        temperature: float = 0.2,
    ) -> AsyncIterator[LLMEvent]:
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError(
                "Ollama backend needs `httpx`. Run `pip install whisper-agent[llm]`."
            ) from e

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "tools": [t.to_json_schema() for t in tools] if tools else None,
            "keep_alive": f"{self.config.keep_alive_seconds}s",
            "options": {"temperature": temperature},
        }
        # Drop the tools key entirely when empty; some Ollama versions reject ``null``.
        if payload["tools"] is None:
            del payload["tools"]

        url = f"{self.config.base_url}/api/chat"
        async with (
            httpx.AsyncClient(timeout=None) as client,
            client.stream("POST", url, json=payload) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("non-JSON line from Ollama: %r", line)
                    continue
                for ev in parse_ollama_chunk(chunk):
                    yield ev
                    if ev.done:
                        return

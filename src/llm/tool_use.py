"""Tool-use core: types, registry, unified backend protocol.

The orchestrator consumes :class:`LLMEvent`s from a
:class:`ToolUseBackend`. An ``LLMEvent`` is either a partial-text token
or a parsed tool call. The orchestrator dispatches calls through
:class:`ToolRegistry` and feeds the result back to the model as a
follow-up turn.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    """Describes a tool to the LLM. JSON-Schema parameters."""

    name: str
    description: str
    parameters: Mapping[str, Any]  # JSON Schema for arguments
    requires_confirmation: bool = False

    def to_json_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """A parsed tool call emitted by the model."""

    name: str
    arguments: Mapping[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass(frozen=True)
class ToolResult:
    """The result of executing one tool call."""

    tool_call_id: str
    ok: bool
    result: Any = None
    error: str | None = None

    def to_message_content(self) -> str:
        """Serialize as a `tool` role message content payload."""
        import json

        if self.ok:
            return json.dumps({"ok": True, "result": self.result})
        return json.dumps({"ok": False, "error": self.error or "unknown error"})


@dataclass(frozen=True)
class LLMEvent:
    """An event emitted by a streaming :class:`ToolUseBackend`.

    Exactly one of ``text``, ``tool_call``, or ``done`` is meaningful per event.
    """

    text: str | None = None
    tool_call: ToolCall | None = None
    done: bool = False


@dataclass(frozen=True)
class Tool:
    """A callable plus its spec. The callable receives keyword arguments
    drawn from the parsed JSON arguments."""

    spec: ToolSpec
    handler: Callable[..., Any | Awaitable[Any]]


class ToolRegistry:
    """In-process registry of tools available to the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"tool {tool.spec.name!r} already registered")
        self._tools[tool.spec.name] = tool

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def has(self, name: str) -> bool:
        return name in self._tools

    async def call(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                ok=False,
                error=f"unknown tool: {call.name!r}",
            )
        try:
            value = tool.handler(**call.arguments)
            if inspect.isawaitable(value):
                value = await value
        except TypeError as e:
            return ToolResult(
                tool_call_id=call.id,
                ok=False,
                error=f"invalid arguments for {call.name!r}: {e}",
            )
        except Exception as e:
            log.exception("tool %s raised", call.name)
            return ToolResult(
                tool_call_id=call.id,
                ok=False,
                error=f"{type(e).__name__}: {e}",
            )
        return ToolResult(tool_call_id=call.id, ok=True, result=value)


class ToolUseBackend(Protocol):
    """A streaming LLM backend that may emit tool calls.

    ``stream`` is declared as plain ``def`` returning an
    :class:`~typing.AsyncIterator`. That is the right Protocol shape
    for async-generator implementations (``async def`` + ``yield``
    already returns an async iterator on call); declaring it as
    ``async def`` would type the call as a coroutine returning an
    iterator, which mypy then flags when callers do
    ``async for ev in backend.stream(...)``.
    """

    def stream(
        self,
        messages: list[dict],
        tools: list[ToolSpec],
        *,
        temperature: float = 0.2,
    ) -> AsyncIterator[LLMEvent]:
        """Yield :class:`LLMEvent` items. Backends must end with ``done=True``."""
        ...

"""MCP (Model Context Protocol) client + registry bridge.

Spawns MCP server subprocesses, speaks JSON-RPC over stdio, and bridges
the discovered tools into the agent's :class:`ToolRegistry`.

Wire format (line-delimited):
- Request:  ``{"jsonrpc": "2.0", "id": N, "method": M, "params": P}``
- Response: ``{"jsonrpc": "2.0", "id": N, "result": ...}`` or
            ``{"jsonrpc": "2.0", "id": N, "error": {...}}``

We use only ``initialize``, ``tools/list``, and ``tools/call``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from whisper_agent.llm.tool_use import Tool, ToolRegistry, ToolSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPServerConfig:
    """One MCP server definition (from ``~/.config/whisper-agent/mcp.toml``)."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    start_timeout_s: float = 10.0


@dataclass
class MCPTool:
    """A tool exposed by an MCP server."""

    server: str
    raw_name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        """Name as seen by the LLM: ``mcp.<server>.<tool>``."""
        return f"mcp.{self.server}.{self.raw_name}"


class MCPError(RuntimeError):
    """Raised when an MCP request fails."""


class MCPClient:
    """JSON-RPC client speaking MCP over a subprocess's stdio.

    Lifetime:
        ``start()`` spawns the subprocess. ``stop()`` closes stdin and
        waits for clean exit, then kills if needed. Restart on
        unexpected exit is the caller's job (typically
        :class:`MCPRegistry`).
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id: int = 1
        self._send_lock = asyncio.Lock()
        self._closed = False

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        if self._proc is not None:
            return
        env = dict(self.config.env or {})
        self._proc = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env if env else None,
        )
        self._reader_task = asyncio.create_task(
            self._reader_loop(), name=f"mcp-reader:{self.config.name}"
        )
        # Drain stderr in a background task. Without this, a chatty MCP
        # server fills the OS pipe buffer (~8 KB on Windows, ~64 KB on
        # Linux) and the subprocess blocks indefinitely on its next stderr write.
        self._stderr_task = asyncio.create_task(
            self._stderr_drain(), name=f"mcp-stderr:{self.config.name}"
        )
        log.info("MCP server %s started (pid=%s)", self.config.name, self._proc.pid)
        # Per MCP spec, the client sends `initialize` first.
        await self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "whisper-agent", "version": "0.0.1"},
            },
            timeout_s=self.config.start_timeout_s,
        )

    async def stop(self) -> None:
        self._closed = True
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None and not self._proc.stdin.is_closing():
                self._proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=2.0)
        except TimeoutError:
            log.warning("MCP server %s did not exit cleanly; terminating", self.config.name)
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=1.0)
            except TimeoutError:
                self._proc.kill()
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._reader_task = None
        self._stderr_task = None
        self._proc = None

    async def list_tools(self) -> list[MCPTool]:
        result = await self._request("tools/list", {})
        tools_raw = result.get("tools") or []
        return [
            MCPTool(
                server=self.config.name,
                raw_name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema") or {},
            )
            for t in tools_raw
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        result = await self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        return result

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        if self._proc is None or self._proc.stdin is None:
            raise MCPError(f"MCP server {self.config.name} is not running")
        async with self._send_lock:
            rid = self._next_id
            self._next_id += 1
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            self._pending[rid] = fut
            payload = (
                json.dumps(
                    {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
                ).encode("utf-8")
                + b"\n"
            )
            try:
                self._proc.stdin.write(payload)
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as e:
                self._pending.pop(rid, None)
                raise MCPError(f"MCP server {self.config.name} stdin closed") from e
        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except TimeoutError as e:
            self._pending.pop(rid, None)
            raise MCPError(f"{method} on {self.config.name} timed out") from e

    async def _reader_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while not self._closed:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    log.warning("MCP %s emitted non-JSON line: %r", self.config.name, line)
                    continue
                rid = msg.get("id")
                if rid is None:
                    # Notification, not a response. Ignore.
                    continue
                fut = self._pending.pop(rid, None)
                if fut is None or fut.done():
                    continue
                if "error" in msg:
                    fut.set_exception(MCPError(str(msg["error"])))
                else:
                    fut.set_result(msg.get("result") or {})
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("MCP %s reader crashed", self.config.name)
        finally:
            # Fail pending requests so callers don't hang forever.
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(MCPError(f"MCP {self.config.name} disconnected"))
            self._pending.clear()

    async def _stderr_drain(self) -> None:
        """Consume the subprocess's stderr so the OS pipe buffer never fills.

        Without this, a chatty MCP server (logs, deprecation warnings)
        will fill the pipe and block on its next ``write()``, which
        deadlocks the whole client.

        Use chunked ``.read(N)`` rather than ``.readline()``: asyncio's
        StreamReader buffer caps at 64 KB by default, so a server that
        writes a single >64 KB line without a newline would hang
        readline() while the underlying pipe filled.
        """
        assert self._proc is not None and self._proc.stderr is not None
        try:
            while not self._closed:
                chunk = await self._proc.stderr.read(4096)
                if not chunk:
                    break
                # Log at debug level so noisy servers don't spam normal logs.
                log.debug(
                    "MCP %s stderr: %s",
                    self.config.name,
                    chunk.decode("utf-8", errors="replace").rstrip(),
                )
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("MCP %s stderr drain crashed", self.config.name)


class MCPRegistry:
    """Manages multiple :class:`MCPClient` instances and registers their
    discovered tools into a :class:`ToolRegistry`."""

    def __init__(self, configs: list[MCPServerConfig]) -> None:
        self._configs = {c.name: c for c in configs}
        self._clients: dict[str, MCPClient] = {}
        self._tools: dict[str, MCPTool] = {}  # qualified_name -> MCPTool

    @property
    def server_names(self) -> list[str]:
        return list(self._configs.keys())

    @property
    def discovered_tools(self) -> list[MCPTool]:
        return list(self._tools.values())

    def is_running(self, server: str) -> bool:
        client = self._clients.get(server)
        return client is not None and client.is_running

    async def discover_all(self, registry: ToolRegistry) -> None:
        """Start every configured server and register their tools."""
        for name in self._configs:
            await self.discover(name, registry)

    async def discover(self, name: str, registry: ToolRegistry) -> None:
        """Start one server and add its tools to ``registry``."""
        config = self._configs.get(name)
        if config is None:
            raise KeyError(f"no MCP server named {name!r}")
        client = self._clients.get(name)
        if client is None or not client.is_running:
            client = MCPClient(config)
            await client.start()
            self._clients[name] = client
        for tool in await client.list_tools():
            self._tools[tool.qualified_name] = tool
            registry.register(
                Tool(
                    spec=ToolSpec(
                        name=tool.qualified_name,
                        description=tool.description,
                        parameters=tool.input_schema
                        or {
                            "type": "object",
                            "properties": {},
                        },
                    ),
                    handler=self._make_handler(tool),
                )
            )

    def _make_handler(self, tool: MCPTool):
        async def _call(**arguments: Any) -> Any:
            client = self._clients.get(tool.server)
            if client is None or not client.is_running:
                raise MCPError(f"MCP server {tool.server!r} is not running")
            return await client.call_tool(tool.raw_name, arguments)

        return _call

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.stop()
        self._clients.clear()

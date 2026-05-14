"""MCP client tests using a stdio echo subprocess.

We launch a tiny Python "MCP server" that speaks JSON-RPC over stdio
following the bits of the protocol we use (initialize, tools/list,
tools/call). This exercises the real subprocess + JSON parsing without
needing any third-party MCP implementation.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from whisper_agent.llm.tool_use import ToolCall, ToolRegistry
from whisper_agent.tools.mcp import (
    MCPClient,
    MCPError,
    MCPRegistry,
    MCPServerConfig,
)

SERVER_SCRIPT = Path(__file__).parent / "_fake_mcp_server.py"


def _make_config(name: str = "fake", **extra) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        command=sys.executable,
        args=(str(SERVER_SCRIPT), *extra.get("args", ())),
        env=extra.get("env"),
        start_timeout_s=5.0,
    )


def test_client_lifecycle_round_trip() -> None:
    async def run() -> None:
        client = MCPClient(_make_config())
        await client.start()
        try:
            assert client.is_running
            tools = await client.list_tools()
            assert {t.raw_name for t in tools} == {"echo", "add"}
        finally:
            await client.stop()
        assert client.is_running is False

    asyncio.run(run())


def test_client_call_tool_returns_result() -> None:
    async def run() -> None:
        client = MCPClient(_make_config())
        await client.start()
        try:
            result = await client.call_tool("echo", {"text": "hi"})
            assert result == {"echoed": "hi"}
            result = await client.call_tool("add", {"a": 2, "b": 5})
            assert result == {"sum": 7}
        finally:
            await client.stop()

    asyncio.run(run())


def test_client_call_unknown_tool_raises() -> None:
    async def run() -> None:
        client = MCPClient(_make_config())
        await client.start()
        try:
            with pytest.raises(MCPError):
                await client.call_tool("nope", {})
        finally:
            await client.stop()

    asyncio.run(run())


def test_registry_discovers_and_dispatches() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        mcp = MCPRegistry([_make_config("fake")])
        await mcp.discover_all(registry)
        try:
            names = registry.names()
            assert "mcp.fake.echo" in names
            assert "mcp.fake.add" in names

            r = await registry.call(ToolCall(name="mcp.fake.echo", arguments={"text": "hello"}))
            assert r.ok is True
            assert r.result == {"echoed": "hello"}
        finally:
            await mcp.aclose()

    asyncio.run(run())


def test_qualified_name_format() -> None:
    from whisper_agent.tools.mcp import MCPTool

    t = MCPTool(server="paper-skills", raw_name="search", description="")
    assert t.qualified_name == "mcp.paper-skills.search"


def test_client_survives_noisy_stderr_subprocess() -> None:
    """If the client doesn't drain stderr, an MCP server writing past
    the OS pipe buffer (~8 KB Windows, ~64 KB Linux) blocks on its
    next stderr write and deadlocks the whole client.

    Launch the fake server with --noisy-stderr=200000 so every request
    emits ~200 KB; issue several tools/call's with a per-call timeout.
    Without the drain task, one will time out.
    """

    async def run() -> None:
        # 200 KB per request is well beyond Windows (8 KB) and Linux
        # (64 KB) default pipe buffers. After ~1-2 round-trips the
        # buffer would fill and the next stderr write would block.
        config = MCPServerConfig(
            name="noisy",
            command=sys.executable,
            args=(str(SERVER_SCRIPT), "--noisy-stderr=200000"),
            env=None,
            start_timeout_s=5.0,
        )
        client = MCPClient(config)
        await client.start()
        try:
            for i in range(5):
                result = await asyncio.wait_for(
                    client.call_tool("echo", {"text": f"call-{i}"}),
                    timeout=5.0,
                )
                assert result == {"echoed": f"call-{i}"}
        finally:
            await client.stop()

    asyncio.run(run())

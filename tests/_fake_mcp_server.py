"""Minimal MCP-shaped JSON-RPC echo server used by tests.

Implements just enough of the protocol that :class:`MCPClient` uses:

- ``initialize`` -> ``{}``
- ``tools/list`` -> two synthetic tools (``echo``, ``add``)
- ``tools/call`` -> dispatch to a tiny in-process handler
- Any other method -> JSON-RPC error response

Reads one JSON-RPC message per line on stdin; writes one per line on
stdout. Lives in ``tests/`` because it is test infrastructure, not part
of the shipped package.
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _send(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _ok(req_id: int, result: Any) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "result": result})


def _err(req_id: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": message}})


TOOLS = [
    {
        "name": "echo",
        "description": "Echo back the supplied text.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "Sum two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
]


def _dispatch(method: str, params: dict[str, Any]) -> Any:
    if method == "initialize":
        return {"protocolVersion": "2024-11-05", "serverInfo": {"name": "fake-mcp"}}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "echo":
            return {"echoed": args.get("text", "")}
        if name == "add":
            return {"sum": args["a"] + args["b"]}
        raise KeyError(f"unknown tool: {name}")
    raise KeyError(f"unknown method: {method}")


def main() -> int:
    # Optional flag --noisy-stderr=N writes N bytes to stderr right after
    # each request. Used by tests to verify the parent drains stderr.
    noisy_bytes = 0
    for arg in sys.argv[1:]:
        if arg.startswith("--noisy-stderr="):
            noisy_bytes = int(arg.split("=", 1)[1])

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = msg.get("id")
        method = msg.get("method") or ""
        params = msg.get("params") or {}
        try:
            result = _dispatch(method, params)
        except KeyError as e:
            _err(rid, str(e))
        else:
            _ok(rid, result)
        if noisy_bytes > 0:
            # Write a wall of stderr in chunks so we exceed any reasonable
            # OS pipe buffer (~8 KB on Windows, ~64 KB on Linux).
            sys.stderr.write("x" * noisy_bytes + "\n")
            sys.stderr.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

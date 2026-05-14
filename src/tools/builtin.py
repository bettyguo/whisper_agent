"""Built-in tools: ``fs.read``, ``fs.write``, ``fs.search``, ``notes.append``.

All four refuse paths that escape a configured workspace root. The root
is set by :func:`register_builtin_tools`; the default is the process
working directory at registration time. Pass ``sandbox=False`` to opt
out, for tooling that genuinely needs filesystem-wide access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from whisper_agent.llm.tool_use import Tool, ToolRegistry, ToolSpec


class WorkspaceSandboxError(PermissionError):
    """Raised when a tool tries to access a path outside the workspace root."""


# Module-level workspace root. ``None`` means the sandbox is disabled.
# Set by :func:`register_builtin_tools`; the default is ``Path.cwd()`` at
# registration time.
_workspace_root: Path | None = None


def _resolve_within_workspace(path: str) -> Path:
    """Resolve ``path`` and assert it lives under the workspace root.

    Returns the resolved path. Raises :class:`WorkspaceSandboxError` if
    the workspace is configured and the resolved path escapes it.
    """
    resolved = Path(path).expanduser().resolve()
    if _workspace_root is None:
        return resolved
    try:
        resolved.relative_to(_workspace_root)
    except ValueError as e:
        raise WorkspaceSandboxError(
            f"path escapes workspace root: {resolved} (root={_workspace_root})"
        ) from e
    return resolved


def _read_file(path: str, max_bytes: int = 256_000) -> dict[str, Any]:
    p = _resolve_within_workspace(path)
    if not p.exists():
        raise FileNotFoundError(f"no such path: {path}")
    if p.is_dir():
        entries = sorted(e.name for e in p.iterdir())
        return {"kind": "directory", "path": str(p), "entries": entries}
    # Check size before reading so a multi-GB file doesn't blow up memory
    # just to be truncated to max_bytes.
    size = p.stat().st_size
    if size > max_bytes:
        with p.open("rb") as fp:
            data = fp.read(max_bytes)
        truncated = True
    else:
        data = p.read_bytes()
        truncated = False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "kind": "binary",
            "path": str(p),
            "size_bytes": size,
        }
    return {
        "kind": "file",
        "path": str(p),
        "content": text,
        "truncated": truncated,
    }


def _write_file(path: str, content: str, *, append: bool = False) -> dict[str, Any]:
    p = _resolve_within_workspace(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with p.open(mode, encoding="utf-8") as fp:
        fp.write(content)
    return {"path": str(p), "bytes": len(content.encode("utf-8")), "append": append}


def _search_files(
    pattern: str,
    *,
    root: str = ".",
    max_results: int = 200,
) -> dict[str, Any]:
    base = _resolve_within_workspace(root)
    matches: list[str] = []
    # Path.glob understands ``**`` natively; fnmatch does not. For a plain
    # filename pattern (no slashes) we still want recursive matching, so
    # promote bare patterns to ``**/<pattern>``.
    glob_pattern = pattern if "/" in pattern or "\\" in pattern else f"**/{pattern}"
    for p in base.glob(glob_pattern):
        if not p.is_file():
            continue
        matches.append(p.relative_to(base).as_posix())
        if len(matches) >= max_results:
            break
    return {"root": str(base), "pattern": pattern, "matches": matches}


def _append_note(text: str, *, notes_path: str = "~/.whisper-agent-notes.md") -> dict[str, Any]:
    p = _resolve_within_workspace(notes_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fp:
        fp.write(text)
        if not text.endswith("\n"):
            fp.write("\n")
    return {"path": str(p), "appended_chars": len(text)}


def register_builtin_tools(
    registry: ToolRegistry,
    *,
    workspace_root: Path | str | None = None,
    sandbox: bool = True,
) -> None:
    """Add the v1 built-in tools to ``registry`` in place.

    When ``sandbox=True`` (the default), all four tools refuse paths
    that escape the workspace root. The root defaults to
    :func:`pathlib.Path.cwd` at the time of this call; pass
    ``workspace_root`` to override. Pass ``sandbox=False`` to opt out
    of the sandbox (rare; used by integration tools that need
    filesystem-wide access).
    """
    global _workspace_root
    if not sandbox:
        _workspace_root = None
    elif workspace_root is None:
        _workspace_root = Path.cwd().resolve()
    else:
        _workspace_root = Path(workspace_root).expanduser().resolve()
    registry.register(
        Tool(
            spec=ToolSpec(
                name="fs.read",
                description="Read a file or list a directory at the given path.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Filesystem path"},
                        "max_bytes": {
                            "type": "integer",
                            "description": "Maximum bytes to read (default 256000)",
                            "default": 256_000,
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            handler=_read_file,
        )
    )
    registry.register(
        Tool(
            spec=ToolSpec(
                name="fs.write",
                description="Write or append text to a file. Creates parent dirs.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "append": {"type": "boolean", "default": False},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                requires_confirmation=True,
            ),
            handler=_write_file,
        )
    )
    registry.register(
        Tool(
            spec=ToolSpec(
                name="fs.search",
                description="Find files under a directory by glob pattern.",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "fnmatch pattern, e.g. '**/*.py'",
                        },
                        "root": {"type": "string", "default": "."},
                        "max_results": {"type": "integer", "default": 200},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            ),
            handler=_search_files,
        )
    )
    registry.register(
        Tool(
            spec=ToolSpec(
                name="notes.append",
                description="Append a line of text to the user's scratch notes file.",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "notes_path": {
                            "type": "string",
                            "default": "~/.whisper-agent-notes.md",
                        },
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            ),
            handler=_append_note,
        )
    )

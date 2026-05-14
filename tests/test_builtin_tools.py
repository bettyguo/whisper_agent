"""Built-in tools: fs.read, fs.write, fs.search, notes.append."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from whisper_agent.llm.tool_use import ToolCall, ToolRegistry
from whisper_agent.tools import register_builtin_tools


@pytest.fixture
def registry(tmp_path: Path) -> ToolRegistry:
    r = ToolRegistry()
    register_builtin_tools(r, workspace_root=tmp_path)
    return r


def test_fs_write_rejects_paths_outside_workspace(registry: ToolRegistry, tmp_path: Path) -> None:
    # Workspace root is tmp_path; writing to its parent must fail.
    outside = tmp_path.parent / "escaped.txt"
    result = asyncio.run(
        registry.call(
            ToolCall(
                name="fs.write",
                arguments={"path": str(outside), "content": "leak"},
            )
        )
    )
    assert result.ok is False
    assert "workspace" in (result.error or "").lower()
    assert not outside.exists()


def test_fs_write_rejects_dotdot_traversal(registry: ToolRegistry, tmp_path: Path) -> None:
    sneaky = str(tmp_path / "sub" / ".." / ".." / "escaped.txt")
    result = asyncio.run(
        registry.call(
            ToolCall(
                name="fs.write",
                arguments={"path": sneaky, "content": "leak"},
            )
        )
    )
    assert result.ok is False
    assert "workspace" in (result.error or "").lower()


def test_fs_read_rejects_paths_outside_workspace(registry: ToolRegistry, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        result = asyncio.run(
            registry.call(ToolCall(name="fs.read", arguments={"path": str(outside)}))
        )
        assert result.ok is False
        assert "workspace" in (result.error or "").lower()
    finally:
        outside.unlink(missing_ok=True)


def test_fs_search_rejects_root_outside_workspace(registry: ToolRegistry, tmp_path: Path) -> None:
    result = asyncio.run(
        registry.call(
            ToolCall(
                name="fs.search",
                arguments={"pattern": "*", "root": str(tmp_path.parent)},
            )
        )
    )
    assert result.ok is False
    assert "workspace" in (result.error or "").lower()


def test_notes_append_rejects_paths_outside_workspace(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    outside = tmp_path.parent / "evil-notes.md"
    result = asyncio.run(
        registry.call(
            ToolCall(
                name="notes.append",
                arguments={"text": "hi", "notes_path": str(outside)},
            )
        )
    )
    assert result.ok is False
    assert "workspace" in (result.error or "").lower()
    assert not outside.exists()


def test_sandbox_can_be_disabled_for_explicit_opt_in() -> None:
    # The "no sandbox" mode exists so existing callers (and integration
    # tests) can opt out. Most callers should NOT use this.
    r = ToolRegistry()
    register_builtin_tools(r, sandbox=False)
    # Just confirm the registration path doesn't blow up; behaviour-on-call
    # is the same as before the sandbox was introduced.
    assert "fs.read" in r.names()


def test_registers_expected_tools(registry: ToolRegistry) -> None:
    assert set(registry.names()) >= {"fs.read", "fs.write", "fs.search", "notes.append"}


def test_fs_write_requires_confirmation(registry: ToolRegistry) -> None:
    spec = next(s for s in registry.specs() if s.name == "fs.write")
    assert spec.requires_confirmation is True


def test_fs_read_does_not_require_confirmation(registry: ToolRegistry) -> None:
    spec = next(s for s in registry.specs() if s.name == "fs.read")
    assert spec.requires_confirmation is False


def test_fs_read_file_round_trip(registry: ToolRegistry, tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    target.write_bytes(b"hello world\n")
    result = asyncio.run(registry.call(ToolCall(name="fs.read", arguments={"path": str(target)})))
    assert result.ok is True
    assert result.result["kind"] == "file"
    assert result.result["content"] == "hello world\n"
    assert result.result["truncated"] is False


def test_fs_read_directory_lists_entries(registry: ToolRegistry, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    result = asyncio.run(registry.call(ToolCall(name="fs.read", arguments={"path": str(tmp_path)})))
    assert result.ok is True
    assert result.result["kind"] == "directory"
    assert set(result.result["entries"]) == {"a.txt", "b.txt"}


def test_fs_read_missing_returns_error(registry: ToolRegistry, tmp_path: Path) -> None:
    result = asyncio.run(
        registry.call(ToolCall(name="fs.read", arguments={"path": str(tmp_path / "no.txt")}))
    )
    assert result.ok is False
    assert "no such path" in (result.error or "")


def test_fs_read_truncates_large_files(registry: ToolRegistry, tmp_path: Path) -> None:
    big = tmp_path / "big.txt"
    big.write_text("x" * 10_000)
    result = asyncio.run(
        registry.call(
            ToolCall(
                name="fs.read",
                arguments={"path": str(big), "max_bytes": 100},
            )
        )
    )
    assert result.ok is True
    assert result.result["truncated"] is True
    assert len(result.result["content"]) == 100


def test_fs_read_does_not_load_full_file_when_truncating(
    registry: ToolRegistry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_read_file`` used to call ``p.read_bytes()`` unconditionally,
    OOMing on multi-GB files even with a small ``max_bytes``. It now
    size-checks first and streams the first N bytes via
    ``open().read(N)``.

    Monkey-patches Path.read_bytes to raise; the streamed path doesn't
    take it.
    """
    big = tmp_path / "big.bin"
    big.write_bytes(b"y" * 1_000_000)

    def boom(self: Path) -> bytes:
        raise AssertionError(f"read_bytes called on {self}; truncated reads must stream, not slurp")

    monkeypatch.setattr(Path, "read_bytes", boom)

    result = asyncio.run(
        registry.call(
            ToolCall(
                name="fs.read",
                arguments={"path": str(big), "max_bytes": 100},
            )
        )
    )
    assert result.ok is True
    assert result.result["truncated"] is True
    assert len(result.result["content"]) == 100


def test_fs_write_creates_file(registry: ToolRegistry, tmp_path: Path) -> None:
    target = tmp_path / "out" / "file.txt"
    result = asyncio.run(
        registry.call(
            ToolCall(
                name="fs.write",
                arguments={"path": str(target), "content": "hi"},
            )
        )
    )
    assert result.ok is True
    assert target.read_text(encoding="utf-8") == "hi"


def test_fs_write_appends(registry: ToolRegistry, tmp_path: Path) -> None:
    target = tmp_path / "log.txt"
    target.write_text("line1\n", encoding="utf-8")
    asyncio.run(
        registry.call(
            ToolCall(
                name="fs.write",
                arguments={"path": str(target), "content": "line2\n", "append": True},
            )
        )
    )
    assert target.read_text(encoding="utf-8") == "line1\nline2\n"


def test_fs_search_matches_glob(registry: ToolRegistry, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("")
    result = asyncio.run(
        registry.call(
            ToolCall(
                name="fs.search",
                arguments={"pattern": "**/*.py", "root": str(tmp_path)},
            )
        )
    )
    assert result.ok is True
    rel = sorted(result.result["matches"])
    # fnmatch's '**' is permissive but matches both top-level and nested .py files
    assert "a.py" in rel
    assert "sub/c.py" in rel


def test_fs_search_respects_max_results(registry: ToolRegistry, tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"f{i}.txt").write_text("")
    result = asyncio.run(
        registry.call(
            ToolCall(
                name="fs.search",
                arguments={"pattern": "**/*.txt", "root": str(tmp_path), "max_results": 3},
            )
        )
    )
    assert len(result.result["matches"]) == 3


def test_notes_append_creates_file(registry: ToolRegistry, tmp_path: Path) -> None:
    notes = tmp_path / "n.md"
    result = asyncio.run(
        registry.call(
            ToolCall(
                name="notes.append",
                arguments={"text": "first note", "notes_path": str(notes)},
            )
        )
    )
    assert result.ok is True
    assert notes.read_text(encoding="utf-8") == "first note\n"


def test_notes_append_adds_trailing_newline_only_if_missing(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    notes = tmp_path / "n.md"
    asyncio.run(
        registry.call(
            ToolCall(
                name="notes.append",
                arguments={"text": "line\n", "notes_path": str(notes)},
            )
        )
    )
    asyncio.run(
        registry.call(
            ToolCall(
                name="notes.append",
                arguments={"text": "second", "notes_path": str(notes)},
            )
        )
    )
    assert notes.read_text(encoding="utf-8") == "line\nsecond\n"

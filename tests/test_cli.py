"""CLI surface tests using typer's CliRunner."""

from __future__ import annotations

from typer.testing import CliRunner

from whisper_agent import __version__
from whisper_agent.cli import app


def _runner() -> CliRunner:
    return CliRunner()


def test_version_prints_installed_version() -> None:
    result = _runner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_doctor_default_passes() -> None:
    result = _runner().invoke(app, ["doctor", "--ascii", "--no-runtime"])
    assert result.exit_code == 0
    assert "offline" in result.stdout
    assert "stt.backend" in result.stdout


def test_doctor_verify_offline_passes_default_config() -> None:
    result = _runner().invoke(app, ["doctor", "--ascii", "--no-runtime", "--verify-offline"])
    assert result.exit_code == 0


def test_talk_stub_is_invocable() -> None:
    result = _runner().invoke(app, ["talk"])
    assert result.exit_code == 0
    assert "talk" in result.stdout.lower()


def test_listen_stub_is_invocable() -> None:
    result = _runner().invoke(app, ["listen", "--wake", "hey computer"])
    assert result.exit_code == 0
    assert "wake" in result.stdout.lower()


def test_mcp_list_runs() -> None:
    result = _runner().invoke(app, ["mcp", "list"])
    assert result.exit_code == 0
    assert "mcp.toml" in result.stdout


def test_wake_subcommands_run() -> None:
    for sub in (["wake", "train", "hey linus"], ["wake", "test"]):
        result = _runner().invoke(app, sub)
        assert result.exit_code == 0

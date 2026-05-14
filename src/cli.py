"""whisper-agent CLI.

Subcommands: ``version``, ``doctor``, ``talk``, ``listen``, ``mcp``,
``wake``. Most accept ``--ascii`` for screen-reader-friendly output.
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console

from whisper_agent import __version__
from whisper_agent.doctor import (
    Severity,
    format_report,
    run_diagnostics,
    run_runtime_checks,
)

app = typer.Typer(
    name="whisper-agent",
    help="Voice-driven local agent loop. STT -> local LLM + tools -> TTS. Offline by default.",
    no_args_is_help=True,
    add_completion=False,
)

mcp_app = typer.Typer(help="Manage MCP servers")
wake_app = typer.Typer(help="Wake-word utilities")
app.add_typer(mcp_app, name="mcp")
app.add_typer(wake_app, name="wake")

console = Console()


@app.command()
def version() -> None:
    """Print the installed whisper-agent version."""
    console.print(f"whisper-agent {__version__}")


@app.command()
def doctor(
    verify_offline: bool = typer.Option(
        False,
        "--verify-offline",
        help="Exit non-zero if any cloud backend is reachable / configured.",
    ),
    runtime: bool = typer.Option(
        True,
        "--runtime/--no-runtime",
        help="Run the runtime checks that touch PATH / network.",
    ),
    ascii_only: bool = typer.Option(
        False,
        "--ascii",
        help="Plain-text output (screen-reader friendly).",
    ),
) -> None:
    """Self-diagnostic: model presence, audio devices, offline-only sanity check."""
    report = run_diagnostics()
    if runtime:
        runtime_report = run_runtime_checks()
        report.findings.extend(runtime_report.findings)

    out = format_report(report, ascii_only=ascii_only)
    if ascii_only:
        print(out)
    else:
        console.print(out)

    if verify_offline and not report.offline_ok:
        sys.exit(2)
    if not report.all_ok:
        sys.exit(1)


@app.command()
def talk(
    key: str = typer.Option("space", "--key", help="push-to-talk key"),
    mode: str = typer.Option(
        "push-to-talk",
        "--mode",
        help="push-to-talk | wake | continuous",
    ),
) -> None:
    """Push-to-talk session.

    Requires PortAudio (microphone + speaker). The live pipeline wiring
    is not yet integrated; this entrypoint reserves the CLI surface.
    """
    console.print(
        f"[yellow]talk[/yellow] is not wired to the live pipeline yet. mode={mode} key={key}"
    )


@app.command()
def listen(
    wake: str = typer.Option(
        "hey computer",
        "--wake",
        help="Wake phrase, or a path to a custom .onnx model.",
    ),
    continuous: bool = typer.Option(
        False,
        "--continuous",
        help="Skip the wake word; mic stays open and is gated by VAD.",
    ),
) -> None:
    """Wake-word or continuous session.

    Requires PortAudio + openWakeWord.
    """
    mode = "continuous" if continuous else "wake"
    console.print(
        f"[yellow]listen[/yellow] is not wired to the live pipeline yet. mode={mode} wake={wake!r}"
    )


@mcp_app.command("list")
def mcp_list() -> None:
    """List configured MCP servers."""
    console.print(
        "Configured MCP servers are read from [bold]~/.config/whisper-agent/mcp.toml[/bold]."
    )


@mcp_app.command("test")
def mcp_test(
    name: str = typer.Argument(..., help="server name from mcp.toml"),
) -> None:
    """Start a configured MCP server and list its tools."""
    console.print(f"[yellow]mcp test {name}[/yellow] is not implemented yet.")


@wake_app.command("train")
def wake_train(
    phrase: str = typer.Argument(..., help="custom wake phrase, e.g. 'hey linus'"),
) -> None:
    """Train a custom wake-word model via openWakeWord."""
    console.print(f"[yellow]wake train {phrase!r}[/yellow] is not implemented yet.")


@wake_app.command("test")
def wake_test(
    phrase: str = typer.Option("hey_computer", "--phrase"),
) -> None:
    """Listen for the wake phrase and print confidence scores live."""
    console.print(f"[yellow]wake test --phrase {phrase!r}[/yellow] is not implemented yet.")


__all__ = ["Severity", "app"]


if __name__ == "__main__":
    app()

"""Rich TUI.

:class:`TranscriptView` is a pure data model with a :meth:`render`
method that returns a Rich renderable. :class:`LiveTUI` wraps a Rich
``Live`` for the optional live-update display; the CLI also has an
``--ascii`` mode that skips Live entirely.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from whisper_agent.orchestrator import AgentState

if TYPE_CHECKING:
    from rich.console import RenderableType


@dataclass(frozen=True)
class TranscriptLine:
    when: datetime
    speaker: str  # "user" | "agent" | "system"
    text: str


@dataclass
class TranscriptView:
    """In-memory rolling transcript with rendering."""

    mode: str = "push-to-talk"
    state: AgentState = AgentState.IDLE
    cloud_mode: bool = False  # surfaces a warning banner if a cloud backend is active
    lines: deque[TranscriptLine] = field(default_factory=lambda: deque(maxlen=200))
    tool_calls: int = 0
    wakes: int = 0
    interrupts: int = 0
    last_latency_ms: float | None = None

    def add(self, speaker: str, text: str) -> None:
        clean = text.strip()
        if not clean:
            return
        self.lines.append(TranscriptLine(when=datetime.now(), speaker=speaker, text=clean))

    def set_state(self, state: AgentState) -> None:
        self.state = state

    def record_tool_call(self) -> None:
        self.tool_calls += 1

    def record_wake(self) -> None:
        self.wakes += 1

    def record_interrupt(self) -> None:
        self.interrupts += 1

    def record_latency(self, ms: float) -> None:
        self.last_latency_ms = ms

    def render(self) -> RenderableType:
        return Group(self._header(), self._transcript_panel(), self._footer())

    def render_ascii(self) -> str:
        """Plain-text rendering for ``--ascii`` mode (screen-reader friendly)."""
        out: list[str] = []
        out.append(f"whisper-agent [mode={self.mode}] [state={self.state.value}]")
        if self.cloud_mode:
            out.append("CLOUD MODE ACTIVE: audio is being sent to a remote backend")
        for line in self.lines:
            stamp = line.when.strftime("%H:%M:%S")
            out.append(f"[{stamp}] {line.speaker}: {line.text}")
        out.append(self._stats_line())
        return "\n".join(out)

    def _header(self) -> RenderableType:
        title = Text("whisper-agent", style="bold")
        title.append(f"  mode={self.mode}", style="dim")
        title.append(f"  state={self.state.value}", style=self._state_style())
        if self.cloud_mode:
            cloud = Text(" CLOUD MODE: audio leaves the machine", style="bold white on red")
            return Group(title, cloud)
        return title

    def _state_style(self) -> str:
        return {
            AgentState.IDLE: "dim",
            AgentState.LISTENING: "cyan",
            AgentState.TRANSCRIBING: "yellow",
            AgentState.THINKING: "magenta",
            AgentState.SPEAKING: "green",
            AgentState.ERROR: "red bold",
        }.get(self.state, "white")

    def _transcript_panel(self) -> RenderableType:
        table = Table(show_header=False, show_edge=False, padding=(0, 1))
        table.add_column("time", style="dim", width=8)
        table.add_column("who", style="bold", width=6)
        table.add_column("text", overflow="fold")
        for line in self.lines:
            who_style = {
                "user": "cyan",
                "agent": "green",
                "system": "yellow",
            }.get(line.speaker, "white")
            table.add_row(
                line.when.strftime("%H:%M:%S"),
                Text(line.speaker, style=who_style),
                line.text,
            )
        return Panel(table, title="transcript", border_style="blue")

    def _footer(self) -> RenderableType:
        return Text(self._stats_line(), style="dim")

    def _stats_line(self) -> str:
        parts = [
            f"tools={self.tool_calls}",
            f"wakes={self.wakes}",
            f"interrupts={self.interrupts}",
        ]
        if self.last_latency_ms is not None:
            parts.append(f"last_latency={self.last_latency_ms:.0f}ms")
        return "  ".join(parts)


class LiveTUI:
    """Convenience wrapper around ``rich.live.Live`` driven by a
    :class:`TranscriptView`."""

    def __init__(self, view: TranscriptView, *, console: Console | None = None) -> None:
        self.view = view
        self._console = console or Console()
        self._live = None

    def __enter__(self) -> LiveTUI:
        from rich.live import Live

        self._live = Live(
            self.view.render(),
            console=self._console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._live is not None:
            self._live.__exit__(*exc)

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self.view.render())

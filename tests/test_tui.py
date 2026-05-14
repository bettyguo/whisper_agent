"""TUI rendering tests."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from whisper_agent.orchestrator import AgentState
from whisper_agent.tui import TranscriptView


def _render(view: TranscriptView) -> str:
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False, color_system=None)
    console.print(view.render())
    return buf.getvalue()


def test_empty_view_renders_header() -> None:
    view = TranscriptView()
    out = _render(view)
    assert "whisper-agent" in out
    assert "mode=push-to-talk" in out
    assert "state=idle" in out


def test_transcript_lines_appear_in_order() -> None:
    view = TranscriptView()
    view.add("user", "Hello")
    view.add("agent", "Hi there")
    out = _render(view)
    user_at = out.find("Hello")
    agent_at = out.find("Hi there")
    assert user_at >= 0
    assert agent_at >= 0
    assert user_at < agent_at


def test_empty_text_does_not_add_line() -> None:
    view = TranscriptView()
    view.add("user", "   ")
    assert not view.lines


def test_state_transitions_persist() -> None:
    view = TranscriptView()
    view.set_state(AgentState.LISTENING)
    out = _render(view)
    assert "state=listening" in out


def test_cloud_mode_banner_shown_only_when_active() -> None:
    view = TranscriptView(cloud_mode=False)
    out = _render(view)
    assert "CLOUD" not in out

    view2 = TranscriptView(cloud_mode=True)
    out2 = _render(view2)
    assert "CLOUD" in out2


def test_record_counters_increment() -> None:
    view = TranscriptView()
    view.record_tool_call()
    view.record_tool_call()
    view.record_wake()
    view.record_interrupt()
    assert view.tool_calls == 2
    assert view.wakes == 1
    assert view.interrupts == 1


def test_stats_line_shows_latency_when_set() -> None:
    view = TranscriptView()
    view.record_latency(1234.0)
    out = _render(view)
    assert "1234" in out


def test_ascii_render_is_plain_text() -> None:
    view = TranscriptView(cloud_mode=True)
    view.add("user", "hi")
    out = view.render_ascii()
    # No ANSI escapes, no Rich tags.
    assert "\x1b" not in out
    assert "[bold]" not in out
    assert "CLOUD MODE" in out
    assert "user: hi" in out


def test_transcript_caps_at_max_len() -> None:
    view = TranscriptView()
    for i in range(250):
        view.add("user", f"line {i}")
    # maxlen=200; oldest 50 dropped.
    assert len(view.lines) == 200
    assert view.lines[0].text == "line 50"
    assert view.lines[-1].text == "line 249"

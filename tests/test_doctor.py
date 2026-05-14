"""Doctor diagnostics: verify offline-mode checks land in the report."""

from __future__ import annotations

from whisper_agent.config import default_config
from whisper_agent.doctor import (
    DoctorReport,
    Finding,
    Severity,
    format_report,
    run_diagnostics,
)


def test_default_config_passes_all_pure_diagnostics() -> None:
    report = run_diagnostics()
    assert report.all_ok
    names = {f.name for f in report.findings}
    assert "version" in names
    assert "offline" in names
    assert "stt.backend" in names
    assert "tts.backend" in names
    assert "llm.base_url" in names


def test_offline_false_is_a_fail() -> None:
    cfg = default_config()
    cfg.offline_only = False
    report = run_diagnostics(cfg)
    offline = next(f for f in report.findings if f.name == "offline")
    assert offline.severity is Severity.FAIL
    assert report.offline_ok is False
    assert report.all_ok is False


def test_llm_config_has_base_url_defaulting_to_localhost() -> None:
    """LLMConfig.base_url must be a real field so users can override
    the Ollama URL via config.toml. Default is localhost."""
    cfg = default_config()
    assert cfg.llm.base_url == "http://localhost:11434"


def test_non_local_llm_url_fails_offline() -> None:
    cfg = default_config()
    cfg.llm.base_url = "http://api.example.com:443"
    report = run_diagnostics(cfg)
    bad = next(f for f in report.findings if f.name == "offline" and f.severity is Severity.FAIL)
    assert "not localhost" in bad.message


def test_non_default_stt_warns() -> None:
    cfg = default_config()
    cfg.stt.backend = "openai"
    report = run_diagnostics(cfg)
    stt = next(f for f in report.findings if f.name == "stt.backend")
    assert stt.severity is Severity.WARN


def test_non_default_tts_warns() -> None:
    cfg = default_config()
    cfg.tts.backend = "elevenlabs"
    report = run_diagnostics(cfg)
    tts = next(f for f in report.findings if f.name == "tts.backend")
    assert tts.severity is Severity.WARN


def test_report_offline_ok_true_when_no_offline_fail() -> None:
    report = DoctorReport(findings=[Finding(severity=Severity.WARN, name="anything", message="x")])
    assert report.offline_ok is True


def test_format_report_ascii_has_no_markup() -> None:
    report = DoctorReport(
        findings=[
            Finding(severity=Severity.OK, name="x", message="all good"),
            Finding(severity=Severity.WARN, name="y", message="hmm"),
            Finding(severity=Severity.FAIL, name="z", message="bad"),
        ]
    )
    out = format_report(report, ascii_only=True)
    assert "[green]" not in out
    assert "[red]" not in out
    assert "OK" in out
    assert "WARN" in out
    assert "FAIL" in out
    # Has a closing summary line.
    assert "Failures present" in out

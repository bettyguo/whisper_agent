"""``whisper-agent doctor`` self-diagnostics.

:func:`run_diagnostics` is pure-Python config-shape checks. They
unit-test without hardware. :func:`run_runtime_checks` touches PATH and
imports (PortAudio, ollama binary), so it lives behind ``--runtime``.
"""

from __future__ import annotations

import importlib
import shutil
from dataclasses import dataclass, field
from enum import Enum

from whisper_agent.config import Config, default_config


class Severity(Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class Finding:
    severity: Severity
    name: str
    message: str


@dataclass
class DoctorReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, severity: Severity, name: str, message: str) -> None:
        self.findings.append(Finding(severity=severity, name=name, message=message))

    @property
    def offline_ok(self) -> bool:
        """True iff no finding mentions cloud mode + no FAILs related to offline."""
        return not any(f.name == "offline" and f.severity is Severity.FAIL for f in self.findings)

    @property
    def all_ok(self) -> bool:
        return not any(f.severity is Severity.FAIL for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(f.severity is Severity.WARN for f in self.findings)


def run_diagnostics(config: Config | None = None) -> DoctorReport:
    """Run pure-data diagnostics that don't touch hardware or network."""
    cfg = config or default_config()
    report = DoctorReport()

    # Version.
    try:
        version = importlib.import_module("whisper_agent").__version__
    except Exception:
        version = "unknown"
    report.add(Severity.OK, "version", f"whisper-agent {version}")

    # Offline-by-default.
    if not cfg.offline_only:
        report.add(
            Severity.FAIL,
            "offline",
            "offline_only=False; cloud backends may be reached.",
        )
    else:
        report.add(Severity.OK, "offline", "offline_only=True")

    # STT backend sanity.
    if cfg.stt.backend != "faster-whisper":
        report.add(
            Severity.WARN,
            "stt.backend",
            f"non-default STT backend in use: {cfg.stt.backend}",
        )
    else:
        report.add(Severity.OK, "stt.backend", "faster-whisper (default)")

    # TTS backend.
    if cfg.tts.backend not in {"kokoro", "f5"}:
        report.add(
            Severity.WARN,
            "tts.backend",
            f"non-default TTS backend in use: {cfg.tts.backend}",
        )
    else:
        report.add(Severity.OK, "tts.backend", f"{cfg.tts.backend} ok")

    # LLM backend host must be localhost.
    base = cfg.llm.base_url
    if "localhost" in base or "127.0.0.1" in base or "::1" in base:
        report.add(Severity.OK, "llm.base_url", f"{base} (local)")
    else:
        report.add(
            Severity.FAIL,
            "offline",
            f"llm.base_url={base!r} is not localhost; transcripts would leave the machine.",
        )

    return report


def run_runtime_checks(config: Config | None = None) -> DoctorReport:
    """Diagnostics that may touch hardware / network. Optional."""
    del config  # reserved for future device-enumeration checks
    report = DoctorReport()

    if shutil.which("ollama") is None:
        report.add(
            Severity.WARN,
            "ollama.binary",
            "`ollama` binary not on PATH; the Ollama backend won't work until it's installed.",
        )
    else:
        report.add(Severity.OK, "ollama.binary", "ollama binary found on PATH")

    try:
        importlib.import_module("sounddevice")
    except OSError:
        report.add(
            Severity.FAIL,
            "audio.portaudio",
            "PortAudio not present; mic + speaker won't initialize. Install via your OS package manager.",
        )
    except ImportError:
        report.add(
            Severity.WARN,
            "audio.sounddevice",
            "sounddevice package not installed; pip install whisper-agent[stt].",
        )
    else:
        report.add(Severity.OK, "audio.sounddevice", "sounddevice + PortAudio available")

    return report


def format_report(report: DoctorReport, *, ascii_only: bool = False) -> str:
    icons = {
        Severity.OK: "OK  " if ascii_only else "[green]OK[/green]  ",
        Severity.WARN: "WARN" if ascii_only else "[yellow]WARN[/yellow]",
        Severity.FAIL: "FAIL" if ascii_only else "[red]FAIL[/red]",
    }
    lines = []
    for f in report.findings:
        lines.append(f"{icons[f.severity]}  {f.name:24s}  {f.message}")
    if report.all_ok and not report.has_warnings:
        lines.append("\nAll checks passed.")
    elif report.all_ok:
        lines.append("\nNo failures; review warnings above.")
    else:
        lines.append("\nFailures present; fix before relying on the offline guarantee.")
    return "\n".join(lines)

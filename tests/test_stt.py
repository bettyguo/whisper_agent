"""STT base + faster-whisper config tests.

The actual faster-whisper model load + inference is gated by the `model`
marker and skipped in CI.
"""

from __future__ import annotations

import platform

import pytest

from whisper_agent.stt import STTResult, Transcriber
from whisper_agent.stt.base import STTSegment
from whisper_agent.stt.whisper import WhisperConfig, pick_compute_type, pick_device


def test_stt_result_defaults() -> None:
    r = STTResult(text="hello")
    assert r.text == "hello"
    assert r.language is None
    assert r.segments == ()
    assert r.duration_s == 0.0


def test_stt_segment_immutable() -> None:
    s = STTSegment(start_s=0.0, end_s=1.0, text="hi")
    with pytest.raises(Exception):  # noqa: B017
        s.text = "bye"  # type: ignore[misc]


def test_pick_compute_type_explicit() -> None:
    assert pick_compute_type("float16") == "float16"
    assert pick_compute_type("int8") == "int8"


def test_pick_compute_type_auto_on_arm64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert pick_compute_type("auto") == "int8_float16"


def test_pick_compute_type_auto_on_x86(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert pick_compute_type("auto") == "int8"


def test_pick_device_explicit() -> None:
    assert pick_device("cpu") == "cpu"
    assert pick_device("cuda") == "cuda"


def test_pick_device_auto_falls_back_to_cpu_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not installed in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert pick_device("auto") == "cpu"


def test_whisper_config_defaults_match_adr_006() -> None:
    cfg = WhisperConfig()
    assert cfg.model == "large-v3"
    assert cfg.compute_type == "auto"
    assert cfg.device == "auto"
    assert cfg.vad_filter is False  # we run our own VAD
    assert cfg.language == "en"


def test_transcriber_is_a_protocol() -> None:
    """Ensure ``Transcriber`` is structural: anything with the right
    method shape satisfies it."""

    class _Fake:
        def transcribe(self, audio, *, sample_rate=16_000, language="en"):
            return STTResult(text="ok")

    inst: Transcriber = _Fake()  # would fail at type-check time if wrong
    out = inst.transcribe(None)  # type: ignore[arg-type]
    assert out.text == "ok"


@pytest.mark.model
def test_faster_whisper_real_transcribe_tiny() -> None:
    """Smoke test against the real model. Skipped in CI; run locally with `-m model`."""
    pytest.skip("requires faster-whisper model; run locally with -m model")

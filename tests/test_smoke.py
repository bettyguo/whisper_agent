"""Smoke tests: package import, CLI loads, defaults are offline."""

from __future__ import annotations


def test_package_imports() -> None:
    import whisper_agent

    assert whisper_agent.__version__


def test_cli_app_exists() -> None:
    from whisper_agent.cli import app

    assert app is not None


def test_default_config_offline_only() -> None:
    from whisper_agent.config import default_config

    cfg = default_config()
    assert cfg.offline_only is True
    assert cfg.stt.backend == "faster-whisper"
    assert cfg.tts.backend in {"kokoro", "f5"}

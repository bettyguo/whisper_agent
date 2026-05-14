"""Runtime configuration.

Loaded from ``~/.config/whisper-agent/config.toml`` (or
``%APPDATA%/whisper-agent/config.toml`` on Windows) with env-var
overrides. The loader itself is not wired up yet; the defaults below
are the ones the package ships with.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AudioConfig(BaseModel):
    sample_rate: int = 16_000  # silero-vad + faster-whisper friendly
    channels: int = 1
    block_ms: int = 30  # silero-vad expects 10/20/30 ms blocks


class STTConfig(BaseModel):
    backend: str = "faster-whisper"
    model: str = "large-v3"
    language: str = "en"
    compute_type: str = "auto"  # int8 / float16 / auto, resolved at load time


class TTSConfig(BaseModel):
    backend: str = "kokoro"
    voice: str = "default"
    speed: float = 1.0


class LLMConfig(BaseModel):
    backend: str = "ollama"  # also: "llamacpp", "vllm"
    model: str = "qwen2.5:7b-instruct"
    base_url: str = "http://localhost:11434"


class WakeConfig(BaseModel):
    enabled: bool = False
    phrase: str = "hey computer"
    sensitivity: float = 0.5


class Config(BaseModel):
    audio: AudioConfig = Field(default_factory=AudioConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    wake: WakeConfig = Field(default_factory=WakeConfig)
    offline_only: bool = True


def default_config() -> Config:
    return Config()

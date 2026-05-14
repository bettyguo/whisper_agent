"""faster-whisper STT backend.

Lazy-imports ``faster_whisper`` so the package is importable on machines
without it. Construction loads the model; keep instances long-lived.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from typing import TYPE_CHECKING

from whisper_agent.stt.base import STTResult, STTSegment

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    Float32 = NDArray[np.float32]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WhisperConfig:
    """faster-whisper config."""

    model: str = "large-v3"
    compute_type: str = "auto"  # "int8" / "int8_float16" / "float16" / "auto"
    device: str = "auto"  # "cpu" / "cuda" / "auto"
    beam_size: int = 1
    vad_filter: bool = False  # whisper-agent runs its own VAD upstream
    language: str = "en"


def pick_compute_type(config_value: str) -> str:
    """Resolve ``compute_type='auto'`` based on host hardware."""
    if config_value != "auto":
        return config_value
    machine = platform.machine().lower()
    if "arm64" in machine or "aarch64" in machine:
        return "int8_float16"
    return "int8"


def pick_device(config_value: str) -> str:
    """Resolve ``device='auto'``: prefer CUDA, fall back to CPU."""
    if config_value != "auto":
        return config_value
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class FasterWhisperTranscriber:
    """faster-whisper backend.

    Methods:
        :meth:`transcribe` is synchronous; takes float32 mono audio at 16 kHz.

    The expensive resource is the model itself. Load once and reuse.
    """

    def __init__(self, config: WhisperConfig | None = None) -> None:
        self.config = config or WhisperConfig()
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "faster-whisper isn't installed. Run `pip install whisper-agent[stt]`."
            ) from e

        compute_type = pick_compute_type(self.config.compute_type)
        device = pick_device(self.config.device)
        log.info(
            "loading faster-whisper model=%s device=%s compute=%s",
            self.config.model,
            device,
            compute_type,
        )
        self._model = WhisperModel(
            self.config.model,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(
        self,
        audio: Float32,
        *,
        sample_rate: int = 16_000,
        language: str | None = None,
    ) -> STTResult:
        if sample_rate != 16_000:
            raise ValueError("faster-whisper expects 16 kHz mono audio")
        segments_iter, info = self._model.transcribe(
            audio,
            beam_size=self.config.beam_size,
            language=language or self.config.language,
            vad_filter=self.config.vad_filter,
        )
        segments: list[STTSegment] = []
        text_parts: list[str] = []
        for seg in segments_iter:
            segments.append(
                STTSegment(
                    start_s=float(seg.start),
                    end_s=float(seg.end),
                    text=seg.text.strip(),
                    confidence=getattr(seg, "avg_logprob", None),
                )
            )
            text_parts.append(seg.text)
        return STTResult(
            text="".join(text_parts).strip(),
            language=info.language,
            segments=tuple(segments),
            duration_s=float(info.duration),
        )

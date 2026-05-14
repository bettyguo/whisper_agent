"""STT transcriber protocol + result types.

Every backend (faster-whisper, distil-whisper, hypothetical others) exposes
the same surface: take a 16 kHz mono float32 array, return text plus
optional segment-level metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    Float32 = NDArray[np.float32]


@dataclass(frozen=True)
class STTSegment:
    """One transcribed segment with timing info."""

    start_s: float
    end_s: float
    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class STTResult:
    """A complete transcription result."""

    text: str
    language: str | None = None
    segments: tuple[STTSegment, ...] = field(default_factory=tuple)
    duration_s: float = 0.0


class Transcriber(Protocol):
    """A loaded STT backend ready to transcribe samples."""

    def transcribe(
        self,
        audio: Float32,
        *,
        sample_rate: int = 16_000,
        language: str | None = "en",
    ) -> STTResult: ...

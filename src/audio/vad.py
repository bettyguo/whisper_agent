"""Voice activity detection: silero-vad wrapper + frame-level state machine.

:class:`VADStateMachine` is pure logic, testable without the silero
model. :class:`SileroVAD` loads the ONNX model lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    from numpy.typing import NDArray

    Float32 = NDArray[np.float32]


@dataclass(frozen=True)
class VADParams:
    """Tunables for the silero VAD state machine."""

    sample_rate: int = 16_000
    threshold: float = 0.5  # silero confidence above which a frame is "speech"
    min_silence_ms: int = 250  # silence required before speech_end
    min_speech_ms: int = 120  # speech required before speech_start
    pre_pad_ms: int = 200  # samples kept *before* speech_start in the captured segment
    frame_ms: int = 30  # silero accepts 10/20/30 ms frames at 16 kHz

    @property
    def frame_samples(self) -> int:
        return self.sample_rate * self.frame_ms // 1000

    @property
    def min_silence_frames(self) -> int:
        return max(1, self.min_silence_ms // self.frame_ms)

    @property
    def min_speech_frames(self) -> int:
        return max(1, self.min_speech_ms // self.frame_ms)


class VADEvent(Enum):
    NONE = "none"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


class VADStateMachine:
    """Convert per-frame voiced/unvoiced into start/end events.

    State:
      - ``in_speech``: are we currently inside an utterance?
      - ``consecutive_voiced``: voiced frames since last unvoiced (for entering speech)
      - ``consecutive_silent``: unvoiced frames since last voiced (for exiting speech)
    """

    __slots__ = ("_silent_streak", "_voiced_streak", "in_speech", "params")

    def __init__(self, params: VADParams | None = None) -> None:
        self.params = params or VADParams()
        self.in_speech: bool = False
        self._voiced_streak: int = 0
        self._silent_streak: int = 0

    def reset(self) -> None:
        self.in_speech = False
        self._voiced_streak = 0
        self._silent_streak = 0

    def step(self, voiced: bool) -> VADEvent:
        """Advance one frame; return any event triggered.

        Edge semantics:
          - ``SPEECH_START`` fires the moment ``min_speech_frames`` of
            consecutive voiced frames is reached after a non-speech state.
          - ``SPEECH_END`` fires the moment ``min_silence_frames`` of
            consecutive silent frames is reached after a speech state.
        """
        params = self.params
        if voiced:
            self._silent_streak = 0
            self._voiced_streak += 1
            if not self.in_speech and self._voiced_streak >= params.min_speech_frames:
                self.in_speech = True
                return VADEvent.SPEECH_START
            return VADEvent.NONE
        else:
            self._voiced_streak = 0
            self._silent_streak += 1
            if self.in_speech and self._silent_streak >= params.min_silence_frames:
                self.in_speech = False
                return VADEvent.SPEECH_END
            return VADEvent.NONE

    def feed_voicing(self, voicing: Iterable[bool]) -> list[tuple[int, VADEvent]]:
        """Convenience: feed an iterable of per-frame booleans; return events.

        Used by tests that pre-compute voiced flags and want to assert on
        event sequences without going through the silero model.
        """
        events: list[tuple[int, VADEvent]] = []
        for i, voiced in enumerate(voicing):
            ev = self.step(voiced)
            if ev is not VADEvent.NONE:
                events.append((i, ev))
        return events


class SileroVAD:
    """Frame-level "is voiced?" classifier backed by silero-vad ONNX.

    Lazy-loaded: importing this module does not import torch / onnxruntime.
    Construction does; keep instances long-lived.
    """

    def __init__(self, params: VADParams | None = None) -> None:
        self.params = params or VADParams()
        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self._model = model
            (self._get_speech_timestamps, _, _, _, _) = utils
        except Exception as e:
            raise RuntimeError(
                "silero-vad failed to load. Install with `pip install whisper-agent[stt]` "
                "and ensure torch is available."
            ) from e

    def is_voiced(self, frame: Float32) -> bool:
        """Return True if ``frame`` (one VAD frame) is voiced."""
        import torch

        if frame.shape[0] != self.params.frame_samples:
            raise ValueError(
                f"frame size mismatch: expected {self.params.frame_samples}, got {frame.shape[0]}"
            )
        tensor = torch.from_numpy(frame)
        with torch.no_grad():
            prob = float(self._model(tensor, self.params.sample_rate).item())
        return prob >= self.params.threshold

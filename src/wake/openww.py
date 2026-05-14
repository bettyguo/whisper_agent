"""openWakeWord wrapper + cooldown state machine.

:class:`WakeStateMachine` is pure logic: per-frame confidence scores
in, ``FIRE`` events out when the phrase crosses threshold and we are
outside the cooldown window. Testable without the model.

:class:`OpenWakeWordDetector` lazy-loads openWakeWord.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import numpy as np
    from numpy.typing import NDArray

    Int16 = NDArray[np.int16]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WakeParams:
    """Wake-word tunables."""

    phrase: str = "hey_computer"  # bundled openWakeWord model name
    sensitivity: float = 0.5  # higher = more permissive
    cooldown_ms: int = 500  # ignore re-triggers within this window
    sample_rate: int = 16_000
    frame_samples: int = 1_280  # openWakeWord's expected ~80 ms at 16 kHz


class WakeEvent(Enum):
    NONE = "none"
    FIRE = "fire"  # wake phrase detected; orchestrator should transition IDLE -> LISTENING


class WakeStateMachine:
    """Turns per-frame confidence scores into FIRE events with cooldown.

    The state machine is monotonic in time: it remembers the last fire
    timestamp and refuses to fire again within ``cooldown_ms``.
    """

    __slots__ = ("_last_fire_s", "params")

    def __init__(self, params: WakeParams | None = None) -> None:
        self.params = params or WakeParams()
        self._last_fire_s: float = -1e9  # effectively "never fired"

    def reset(self) -> None:
        self._last_fire_s = -1e9

    def step(self, confidence: float, *, now_s: float | None = None) -> WakeEvent:
        """Advance one frame. Returns ``FIRE`` if and only if the wake
        threshold is crossed AND we are outside the cooldown window."""
        if confidence < self.params.sensitivity:
            return WakeEvent.NONE
        ts = time.monotonic() if now_s is None else now_s
        if (ts - self._last_fire_s) * 1000.0 < self.params.cooldown_ms:
            return WakeEvent.NONE
        self._last_fire_s = ts
        return WakeEvent.FIRE

    def feed_confidences(
        self, scores: Iterable[float], *, start_s: float = 0.0, step_s: float = 0.08
    ) -> list[tuple[int, WakeEvent]]:
        """Convenience for tests. Feed a sequence of confidences with a
        synthetic clock and collect events."""
        events: list[tuple[int, WakeEvent]] = []
        for i, score in enumerate(scores):
            ev = self.step(score, now_s=start_s + i * step_s)
            if ev is not WakeEvent.NONE:
                events.append((i, ev))
        return events


class OpenWakeWordDetector:
    """Frame-level wake-word classifier backed by openWakeWord ONNX models.

    openWakeWord operates on int16 PCM frames at 16 kHz. We convert
    float32 mono audio internally.

    Lazy-loaded: import + construction pull torch / onnxruntime; tests
    that don't need the model should NOT instantiate this class.
    """

    def __init__(self, params: WakeParams | None = None) -> None:
        self.params = params or WakeParams()
        try:
            from openwakeword.model import Model
        except ImportError as e:
            raise RuntimeError(
                "openWakeWord isn't installed. Run `pip install whisper-agent[wake]`."
            ) from e
        # openWakeWord ships several pretrained models; we load only the
        # phrase the user configured.
        self._model = Model(wakeword_models=[self.params.phrase])
        log.info("openWakeWord loaded for phrase=%s", self.params.phrase)

    def predict_frame(self, frame: Int16) -> float:
        """Return max confidence across the configured phrase model."""
        scores: Mapping[str, float] = self._model.predict(frame)
        # scores is a dict like {"hey_computer": 0.91}; we return the max.
        return max(scores.values()) if scores else 0.0

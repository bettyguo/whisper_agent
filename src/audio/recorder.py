"""Utterance recorder: contiguous audio between two events.

Designed for push-to-talk. Press starts a session, release plus VAD's
``speech_end`` finalizes it. Reads from the shared ``RingBuffer`` via a
cursor and snapshots a configurable ``pre_pad_ms`` before the trigger
so the leading edge of speech isn't clipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from whisper_agent.audio.ring_buffer import RingBuffer
from whisper_agent.audio.vad import VADParams

Float32 = NDArray[np.float32]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecorderConfig:
    sample_rate: int = 16_000
    pre_pad_ms: int = 200  # capture this much audio before the trigger
    max_duration_s: float = 30.0


class UtteranceRecorder:
    """Reads contiguous samples from a RingBuffer between start/stop.

    Behavior:
      * ``start()`` snapshots the buffer's ``total_written`` minus the
        configured pre-pad, so we don't miss the leading edge of speech.
      * ``stop()`` reads from the snapshot cursor up to the current
        ``total_written`` and returns one float32 array.
      * ``stop()`` clamps to ``max_duration_s`` and warns if it triggers.
    """

    def __init__(
        self,
        buffer: RingBuffer,
        config: RecorderConfig | None = None,
    ) -> None:
        self.buffer = buffer
        self.config = config or RecorderConfig()
        self._cursor: int | None = None

    @classmethod
    def from_vad_params(cls, buffer: RingBuffer, vad: VADParams) -> UtteranceRecorder:
        """Build a recorder whose pre-pad matches the VAD's pre-pad."""
        return cls(
            buffer,
            RecorderConfig(
                sample_rate=vad.sample_rate,
                pre_pad_ms=vad.pre_pad_ms,
            ),
        )

    @property
    def is_active(self) -> bool:
        return self._cursor is not None

    def start(self) -> None:
        if self._cursor is not None:
            log.warning("recorder.start() called while active; ignoring")
            return
        pre_pad_samples = self.config.sample_rate * self.config.pre_pad_ms // 1000
        # Snapshot cursor; read_from clamps if pre_pad is older than the buffer.
        start_cursor = max(0, self.buffer.total_written - pre_pad_samples)
        self._cursor = start_cursor

    def stop(self) -> Float32:
        if self._cursor is None:
            return np.zeros(0, dtype=np.float32)
        max_samples = int(self.config.sample_rate * self.config.max_duration_s)
        samples, _ = self.buffer.read_from(self._cursor, max_samples=max_samples)
        if samples.size >= max_samples:
            log.warning(
                "recorder hit max_duration_s=%.1f; truncating utterance",
                self.config.max_duration_s,
            )
        self._cursor = None
        return samples

    def cancel(self) -> None:
        """Drop the current recording without returning samples."""
        self._cursor = None

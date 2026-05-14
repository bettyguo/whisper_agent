"""Microphone capture into a RingBuffer via sounddevice.

The mic stream is a real-time callback thread owned by PortAudio. It
writes to the shared :class:`RingBuffer`; the asyncio side reads frames
via cursor or "latest N" reads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from whisper_agent.audio.ring_buffer import RingBuffer

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MicConfig:
    sample_rate: int = 16_000
    channels: int = 1  # mono; STT + VAD want mono
    block_size: int = 480  # 30 ms at 16 kHz
    device: int | str | None = None  # None = system default
    buffer_seconds: float = 30.0


class MicStream:
    """Wraps a sounddevice InputStream + a RingBuffer.

    ``sounddevice`` is imported lazily so that core unit tests do not
    require PortAudio to be installed.
    """

    def __init__(
        self,
        config: MicConfig | None = None,
        *,
        on_block: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.config = config or MicConfig()
        self.buffer = RingBuffer(int(self.config.sample_rate * self.config.buffer_seconds))
        self._on_block = on_block
        self._stream = None  # populated on start()

    def start(self) -> None:
        try:
            import sounddevice as sd
        except OSError as e:
            raise RuntimeError(
                "Mic capture needs PortAudio + the `sounddevice` package. "
                "Install with `pip install whisper-agent[stt]`."
            ) from e

        def _callback(indata, frames, time_info, status):
            if status:
                log.warning("mic callback status: %s", status)
            # indata shape: (frames, channels); we collapse to mono float32.
            if indata.ndim == 2 and indata.shape[1] > 1:
                samples = indata.mean(axis=1).astype(np.float32, copy=False)
            else:
                samples = np.ascontiguousarray(indata.reshape(-1), dtype=np.float32)
            self.buffer.write(samples)
            if self._on_block is not None:
                try:
                    self._on_block(samples)
                except Exception:
                    log.exception("mic on_block callback raised; continuing")

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="float32",
            blocksize=self.config.block_size,
            device=self.config.device,
            callback=_callback,
        )
        self._stream.start()
        log.info(
            "mic started: sr=%d ch=%d block=%d device=%s",
            self.config.sample_rate,
            self.config.channels,
            self.config.block_size,
            self.config.device,
        )

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> MicStream:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

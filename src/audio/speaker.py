"""Speaker output with mid-stream interrupt support.

Plays float32 mono PCM chunks from a thread-safe queue. The producer is
the asyncio orchestrator; the consumer is the PortAudio callback
thread. ``interrupt()`` drops pending chunks and stops the current
one; the orchestrator calls it on VAD speech_start during a reply.
"""

from __future__ import annotations

import asyncio
import logging
import queue as _q
import threading
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Float32 = NDArray[np.float32]

log = logging.getLogger(__name__)

_EOS = object()  # sentinel for end-of-stream


@dataclass(frozen=True)
class SpeakerConfig:
    sample_rate: int = 24_000  # Kokoro default; matches TTS chunker default
    channels: int = 1
    block_size: int = 1024
    device: int | str | None = None
    queue_size: int = 64


class Speaker:
    """Plays queued float32 chunks via sounddevice; supports interrupt."""

    def __init__(self, config: SpeakerConfig | None = None) -> None:
        self.config = config or SpeakerConfig()
        self._queue: _q.Queue = _q.Queue(maxsize=self.config.queue_size)
        self._stream = None
        self._current_chunk: Float32 | None = None
        self._chunk_pos: int = 0
        self._drain_event = threading.Event()
        self._drain_event.set()  # drained by default
        self._lock = threading.Lock()

    def start(self) -> None:
        try:
            import sounddevice as sd
        except OSError as e:
            raise RuntimeError("Speaker output needs PortAudio + the `sounddevice` package.") from e

        def _callback(outdata, frames, time_info, status):
            if status:
                log.warning("speaker callback status: %s", status)
            self._fill(outdata, frames)

        self._stream = sd.OutputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="float32",
            blocksize=self.config.block_size,
            device=self.config.device,
            callback=_callback,
        )
        self._stream.start()
        log.info("speaker started: sr=%d ch=%d", self.config.sample_rate, self.config.channels)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def play(self, chunk: Float32) -> None:
        """Queue ``chunk`` for playback. Blocks the producer loop if the queue is full."""
        if chunk.size == 0:
            return
        contig = np.ascontiguousarray(chunk, dtype=np.float32)
        self._drain_event.clear()
        # Don't block the asyncio loop on a full queue; push via executor.
        await asyncio.to_thread(self._queue.put, contig)

    async def flush_eos(self) -> None:
        """Mark end-of-stream so the caller can ``await drain()``."""
        await asyncio.to_thread(self._queue.put, _EOS)

    async def drain(self) -> None:
        """Wait until queued audio has played out."""
        await asyncio.to_thread(self._drain_event.wait)

    def interrupt(self) -> None:
        """Drop pending chunks; stop the currently playing chunk.

        Safe to call from any thread.
        """
        with self._lock:
            self._current_chunk = None
            self._chunk_pos = 0
        # Drain pending chunks.
        while True:
            try:
                self._queue.get_nowait()
            except _q.Empty:
                break
        self._drain_event.set()

    def _fill(self, outdata: np.ndarray, frames: int) -> None:
        out_pos = 0
        outdata.fill(0.0)
        with self._lock:
            while out_pos < frames:
                if self._current_chunk is None:
                    try:
                        item = self._queue.get_nowait()
                    except _q.Empty:
                        if self._queue.empty():
                            self._drain_event.set()
                        return
                    if item is _EOS:
                        self._drain_event.set()
                        return
                    self._current_chunk = item
                    self._chunk_pos = 0
                remaining_chunk = self._current_chunk.shape[0] - self._chunk_pos
                remaining_out = frames - out_pos
                take = min(remaining_chunk, remaining_out)
                outdata[out_pos : out_pos + take, 0] = self._current_chunk[
                    self._chunk_pos : self._chunk_pos + take
                ]
                self._chunk_pos += take
                out_pos += take
                if self._chunk_pos >= self._current_chunk.shape[0]:
                    self._current_chunk = None
                    self._chunk_pos = 0

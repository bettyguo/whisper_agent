"""Lock-protected float32 ring buffer for raw mic samples.

A sounddevice callback writes here from a real-time thread; the asyncio
side reads frames for VAD and replay. When full, the buffer overwrites
the oldest samples: capture is always the most recent ``capacity``
samples.
"""

from __future__ import annotations

import threading

import numpy as np
from numpy.typing import NDArray

Float32 = NDArray[np.float32]


class RingBuffer:
    """Fixed-capacity mono float32 ring buffer.

    Single contiguous ``np.ndarray`` + a write head + a "samples written
    so far" counter + a lock. The lock cost is fine for the throughput
    we run (~16 kHz x 30 ms = ~480 floats per push); a lock-free SPSC
    queue would be more code without a measured win.
    """

    __slots__ = ("_buf", "_capacity", "_lock", "_total_written", "_write")

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self._buf: Float32 = np.zeros(capacity_samples, dtype=np.float32)
        self._capacity = capacity_samples
        self._write = 0  # next write index
        self._total_written = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_written(self) -> int:
        """Monotonic count of samples written. Useful for cursor reads."""
        with self._lock:
            return self._total_written

    @property
    def available(self) -> int:
        """Samples currently readable (== min(total_written, capacity))."""
        with self._lock:
            return min(self._total_written, self._capacity)

    def write(self, samples: Float32) -> None:
        """Append ``samples`` to the buffer, overwriting oldest data if full.

        ``samples`` must be a 1-D float32 array. Writes longer than the
        buffer keep only the trailing ``capacity`` samples.
        """
        if samples.ndim != 1:
            raise ValueError(f"expected 1-D samples, got shape {samples.shape}")
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)

        n_original = samples.shape[0]
        if n_original == 0:
            return

        # Drop samples that would be overwritten anyway in the same call.
        if n_original > self._capacity:
            samples = samples[-self._capacity :]
        n = samples.shape[0]

        with self._lock:
            end = self._write + n
            if end <= self._capacity:
                self._buf[self._write : end] = samples
            else:
                first = self._capacity - self._write
                self._buf[self._write :] = samples[:first]
                self._buf[: end - self._capacity] = samples[first:]
            self._write = end % self._capacity
            # total_written counts presented samples even when we drop the head;
            # this preserves cursor-monotonicity for callers tracking gaps.
            self._total_written += n_original

    def read_latest(self, n: int) -> Float32:
        """Copy the most recent ``n`` samples.

        Returns an array shorter than ``n`` if the buffer hasn't been
        filled yet.
        """
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        with self._lock:
            available = min(self._total_written, self._capacity)
            n = min(n, available)
            if n == 0:
                return np.zeros(0, dtype=np.float32)
            start = (self._write - n) % self._capacity
            end = start + n
            if end <= self._capacity:
                return self._buf[start:end].copy()
            first = self._capacity - start
            out = np.empty(n, dtype=np.float32)
            out[:first] = self._buf[start:]
            out[first:] = self._buf[: n - first]
            return out

    def read_from(self, cursor: int, max_samples: int | None = None) -> tuple[Float32, int]:
        """Copy samples written since ``cursor`` (a value of ``total_written``).

        Returns ``(samples, new_cursor)``. If the cursor is older than
        the oldest sample still in the buffer, only the buffer's
        available contents are returned and ``new_cursor`` jumps forward
        to the oldest retrievable position. Callers detect the gap by
        comparing ``new_cursor - cursor`` against ``len(samples)``.
        """
        with self._lock:
            total = self._total_written
            if cursor < 0:
                cursor = 0
            available = total - cursor
            if available <= 0:
                return np.zeros(0, dtype=np.float32), total

            in_buffer = min(self._capacity, total)
            # Drop samples we no longer have.
            drop = max(0, available - in_buffer)
            effective_cursor = cursor + drop
            n = total - effective_cursor
            if max_samples is not None:
                n = min(n, max_samples)
            if n <= 0:
                return np.zeros(0, dtype=np.float32), total

            # Position of "first sample we want" within the linear ring:
            start_pos = self._write - (total - effective_cursor)
            start = start_pos % self._capacity
            end = start + n
            if end <= self._capacity:
                out = self._buf[start:end].copy()
            else:
                first = self._capacity - start
                out = np.empty(n, dtype=np.float32)
                out[:first] = self._buf[start:]
                out[first:] = self._buf[: n - first]
            return out, effective_cursor + n

    def clear(self) -> None:
        with self._lock:
            self._buf.fill(0.0)
            self._write = 0
            self._total_written = 0

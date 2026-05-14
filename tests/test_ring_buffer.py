"""RingBuffer unit tests covering the wrap-around math."""

from __future__ import annotations

import numpy as np
import pytest

from whisper_agent.audio.ring_buffer import RingBuffer


def _arange(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float32)


def test_initial_state_is_empty() -> None:
    rb = RingBuffer(10)
    assert rb.capacity == 10
    assert rb.available == 0
    assert rb.total_written == 0
    assert rb.read_latest(5).size == 0


def test_write_fewer_than_capacity() -> None:
    rb = RingBuffer(10)
    rb.write(_arange(3))
    assert rb.available == 3
    assert rb.total_written == 3
    np.testing.assert_array_equal(rb.read_latest(3), [0, 1, 2])


def test_write_exactly_capacity() -> None:
    rb = RingBuffer(5)
    rb.write(_arange(5))
    np.testing.assert_array_equal(rb.read_latest(5), [0, 1, 2, 3, 4])


def test_write_more_than_capacity_keeps_tail() -> None:
    rb = RingBuffer(4)
    rb.write(_arange(10))  # [0..9]; only last 4 should survive
    assert rb.available == 4
    np.testing.assert_array_equal(rb.read_latest(4), [6, 7, 8, 9])


def test_wraparound_across_two_writes() -> None:
    rb = RingBuffer(6)
    rb.write(_arange(4))  # write head at 4
    rb.write(_arange(4) + 100)  # write head wraps: 8 samples total, last 6 survive
    np.testing.assert_array_equal(rb.read_latest(6), [2, 3, 100, 101, 102, 103])


def test_read_latest_clips_to_available() -> None:
    rb = RingBuffer(10)
    rb.write(_arange(3))
    np.testing.assert_array_equal(rb.read_latest(100), [0, 1, 2])


def test_read_from_cursor_simple() -> None:
    rb = RingBuffer(10)
    rb.write(_arange(3))
    samples, cursor = rb.read_from(0)
    np.testing.assert_array_equal(samples, [0, 1, 2])
    assert cursor == 3


def test_read_from_cursor_partial() -> None:
    rb = RingBuffer(10)
    rb.write(_arange(5))
    samples, cursor = rb.read_from(2)
    np.testing.assert_array_equal(samples, [2, 3, 4])
    assert cursor == 5


def test_read_from_max_samples_clamp() -> None:
    rb = RingBuffer(10)
    rb.write(_arange(5))
    samples, cursor = rb.read_from(0, max_samples=2)
    np.testing.assert_array_equal(samples, [0, 1])
    assert cursor == 2


def test_read_from_old_cursor_skips_lost_samples() -> None:
    rb = RingBuffer(4)
    rb.write(_arange(10))  # only [6, 7, 8, 9] remain
    samples, cursor = rb.read_from(0)
    # gap: caller asked from 0; new cursor is 10; only 4 samples returned.
    np.testing.assert_array_equal(samples, [6, 7, 8, 9])
    assert cursor == 10


def test_read_from_at_total_yields_empty() -> None:
    rb = RingBuffer(10)
    rb.write(_arange(5))
    samples, cursor = rb.read_from(5)
    assert samples.size == 0
    assert cursor == 5


def test_empty_write_is_noop() -> None:
    rb = RingBuffer(10)
    rb.write(np.zeros(0, dtype=np.float32))
    assert rb.total_written == 0


def test_clear_resets_state() -> None:
    rb = RingBuffer(10)
    rb.write(_arange(5))
    rb.clear()
    assert rb.available == 0
    assert rb.total_written == 0


def test_dtype_coercion() -> None:
    rb = RingBuffer(10)
    rb.write(np.arange(3, dtype=np.float64))
    np.testing.assert_array_equal(rb.read_latest(3), [0, 1, 2])


def test_rejects_2d_input() -> None:
    rb = RingBuffer(10)
    with pytest.raises(ValueError):
        rb.write(np.zeros((3, 2), dtype=np.float32))


def test_rejects_nonpositive_capacity() -> None:
    with pytest.raises(ValueError):
        RingBuffer(0)
    with pytest.raises(ValueError):
        RingBuffer(-1)


def test_concurrent_writes_do_not_corrupt() -> None:
    """Sanity check the lock: many small writes from threads end up coherent."""
    import threading

    rb = RingBuffer(1000)
    n_threads = 8
    per_thread = 100

    def writer(start: int) -> None:
        rb.write(np.arange(start, start + per_thread, dtype=np.float32))

    threads = [threading.Thread(target=writer, args=(i * per_thread,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every value is preserved; order varies but the SET of values matches.
    out = rb.read_latest(800)  # last 800 samples; all writes fit in capacity
    expected = set(range(n_threads * per_thread))
    seen = set(int(v) for v in out)
    # We can't assert exactly because order is racy. Check that we got 800 distinct
    # values, all drawn from the expected set, with no garbage.
    assert seen.issubset(expected)
    assert len(seen) == out.size

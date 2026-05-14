"""UtteranceRecorder tests."""

from __future__ import annotations

import numpy as np
import pytest

from whisper_agent.audio.recorder import RecorderConfig, UtteranceRecorder
from whisper_agent.audio.ring_buffer import RingBuffer


def _samples(n: int, start: int = 0) -> np.ndarray:
    return np.arange(start, start + n, dtype=np.float32)


@pytest.fixture
def rb() -> RingBuffer:
    # Big enough to hold a few seconds at 16 kHz.
    return RingBuffer(16_000 * 5)


@pytest.fixture
def recorder(rb: RingBuffer) -> UtteranceRecorder:
    return UtteranceRecorder(
        rb,
        RecorderConfig(sample_rate=16_000, pre_pad_ms=0, max_duration_s=5.0),
    )


def test_inactive_recorder_returns_empty(recorder: UtteranceRecorder) -> None:
    assert recorder.is_active is False
    out = recorder.stop()
    assert out.size == 0


def test_records_audio_between_start_and_stop(rb: RingBuffer, recorder: UtteranceRecorder) -> None:
    rb.write(_samples(100))  # pre-existing samples; should be ignored when pre_pad=0
    recorder.start()
    rb.write(_samples(200, start=100))
    out = recorder.stop()
    np.testing.assert_array_equal(out, _samples(200, start=100))


def test_pre_pad_includes_earlier_samples(rb: RingBuffer) -> None:
    rec = UtteranceRecorder(
        rb,
        RecorderConfig(sample_rate=16_000, pre_pad_ms=10, max_duration_s=5.0),
    )
    # 10 ms at 16 kHz = 160 samples
    rb.write(_samples(500))
    rec.start()
    rb.write(_samples(100, start=500))
    out = rec.stop()
    # We should get the trailing 160 of the first write + all 100 of the second.
    assert out.size == 160 + 100
    np.testing.assert_array_equal(out[:160], _samples(160, start=340))
    np.testing.assert_array_equal(out[160:], _samples(100, start=500))


def test_max_duration_truncates(rb: RingBuffer) -> None:
    rec = UtteranceRecorder(
        rb,
        RecorderConfig(sample_rate=16_000, pre_pad_ms=0, max_duration_s=0.01),
    )
    # max_duration_s=0.01 at 16 kHz => 160 samples
    rec.start()
    rb.write(_samples(1000))
    out = rec.stop()
    assert out.size == 160


def test_cancel_drops_recording(rb: RingBuffer, recorder: UtteranceRecorder) -> None:
    recorder.start()
    rb.write(_samples(100))
    recorder.cancel()
    assert recorder.is_active is False
    # A second stop yields no samples even though there's data.
    out = recorder.stop()
    assert out.size == 0


def test_double_start_is_idempotent(rb: RingBuffer, recorder: UtteranceRecorder) -> None:
    recorder.start()
    rb.write(_samples(50))
    recorder.start()  # second start should be a no-op
    rb.write(_samples(50, start=50))
    out = recorder.stop()
    np.testing.assert_array_equal(out, _samples(100))


def test_pre_pad_clamps_to_buffer_start(rb: RingBuffer) -> None:
    """pre_pad longer than what's in the buffer just starts at zero."""
    rec = UtteranceRecorder(
        rb,
        RecorderConfig(sample_rate=16_000, pre_pad_ms=1000, max_duration_s=5.0),
    )
    rb.write(_samples(100))
    rec.start()
    rb.write(_samples(50, start=100))
    out = rec.stop()
    # We had only 100 + 50 = 150 samples total; pre_pad clamps to 0.
    assert out.size == 150

"""VADStateMachine unit tests (pure logic, no model)."""

from __future__ import annotations

import pytest

from whisper_agent.audio.vad import VADEvent, VADParams, VADStateMachine


def _voicing(pattern: str) -> list[bool]:
    """Helper: '.' = silent, '#' = voiced. Newlines stripped."""
    return [c == "#" for c in pattern if c in ".#"]


@pytest.fixture
def fast_params() -> VADParams:
    """Tighter thresholds for snappy tests."""
    return VADParams(
        sample_rate=16_000,
        threshold=0.5,
        min_silence_ms=90,  # 3 frames at 30 ms
        min_speech_ms=60,  # 2 frames at 30 ms
        pre_pad_ms=0,
        frame_ms=30,
    )


def test_no_events_on_pure_silence(fast_params: VADParams) -> None:
    sm = VADStateMachine(fast_params)
    events = sm.feed_voicing(_voicing("." * 20))
    assert events == []
    assert sm.in_speech is False


def test_start_event_after_min_speech_frames(fast_params: VADParams) -> None:
    sm = VADStateMachine(fast_params)
    # 2 voiced frames needed (min_speech_ms=60, frame=30 => 2 frames)
    events = sm.feed_voicing(_voicing("##"))
    assert events == [(1, VADEvent.SPEECH_START)]
    assert sm.in_speech is True


def test_no_start_below_min_speech_frames(fast_params: VADParams) -> None:
    sm = VADStateMachine(fast_params)
    events = sm.feed_voicing(_voicing("#.#.#."))  # voiced bursts broken by silence
    assert events == []
    assert sm.in_speech is False


def test_end_event_after_min_silence_frames(fast_params: VADParams) -> None:
    sm = VADStateMachine(fast_params)
    events = sm.feed_voicing(_voicing("######..."))
    # SPEECH_START at index 1 (after 2 voiced); SPEECH_END after 3 silent frames
    assert events[0] == (1, VADEvent.SPEECH_START)
    assert events[-1] == (8, VADEvent.SPEECH_END)
    assert sm.in_speech is False


def test_short_silence_inside_speech_does_not_end(fast_params: VADParams) -> None:
    sm = VADStateMachine(fast_params)
    # Two silent frames < min_silence_frames (3); speech should continue
    events = sm.feed_voicing(_voicing("####..####"))
    starts = [ev for _, ev in events if ev is VADEvent.SPEECH_START]
    ends = [ev for _, ev in events if ev is VADEvent.SPEECH_END]
    assert len(starts) == 1
    assert ends == []
    assert sm.in_speech is True


def test_multiple_utterances(fast_params: VADParams) -> None:
    sm = VADStateMachine(fast_params)
    events = sm.feed_voicing(_voicing("###......###......"))
    kinds = [ev for _, ev in events]
    assert kinds == [
        VADEvent.SPEECH_START,
        VADEvent.SPEECH_END,
        VADEvent.SPEECH_START,
        VADEvent.SPEECH_END,
    ]


def test_reset_clears_state(fast_params: VADParams) -> None:
    sm = VADStateMachine(fast_params)
    sm.feed_voicing(_voicing("######"))
    assert sm.in_speech is True
    sm.reset()
    assert sm.in_speech is False
    events = sm.feed_voicing(_voicing("."))
    assert events == []


def test_default_params_match_adr_006() -> None:
    """Defaults: threshold=0.5, min_silence_ms=250, min_speech_ms=120, pre_pad_ms=200."""
    p = VADParams()
    assert p.threshold == 0.5
    assert p.min_silence_ms == 250
    assert p.min_speech_ms == 120
    assert p.pre_pad_ms == 200
    assert p.sample_rate == 16_000
    assert p.frame_ms == 30


def test_frame_samples_consistency() -> None:
    p = VADParams(sample_rate=16_000, frame_ms=30)
    assert p.frame_samples == 480
    p2 = VADParams(sample_rate=16_000, frame_ms=20)
    assert p2.frame_samples == 320

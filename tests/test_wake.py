"""WakeStateMachine tests (pure logic, no model)."""

from __future__ import annotations

from whisper_agent.wake.openww import WakeEvent, WakeParams, WakeStateMachine


def test_default_params_match_adr_008() -> None:
    p = WakeParams()
    assert p.phrase == "hey_computer"
    assert p.sensitivity == 0.5
    assert p.cooldown_ms == 500


def test_below_sensitivity_does_not_fire() -> None:
    sm = WakeStateMachine(WakeParams(sensitivity=0.6))
    assert sm.step(0.5, now_s=0.0) is WakeEvent.NONE
    assert sm.step(0.59, now_s=0.1) is WakeEvent.NONE


def test_crossing_sensitivity_fires_once() -> None:
    sm = WakeStateMachine(WakeParams(sensitivity=0.6, cooldown_ms=500))
    assert sm.step(0.9, now_s=0.0) is WakeEvent.FIRE
    # Within cooldown: no second fire even on high score.
    assert sm.step(0.95, now_s=0.1) is WakeEvent.NONE
    assert sm.step(0.95, now_s=0.4) is WakeEvent.NONE


def test_fires_again_after_cooldown() -> None:
    sm = WakeStateMachine(WakeParams(sensitivity=0.6, cooldown_ms=500))
    assert sm.step(0.9, now_s=0.0) is WakeEvent.FIRE
    assert sm.step(0.9, now_s=0.6) is WakeEvent.FIRE


def test_reset_clears_cooldown() -> None:
    sm = WakeStateMachine(WakeParams(sensitivity=0.6, cooldown_ms=500))
    sm.step(0.9, now_s=0.0)
    sm.reset()
    # Immediately after reset, a fresh high score fires.
    assert sm.step(0.9, now_s=0.05) is WakeEvent.FIRE


def test_feed_confidences_walks_synthetic_clock() -> None:
    sm = WakeStateMachine(WakeParams(sensitivity=0.7, cooldown_ms=200))
    # 80 ms per frame; sensitivity 0.7; cooldown 200 ms ≈ 3 frames.
    # Pattern: low, high, high, low, high, high. Fires at idx 1, then waits ~3
    # frames before firing again at idx 4.
    events = sm.feed_confidences(
        [0.1, 0.9, 0.9, 0.0, 0.9, 0.9],
        step_s=0.08,
    )
    fires = [i for i, ev in events if ev is WakeEvent.FIRE]
    assert fires == [1, 4]


def test_exactly_at_sensitivity_fires() -> None:
    sm = WakeStateMachine(WakeParams(sensitivity=0.5))
    assert sm.step(0.5, now_s=0.0) is WakeEvent.FIRE

"""Wake-word detection."""

from whisper_agent.wake.openww import (
    OpenWakeWordDetector,
    WakeEvent,
    WakeParams,
    WakeStateMachine,
)

__all__ = [
    "OpenWakeWordDetector",
    "WakeEvent",
    "WakeParams",
    "WakeStateMachine",
]

"""Push-to-talk hotkey listener.

Thin wrapper over ``pynput``. Lazy-imports the dep so unit tests that
don't touch the keyboard don't need it installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HotkeyConfig:
    """Push-to-talk hotkey + grace timing."""

    key: str = "space"
    release_grace_ms: int = 200  # let trailing speech finish after release


class PushToTalkListener:
    """Calls ``on_press`` / ``on_release`` for the configured key.

    Backed by ``pynput.keyboard.Listener``. The listener runs on its own
    thread, so callbacks are not on the asyncio loop. Bridge with
    ``loop.call_soon_threadsafe`` if you need to schedule async work.
    """

    def __init__(
        self,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        config: HotkeyConfig | None = None,
    ) -> None:
        self.config = config or HotkeyConfig()
        self._on_press = on_press
        self._on_release = on_release
        self._listener = None
        self._held = False

    def start(self) -> None:
        try:
            from pynput import keyboard
        except ImportError as e:
            raise RuntimeError("Push-to-talk needs `pynput`. Run `pip install pynput`.") from e

        target_key = self._parse_key(self.config.key, keyboard)

        def _on_press(key):
            if self._held:
                return  # ignore key-repeat
            if self._matches(key, target_key, keyboard):
                self._held = True
                try:
                    self._on_press()
                except Exception:
                    log.exception("push-to-talk on_press raised; continuing")

        def _on_release(key):
            if not self._held:
                return
            if self._matches(key, target_key, keyboard):
                self._held = False
                try:
                    self._on_release()
                except Exception:
                    log.exception("push-to-talk on_release raised; continuing")

        self._listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
        self._listener.start()
        log.info("push-to-talk listener started: key=%s", self.config.key)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @staticmethod
    def _parse_key(name: str, keyboard):
        name = name.strip().lower()
        # Common special keys first
        special = {
            "space": keyboard.Key.space,
            "tab": keyboard.Key.tab,
            "esc": keyboard.Key.esc,
            "escape": keyboard.Key.esc,
            "f1": keyboard.Key.f1,
            "f2": keyboard.Key.f2,
            "f3": keyboard.Key.f3,
            "f4": keyboard.Key.f4,
            "f5": keyboard.Key.f5,
            "f6": keyboard.Key.f6,
            "f7": keyboard.Key.f7,
            "f8": keyboard.Key.f8,
            "f9": keyboard.Key.f9,
            "f10": keyboard.Key.f10,
            "f11": keyboard.Key.f11,
            "f12": keyboard.Key.f12,
        }
        if name in special:
            return special[name]
        # Single character (e.g. "a")
        if len(name) == 1:
            return keyboard.KeyCode.from_char(name)
        raise ValueError(f"unsupported hotkey: {name!r}")

    @staticmethod
    def _matches(event_key, target, keyboard) -> bool:
        if event_key == target:
            return True
        # KeyCode equality is finicky across platforms; compare .char if both have one.
        char_a = getattr(event_key, "char", None)
        char_b = getattr(target, "char", None)
        return char_a is not None and char_b is not None and char_a == char_b

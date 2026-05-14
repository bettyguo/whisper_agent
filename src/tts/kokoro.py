"""Kokoro TTS wrapper.

Lazy-imports ``kokoro`` so the package is importable without the dep.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from whisper_agent.tts.base import TTSChunk

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class KokoroConfig:
    voice: str = "af_default"  # Kokoro's default voice ID
    speed: float = 1.0
    sample_rate: int = 24_000


class KokoroSynthesizer:
    """Kokoro TTS synthesizer.

    Construction loads the model. The model is small enough (~500 MB) that
    a single instance can serve many requests without per-call overhead.
    """

    def __init__(self, config: KokoroConfig | None = None) -> None:
        self.config = config or KokoroConfig()
        self.sample_rate = self.config.sample_rate
        try:
            from kokoro import KPipeline
        except ImportError as e:
            raise RuntimeError(
                "Kokoro TTS isn't installed. Run `pip install whisper-agent[tts]`."
            ) from e
        self._pipeline = KPipeline(lang_code="a")  # American English
        log.info(
            "Kokoro pipeline loaded (voice=%s, speed=%.2f)", self.config.voice, self.config.speed
        )

    async def synthesize(self, text: str) -> AsyncIterator[TTSChunk]:
        """Yield TTSChunks for ``text``.

        Kokoro returns chunks at sentence granularity. We surface each as
        a separate :class:`TTSChunk` and mark the final one ``is_last``.
        """
        text = text.strip()
        if not text:
            return
        generator = self._pipeline(
            text,
            voice=self.config.voice,
            speed=self.config.speed,
        )
        # Greedy: collect first; the pipeline is a Python generator and we
        # don't want to block the event loop on the model.
        import asyncio

        loop = asyncio.get_running_loop()

        def _collect():
            return list(generator)

        chunks = await loop.run_in_executor(None, _collect)
        n = len(chunks)
        for i, (_, _, audio) in enumerate(chunks):
            samples = np.asarray(audio, dtype=np.float32)
            yield TTSChunk(
                samples=samples,
                sample_rate=self.sample_rate,
                is_last=(i == n - 1),
            )

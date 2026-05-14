"""TTS synthesizer protocol + chunk type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import numpy as np
    from numpy.typing import NDArray

    Float32 = NDArray[np.float32]


@dataclass(frozen=True)
class TTSChunk:
    """One emitted PCM chunk."""

    samples: Float32
    sample_rate: int
    is_last: bool = False


class TTSSynthesizer(Protocol):
    """Backend that turns text into a stream of PCM chunks.

    ``synthesize`` is plain ``def`` returning an
    :class:`~typing.AsyncIterator`. See the note on
    :class:`~whisper_agent.llm.tool_use.ToolUseBackend.stream` for why.
    """

    sample_rate: int

    def synthesize(self, text: str) -> AsyncIterator[TTSChunk]:
        """Yield one or more :class:`TTSChunk` for ``text``.

        The final chunk MUST have ``is_last=True``.
        """
        ...

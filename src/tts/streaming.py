"""Streaming sentence chunker.

Consumes LLM tokens incrementally and emits speakable chunks. The
latency floor is the time to the first sentence-ending punctuation, or
a soft flush threshold. After that, chunks flow as the model streams.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Match the *position* (index) right after a sentence terminator followed
# by whitespace or end-of-string. We keep the punctuation with the chunk.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(\s+|$)")

# Soft break points for over-long chunks: commas, semicolons, colons, em-dashes.
_SOFT_BREAK = re.compile(r"(?<=[,;:—])\s+")


@dataclass(frozen=True)
class ChunkerConfig:
    """How aggressively to chunk.

    - ``min_chunk_chars``: don't emit a chunk shorter than this unless we
      reach end-of-stream. Prevents micro-chunks like "OK." from going through
      TTS individually and sounding choppy.
    - ``max_chunk_chars``: force a soft break if a chunk grows past this.
      Keeps first-chunk latency bounded when the model rambles without
      punctuation.
    """

    min_chunk_chars: int = 24
    max_chunk_chars: int = 220


class SentenceChunker:
    """Stateful chunker. Feed tokens, get speakable text chunks."""

    __slots__ = ("_buf", "config")

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()
        self._buf: str = ""

    def feed(self, token: str) -> list[str]:
        """Append ``token`` to the internal buffer; return any chunks ready."""
        if not token:
            return []
        self._buf += token
        return self._drain(force=False)

    def flush(self) -> list[str]:
        """End of input. Return whatever's left as one final chunk."""
        out = self._drain(force=True)
        tail = self._buf.strip()
        if tail:
            out.append(tail)
            self._buf = ""
        return out

    def reset(self) -> None:
        self._buf = ""

    def _drain(self, *, force: bool) -> list[str]:
        chunks: list[str] = []
        while True:
            buf = self._buf
            if not buf:
                break

            # Try a hard sentence break first.
            m = _SENTENCE_BREAK.search(buf)
            if m:
                cut = m.end()
                head, tail = buf[:cut], buf[cut:]
                head_stripped = head.strip()
                if head_stripped and (len(head_stripped) >= self.config.min_chunk_chars or force):
                    chunks.append(head_stripped)
                    self._buf = tail
                    continue
                # Too short; keep accumulating.
                break

            # No sentence break in sight; consider a soft break if too long.
            if len(buf) >= self.config.max_chunk_chars:
                m_soft = _SOFT_BREAK.search(buf)
                if m_soft:
                    cut = m_soft.end()
                    head, tail = buf[:cut], buf[cut:]
                    head_stripped = head.strip()
                    if head_stripped:
                        chunks.append(head_stripped)
                        self._buf = tail
                        continue
                # No soft break either: emit the whole buffer.
                chunks.append(buf.strip())
                self._buf = ""
                continue

            break
        return chunks

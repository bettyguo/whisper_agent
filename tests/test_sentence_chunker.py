"""SentenceChunker: feed tokens, get speakable chunks."""

from __future__ import annotations

from whisper_agent.tts.streaming import ChunkerConfig, SentenceChunker


def _feed(chunker: SentenceChunker, text: str, chunk_size: int = 4) -> list[str]:
    """Token-feed ``text`` in ``chunk_size``-character pieces."""
    out: list[str] = []
    for i in range(0, len(text), chunk_size):
        out.extend(chunker.feed(text[i : i + chunk_size]))
    out.extend(chunker.flush())
    return out


def test_empty_input_emits_nothing() -> None:
    c = SentenceChunker()
    assert c.feed("") == []
    assert c.flush() == []


def test_single_sentence_only_on_flush() -> None:
    c = SentenceChunker(ChunkerConfig(min_chunk_chars=4, max_chunk_chars=200))
    # Tiny utterance without terminator: only emits on flush.
    chunks = _feed(c, "hello there", chunk_size=4)
    assert chunks == ["hello there"]


def test_sentence_boundary_emits_immediately() -> None:
    c = SentenceChunker(ChunkerConfig(min_chunk_chars=4, max_chunk_chars=200))
    chunks = _feed(c, "Hello there. How are you?", chunk_size=4)
    # Two complete sentences.
    assert chunks == ["Hello there.", "How are you?"]


def test_short_sentence_below_min_chunk_buffers() -> None:
    c = SentenceChunker(ChunkerConfig(min_chunk_chars=20, max_chunk_chars=200))
    # "Hi." is too short to emit; should buffer until the next sentence.
    chunks = _feed(c, "Hi. How are you doing today?", chunk_size=4)
    # First sentence is below min but flushes when the next sentence pushes
    # the buffer past the threshold.
    assert len(chunks) >= 1
    joined = " ".join(chunks)
    assert "Hi." in joined
    assert "How are you doing today?" in joined


def test_long_sentence_forces_soft_break() -> None:
    c = SentenceChunker(ChunkerConfig(min_chunk_chars=4, max_chunk_chars=40))
    text = "This is a long clause, with another part, and yet another portion that keeps going."
    chunks = _feed(c, text, chunk_size=8)
    # At least one chunk should have come out before the final terminator.
    assert len(chunks) >= 2
    # All content preserved when joined.
    rejoined = " ".join(c.strip() for c in chunks)
    assert "long clause" in rejoined
    assert "keeps going" in rejoined


def test_flush_emits_trailing_buffer() -> None:
    c = SentenceChunker()
    c.feed("Tail text without terminator")
    assert c.flush() == ["Tail text without terminator"]


def test_reset_clears_buffer() -> None:
    c = SentenceChunker()
    c.feed("some text")
    c.reset()
    assert c.flush() == []


def test_question_and_exclamation_count_as_terminators() -> None:
    c = SentenceChunker(ChunkerConfig(min_chunk_chars=2, max_chunk_chars=200))
    chunks = _feed(c, "What? Yes! Done.", chunk_size=2)
    assert chunks == ["What?", "Yes!", "Done."]

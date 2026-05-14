"""Text-to-speech backends with streaming support."""

from whisper_agent.tts.base import TTSChunk, TTSSynthesizer
from whisper_agent.tts.streaming import SentenceChunker

__all__ = ["SentenceChunker", "TTSChunk", "TTSSynthesizer"]

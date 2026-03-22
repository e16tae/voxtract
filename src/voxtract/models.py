"""Core Pydantic data models for the voxtract pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Utterance(BaseModel):
    """A single speaker utterance with timestamps."""

    speaker: str = Field(description="Speaker identifier (e.g. 'Speaker 1')")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text content")


class Transcript(BaseModel):
    """Full transcript with speaker diarization."""

    language: str = Field(description="Detected or specified language code (e.g. 'ko')")
    speakers: list[str] = Field(description="List of unique speaker identifiers")
    utterances: list[Utterance] = Field(default_factory=list)
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (duration, audio_file, created_at, etc.)",
    )


class ChunkInfo(BaseModel):
    """Metadata for a single audio chunk."""

    index: int
    start_time: float = Field(description="Start time in seconds relative to original audio")
    end_time: float = Field(description="End time in seconds relative to original audio")
    audio_path: str = Field(description="Path to the chunk audio file")

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

"""STT provider protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from voxtract.models import Transcript


@runtime_checkable
class STTProvider(Protocol):
    """Protocol for speech-to-text providers.

    Providers produce Transcript with timestamps but without speaker labels.
    Speaker diarization is handled separately by speaker/diarizer.
    """

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
        """Transcribe an audio file and return a Transcript."""
        ...

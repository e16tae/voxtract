"""Application configuration via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global settings loaded from environment variables and .env files."""

    model_config = {"env_prefix": "VOXTRACT_", "env_file": ".env"}

    # Device
    device: str = "auto"  # "auto", "cuda", "cuda:0", "cuda:1", "cpu"

    # STT
    stt_provider: str = "qwen3"
    stt_model: str = "Qwen/Qwen3-ASR-1.7B"
    stt_aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    stt_context: str = ""  # contextual hints for ASR (e.g. topic, names, jargon)

    # Speaker diarization
    speaker_model: str = "pyannote/speaker-diarization-community-1"

    # Chunking
    chunk_minutes: int = 25
    overlap_seconds: int = 60

    # Output
    output_dir: str = "."


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


def resolve_device(settings: Settings | None = None) -> str:
    """Resolve the device string from settings.

    "auto" → "cuda" if available, else "cpu".
    """
    if settings is None:
        settings = get_settings()
    device = settings.device
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device

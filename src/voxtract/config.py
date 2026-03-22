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


# Empirical constants from RTX 4070 SUPER (11.6GB) testing
_MODEL_MEMORY_GB = 9.5   # ASR 1.7B + Aligner 0.6B in bfloat16
_OVERHEAD_GB = 0.5        # CUDA context + fragmentation buffer
_PEAK_PER_BATCH_GB = 0.68  # lm_head peak allocation per batch item


def resolve_batch_size(device: str) -> int:
    """Calculate optimal max_inference_batch_size from available VRAM.

    Based on empirical measurement: lm_head peak is ~0.68GB per batch item,
    model weights take ~9.5GB in bfloat16, plus ~0.5GB overhead.
    """
    if not device.startswith("cuda"):
        return 4

    try:
        import torch
        dev_idx = int(device.split(":")[-1]) if ":" in device else 0
        total_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024 ** 3)
    except Exception:
        return 2

    available_gb = total_gb - _MODEL_MEMORY_GB - _OVERHEAD_GB
    batch_size = max(1, int(available_gb / _PEAK_PER_BATCH_GB))
    return min(batch_size, 16)

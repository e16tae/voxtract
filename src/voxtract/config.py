"""Application configuration via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global settings loaded from environment variables and .env files."""

    model_config = {"env_prefix": "VOXTRACT_", "env_file": ".env"}

    # Device — separate GPUs for STT (heavy) vs speaker/VAD (light)
    device: str = "auto"  # "auto", "cuda", "cuda:0", "cuda:1", "cpu"
    device_stt: str = ""  # device for ASR model (empty = use 'device')
    device_speaker: str = ""  # device for diarizer + VAD (empty = use 'device')

    # Language
    language: str = "ko"  # default language for STT (ISO 639-1)

    # STT
    stt_provider: str = "qwen3"
    stt_model: str = "Qwen/Qwen3-ASR-1.7B"
    stt_aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    stt_context: str = ""  # contextual hints for ASR (e.g. topic, names, jargon)
    stt_repetition_penalty: float = 1.2  # suppress repeated token hallucinations
    stt_max_tokens: int = 1024  # max generated tokens per inference batch
    stt_temperature: float = 0.0  # decoding temperature (0.0 = greedy, most stable for ASR)
    stt_num_beams: int = 3  # beam search width (1 = greedy, higher = better quality, more VRAM)

    # Speaker diarization
    speaker_model: str = "pyannote/speaker-diarization-3.1"

    # Audio preprocessing
    audio_normalize: bool = True   # apply EBU R128 loudness normalization before STT
    audio_highpass: bool = True    # apply 80Hz high-pass filter to remove low-frequency rumble

    # VAD (Voice Activity Detection)
    vad_filter: bool = True        # filter STT hallucinations using VAD
    vad_model: str = "pyannote/segmentation-3.0"

    # Chunking
    chunk_minutes: int = 25
    overlap_seconds: int = 60

    # Output
    output_dir: str = "."


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


def _resolve_auto() -> str:
    """Resolve 'auto' to 'cuda' or 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _pick_best_gpu() -> str:
    """Return the CUDA device with the most total memory."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        best_idx, best_mem = 0, 0
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory
            if mem > best_mem:
                best_idx, best_mem = i, mem
        return f"cuda:{best_idx}"
    except Exception:
        return "cpu"


def _pick_secondary_gpu(primary: str) -> str:
    """Return a CUDA device different from *primary*, or *primary* if only one GPU."""
    try:
        import torch
        n = torch.cuda.device_count()
        if n < 2:
            return primary
        primary_idx = int(primary.split(":")[-1]) if ":" in primary else 0
        for i in range(n):
            if i != primary_idx:
                return f"cuda:{i}"
        return primary
    except Exception:
        return primary


def resolve_device(settings: Settings | None = None) -> str:
    """Resolve the main device string from settings.

    "auto" → "cuda" if available, else "cpu".
    """
    if settings is None:
        settings = get_settings()
    device = settings.device
    if device == "auto":
        return _resolve_auto()
    return device


def resolve_device_stt(settings: Settings | None = None) -> str:
    """Resolve device for STT model. Auto-selects the GPU with most VRAM."""
    if settings is None:
        settings = get_settings()
    if settings.device_stt:
        d = settings.device_stt
        return _resolve_auto() if d == "auto" else d
    d = settings.device
    if d == "auto":
        return _pick_best_gpu()
    return d


def resolve_device_speaker(settings: Settings | None = None) -> str:
    """Resolve device for diarizer/VAD. Auto-selects a secondary GPU if available."""
    if settings is None:
        settings = get_settings()
    if settings.device_speaker:
        d = settings.device_speaker
        return _resolve_auto() if d == "auto" else d
    d = settings.device
    if d == "auto":
        stt_dev = resolve_device_stt(settings)
        return _pick_secondary_gpu(stt_dev)
    return d

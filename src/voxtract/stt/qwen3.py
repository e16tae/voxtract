"""Qwen3 ASR STT provider via qwen-asr package.

Speech-to-text using Qwen3-ASR-1.7B + ForcedAligner.
Runs on CUDA (bfloat16) when available, falls back to CPU (float32).
Produces word-level timestamps, grouped into utterance-level segments.
No built-in speaker diarization — use speaker/diarizer for that.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from voxtract.config import Settings, resolve_device
from voxtract.errors import STTError
from voxtract.models import Transcript, Utterance
from voxtract.stt import register

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
_DEFAULT_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"
_PAUSE_THRESHOLD_S = 1.5  # gap > 1.5s between words → new utterance

# Empirical constants from RTX 4070 SUPER (12GB) testing.
# Actual model weights: ASR ~4.46GB + Aligner ~1.84GB = ~6.3GB (bf16).
# _MODEL_MEMORY_GB represents peak inference VRAM baseline (weights + activations),
# not weights alone.
_MODEL_MEMORY_GB = 9.5   # peak VRAM baseline during inference (bf16)
_OVERHEAD_GB = 0.5        # CUDA context + fragmentation buffer
_PEAK_PER_BATCH_GB = 0.68  # lm_head peak allocation per batch item


def _resolve_batch_size(device: str) -> int:
    """Calculate optimal max_inference_batch_size from available VRAM."""
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


class Qwen3Provider:
    """STT provider using qwen-asr (Qwen3-ASR + ForcedAligner).

    - qwen-asr handles long audio internally (low-energy boundary splitting)
    - Word-level timestamps via ForcedAligner
    - No speaker diarization (all utterances "Speaker 0")
    """

    handles_long_audio: bool = True

    def __init__(self, *, settings: Settings | None = None) -> None:
        if settings is None:
            from voxtract.config import get_settings
            settings = get_settings()
        self._settings = settings
        self._model_repo = settings.stt_model or _DEFAULT_MODEL
        self._aligner_repo = settings.stt_aligner or _DEFAULT_ALIGNER
        self._model = None

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return self._model

        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise STTError(
                "qwen-asr is not installed. "
                "Install it with: uv pip install voxtract[stt]",
                code="STT_DEPENDENCY_MISSING",
                recoverable=False,
            )

        device = resolve_device(self._settings)
        if device.startswith("cuda"):
            dtype = torch.bfloat16
            import os
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        else:
            dtype = torch.float32

        batch_size = _resolve_batch_size(device)
        logger.info(
            "Loading Qwen3 ASR on device=%s dtype=%s batch_size=%d",
            device, dtype, batch_size,
        )

        try:
            attn_impl = "sdpa" if device.startswith("cuda") else "eager"

            self._model = Qwen3ASRModel.from_pretrained(
                self._model_repo,
                dtype=dtype,
                device_map=device,
                attn_implementation=attn_impl,
                max_new_tokens=512,
                max_inference_batch_size=batch_size,
                forced_aligner=self._aligner_repo,
                forced_aligner_kwargs=dict(
                    dtype=dtype,
                    device_map=device,
                    attn_implementation=attn_impl,
                ),
            )
            # Disable KV cache — saves VRAM, no effect on output quality
            if hasattr(self._model, 'model') and hasattr(self._model.model, 'config'):
                self._model.model.config.use_cache = False
        except Exception as exc:
            raise STTError(
                f"Failed to load Qwen3 ASR model '{self._model_repo}': {exc}",
                code="STT_MODEL_LOAD",
                recoverable=False,
            ) from exc

        return self._model

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> Transcript:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise STTError(
                f"Audio file not found: {audio_path}",
                code="STT_FILE_NOT_FOUND",
            )

        model = self._load_model()
        context = self._settings.stt_context or ""

        try:
            results = model.transcribe(
                audio=str(audio_path),
                context=context,
                language=language,
                return_time_stamps=True,
            )
        except Exception as exc:
            raise STTError(
                f"Qwen3 ASR transcription failed: {exc}",
                code="STT_TRANSCRIBE",
                recoverable=True,
            ) from exc

        result = results[0]
        return self._build_transcript(result, audio_path)

    def _build_transcript(self, result, audio_path: Path) -> Transcript:
        """Convert qwen-asr ASRTranscription to our Transcript model.

        Groups word-level timestamps into utterance-level segments
        by splitting on pauses > _PAUSE_THRESHOLD_S.
        """
        utterances: list[Utterance] = []

        if result.time_stamps is not None and result.time_stamps.items:
            utterances = self._group_words_into_utterances(result.time_stamps.items)
        elif result.text and result.text.strip():
            utterances = [
                Utterance(
                    speaker="Speaker 0",
                    start_time=0.0,
                    end_time=0.0,
                    text=result.text.strip(),
                )
            ]

        lang = result.language if isinstance(result.language, str) else "auto"

        return Transcript(
            language=lang,
            speakers=sorted({u.speaker for u in utterances}) if utterances else [],
            utterances=utterances,
            metadata={
                "audio_file": str(audio_path),
                "duration": utterances[-1].end_time if utterances else 0.0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": self._model_repo,
            },
        )

    @staticmethod
    def _group_words_into_utterances(
        items: list,
        pause_threshold: float = _PAUSE_THRESHOLD_S,
    ) -> list[Utterance]:
        """Group ForcedAlignItems into utterances based on pause gaps.

        ForcedAligner returns start_time/end_time in seconds (float).
        """
        if not items:
            return []

        utterances: list[Utterance] = []
        current_words: list[str] = []
        current_start: float = float(items[0].start_time)
        prev_end: float = float(items[0].end_time)

        for item in items:
            gap = float(item.start_time) - prev_end

            if gap > pause_threshold and current_words:
                text = " ".join(current_words).strip()
                if text:
                    utterances.append(Utterance(
                        speaker="Speaker 0",
                        start_time=current_start,
                        end_time=prev_end,
                        text=text,
                    ))
                current_words = []
                current_start = float(item.start_time)

            current_words.append(item.text)
            prev_end = float(item.end_time)

        if current_words:
            text = " ".join(current_words).strip()
            if text:
                utterances.append(Utterance(
                    speaker="Speaker 0",
                    start_time=current_start,
                    end_time=prev_end,
                    text=text,
                ))

        return utterances


register("qwen3", Qwen3Provider)

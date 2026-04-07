"""Whisper STT provider via faster-whisper (CTranslate2).

Speech-to-text using OpenAI Whisper models (large-v3, large-v3-turbo, etc.)
via the faster-whisper library for efficient GPU/CPU inference.
Produces word-level timestamps via Whisper's built-in alignment.
No built-in speaker diarization — use speaker/diarizer for that.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from voxtract.config import Settings, resolve_device_stt
from voxtract.errors import STTError
from voxtract.models import Transcript, Utterance, WordTimestamp
from voxtract.stt import register

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "large-v3-turbo"

# ISO 639-1 codes that Whisper accepts directly
_SUPPORTED_LANGS = {
    "ko", "en", "zh", "ja", "de", "fr", "es", "pt", "ru", "ar",
    "it", "nl", "sv", "da", "fi", "pl", "cs", "tr", "hi", "th",
    "vi", "id", "ms", "ro", "hu", "el", "fa", "fil", "uk", "he",
}


class WhisperProvider:
    """STT provider using faster-whisper (CTranslate2-optimized Whisper).

    - Handles long audio natively (sequential decoding with seek)
    - Word-level timestamps via Whisper alignment
    - No speaker diarization (all utterances "Speaker 0")
    """

    handles_long_audio: bool = True

    def __init__(self, *, settings: Settings | None = None) -> None:
        if settings is None:
            from voxtract.config import get_settings
            settings = get_settings()
        self._settings = settings
        self._model_size = settings.stt_model or _DEFAULT_MODEL
        self._model = None

    def _load_model(self):
        """Lazy-load the faster-whisper model on first use."""
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise STTError(
                "faster-whisper is not installed. "
                "Install it with: uv pip install faster-whisper",
                code="STT_DEPENDENCY_MISSING",
                recoverable=False,
            )

        device = resolve_device_stt(self._settings)
        compute_type = "float16" if device.startswith("cuda") else "int8"

        # Map device string to faster-whisper format
        if device.startswith("cuda"):
            fw_device = "cuda"
            device_index = int(device.split(":")[-1]) if ":" in device else 0
        else:
            fw_device = "cpu"
            device_index = 0

        logger.info(
            "Loading Whisper %s on device=%s compute_type=%s",
            self._model_size, device, compute_type,
        )

        try:
            self._model = WhisperModel(
                self._model_size,
                device=fw_device,
                device_index=device_index,
                compute_type=compute_type,
            )
        except Exception as exc:
            raise STTError(
                f"Failed to load Whisper model '{self._model_size}': {exc}",
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

        lang = language if language in _SUPPORTED_LANGS else None
        context = self._settings.stt_context or None

        beam_size = max(1, self._settings.stt_num_beams)

        # Temperature fallback: if greedy decoding fails (high compression ratio
        # or low log prob), retry with progressively higher temperatures to
        # recover content from difficult segments instead of dropping them.
        temperature = self._settings.stt_temperature
        if temperature == 0.0:
            temperature_seq = [0.0, 0.2, 0.4, 0.6, 0.8]
        else:
            temperature_seq = temperature

        try:
            segments, info = model.transcribe(
                str(audio_path),
                language=lang,
                beam_size=beam_size,
                temperature=temperature_seq,
                initial_prompt=context,
                word_timestamps=True,
                repetition_penalty=self._settings.stt_repetition_penalty,
                vad_filter=False,  # we have our own VAD
                condition_on_previous_text=True,
                hallucination_silence_threshold=2.0,
                compression_ratio_threshold=2.0,
                no_speech_threshold=0.35,
            )
        except Exception as exc:
            raise STTError(
                f"Whisper transcription failed: {exc}",
                code="STT_TRANSCRIBE",
                recoverable=True,
            ) from exc

        return self._build_transcript(segments, info, audio_path)

    def _build_transcript(self, segments, info, audio_path: Path) -> Transcript:
        """Convert faster-whisper segments to our Transcript model."""
        utterances: list[Utterance] = []
        low_confidence: list[dict] = []

        for segment in segments:
            words = None
            if segment.words:
                words = [
                    WordTimestamp(
                        text=w.word.strip(),
                        start_time=w.start,
                        end_time=w.end,
                    )
                    for w in segment.words
                    if w.word.strip()
                ]

            text = segment.text.strip()
            if not text:
                continue

            utterances.append(Utterance(
                speaker="Speaker 0",
                start_time=segment.start,
                end_time=segment.end,
                text=text,
                words=words,
            ))

            # Track low-confidence segments for downstream review
            avg_lp = getattr(segment, "avg_logprob", 0)
            if avg_lp is not None and avg_lp < -0.8:
                low_confidence.append({
                    "start": segment.start,
                    "end": segment.end,
                    "avg_logprob": round(avg_lp, 3),
                    "no_speech_prob": round(getattr(segment, "no_speech_prob", 0) or 0, 3),
                    "text": text[:80],
                })

        lang = info.language or "auto"

        return Transcript(
            language=lang,
            speakers=sorted({u.speaker for u in utterances}) if utterances else [],
            utterances=utterances,
            metadata={
                "audio_file": str(audio_path),
                "duration": utterances[-1].end_time if utterances else 0.0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": f"whisper-{self._model_size}",
                "language_probability": info.language_probability,
                "low_confidence_segments": low_confidence,
            },
        )


register("whisper", WhisperProvider)

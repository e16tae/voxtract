"""Voice Activity Detection using pyannote.audio segmentation model.

Identifies speech regions in audio to filter STT hallucinations
that arise from silence, noise, or music segments.
"""
from __future__ import annotations

import logging
from pathlib import Path

from voxtract.config import Settings, resolve_device_speaker
from voxtract.errors import AudioError
from voxtract.models import Utterance

logger = logging.getLogger(__name__)

_MIN_SPEECH_OVERLAP = 0.5  # keep utterance if >= 50% overlaps with speech


def get_speech_segments(
    audio_path: Path,
    *,
    device: str | None = None,
    settings: Settings | None = None,
) -> list[tuple[float, float]]:
    """Run VAD on audio and return speech segment timestamps.

    Returns a sorted list of (start_time, end_time) tuples in seconds.
    Uses the pyannote segmentation model configured in settings.
    """
    try:
        from pyannote.audio import Model
        from pyannote.audio.pipelines import VoiceActivityDetection
    except ImportError:
        raise AudioError(
            "pyannote.audio is not installed. "
            "Install it with: uv pip install voxtract[speaker]",
            code="AUDIO_VAD_DEPENDENCY",
            recoverable=False,
        )

    if settings is None:
        from voxtract.config import get_settings
        settings = get_settings()

    if device is None:
        device = resolve_device_speaker(settings)

    try:
        model = Model.from_pretrained(settings.vad_model)
        pipeline = VoiceActivityDetection(segmentation=model)
        pipeline.instantiate({
            "min_duration_on": 0.3,
            "min_duration_off": 0.3,
        })

        if device.startswith("cuda"):
            import torch
            pipeline.to(torch.device(device))

        vad_result = pipeline(str(audio_path))
    except Exception as exc:
        raise AudioError(
            f"VAD failed: {exc}",
            code="AUDIO_VAD_FAILED",
            recoverable=True,
        ) from exc

    segments = [
        (speech_turn.start, speech_turn.end)
        for speech_turn, _, _ in vad_result.itertracks(yield_label=True)
    ]

    logger.info("VAD: %d speech segments in %s", len(segments), audio_path.name)
    return segments


def _speech_overlap_ratio(
    utterance: Utterance,
    speech_segments: list[tuple[float, float]],
) -> float:
    """Fraction of an utterance's duration that overlaps with speech segments."""
    utt_duration = utterance.end_time - utterance.start_time
    if utt_duration <= 0:
        return 1.0  # zero-duration — keep

    total_overlap = 0.0
    for seg_start, seg_end in speech_segments:
        ov_start = max(utterance.start_time, seg_start)
        ov_end = min(utterance.end_time, seg_end)
        if ov_end > ov_start:
            total_overlap += ov_end - ov_start

    return total_overlap / utt_duration


def filter_utterances_by_vad(
    utterances: list[Utterance],
    speech_segments: list[tuple[float, float]],
    min_overlap: float = _MIN_SPEECH_OVERLAP,
) -> list[Utterance]:
    """Remove utterances that don't sufficiently overlap with speech regions.

    An utterance is dropped if less than *min_overlap* of its duration
    falls within VAD-detected speech.  This catches STT hallucinations
    produced from silence, noise, or music.
    """
    if not speech_segments:
        return utterances

    kept: list[Utterance] = []
    dropped = 0
    for utt in utterances:
        if _speech_overlap_ratio(utt, speech_segments) >= min_overlap:
            kept.append(utt)
        else:
            dropped += 1
            logger.debug(
                "VAD drop: '%.40s' (%.1f-%.1fs)",
                utt.text, utt.start_time, utt.end_time,
            )

    if dropped:
        logger.info("VAD filtered %d hallucinated utterance(s)", dropped)
    return kept

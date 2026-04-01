"""Speaker diarization using pyannote.audio pipeline.

Assigns speaker labels to STT utterances by running pyannote
diarization on the audio and matching speaker segments by time overlap.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from voxtract.audio.splitter import convert_to_wav16k
from voxtract.config import Settings, resolve_device
from voxtract.errors import SpeakerError
from voxtract.models import Transcript, Utterance

logger = logging.getLogger(__name__)


def _load_pipeline(device: str, *, settings: Settings | None = None):
    """Load pyannote diarization pipeline onto the given device."""
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        raise SpeakerError(
            "pyannote.audio is not installed. "
            "Install it with: uv pip install pyannote.audio",
            code="SPEAKER_DEPENDENCY_MISSING",
            recoverable=False,
        )

    if settings is None:
        from voxtract.config import get_settings
        settings = get_settings()

    model_name = settings.speaker_model

    try:
        pipeline = Pipeline.from_pretrained(model_name)
    except Exception as exc:
        raise SpeakerError(
            f"Failed to load pyannote pipeline '{model_name}': {exc}. "
            "You may need to accept the model terms at "
            f"https://huggingface.co/{model_name} and set HF_TOKEN.",
            code="SPEAKER_MODEL_LOAD",
            recoverable=False,
        ) from exc

    if device.startswith("cuda"):
        import torch
        pipeline.to(torch.device(device))

    return pipeline


_MIN_SPLIT_DURATION_S = 2.0  # Don't split utterances shorter than this


def _split_by_speaker_change(
    utterances: list[Utterance],
    segments: list[tuple[float, float, str]],
) -> list[Utterance]:
    """Split utterances that span speaker change boundaries.

    When an utterance's time range crosses from one speaker segment to another,
    split it at the boundary. Text is proportionally divided based on time.
    Short utterances (<_MIN_SPLIT_DURATION_S) are not split.
    """
    if not utterances or not segments:
        return utterances

    result: list[Utterance] = []

    for utt in utterances:
        duration = utt.end_time - utt.start_time
        if duration < _MIN_SPLIT_DURATION_S:
            result.append(utt)
            continue

        # Find all speaker segments that overlap this utterance
        overlapping = []
        for seg_start, seg_end, speaker in segments:
            overlap_start = max(utt.start_time, seg_start)
            overlap_end = min(utt.end_time, seg_end)
            if overlap_end > overlap_start:
                overlapping.append((overlap_start, overlap_end, speaker))

        # Merge consecutive segments with the same speaker to avoid
        # unnecessary splits (pyannote can emit adjacent same-speaker segments)
        merged: list[tuple[float, float, str]] = []
        for seg in overlapping:
            if merged and merged[-1][2] == seg[2]:
                prev = merged[-1]
                merged[-1] = (prev[0], seg[1], seg[2])
            else:
                merged.append(seg)
        overlapping = merged

        if len(overlapping) <= 1:
            result.append(utt)
            continue

        # Split text proportionally by actual overlap duration (not utterance
        # duration) so that gaps between segments don't skew word distribution
        words = utt.text.split()
        total_overlap = sum(e - s for s, e, _ in overlapping)

        for seg_start, seg_end, speaker in overlapping:
            seg_duration = seg_end - seg_start
            ratio = seg_duration / total_overlap
            word_count = max(1, round(len(words) * ratio))

            seg_words = words[:word_count]
            words = words[word_count:]

            text = " ".join(seg_words).strip()
            if text:
                result.append(Utterance(
                    speaker=speaker,
                    start_time=seg_start,
                    end_time=seg_end,
                    text=text,
                ))

        # Remaining words go to last segment
        if words:
            last = result[-1]
            result[-1] = Utterance(
                speaker=last.speaker,
                start_time=last.start_time,
                end_time=last.end_time,
                text=last.text + " " + " ".join(words),
            )

    return result


def _assign_speakers(
    utterances: list[Utterance],
    segments: list[tuple[float, float, str]],
) -> list[Utterance]:
    """Assign pyannote speaker labels to utterances by time overlap."""
    new_utterances = []
    for utt in utterances:
        mid = (utt.start_time + utt.end_time) / 2
        best_speaker = "Speaker 0"
        best_overlap = 0.0

        for seg_start, seg_end, speaker in segments:
            overlap_start = max(utt.start_time, seg_start)
            overlap_end = min(utt.end_time, seg_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        if best_overlap == 0.0 and segments:
            # No overlap found; assign the temporally nearest segment's speaker
            best_dist = float("inf")
            for seg_start, seg_end, speaker in segments:
                dist = min(abs(mid - seg_start), abs(mid - seg_end))
                if dist < best_dist:
                    best_dist = dist
                    best_speaker = speaker

        new_utterances.append(Utterance(
            speaker=best_speaker,
            start_time=utt.start_time,
            end_time=utt.end_time,
            text=utt.text,
        ))

    return new_utterances


def _normalize_speaker_labels(utterances: list[Utterance]) -> list[Utterance]:
    """Rename pyannote labels (SPEAKER_00, SPEAKER_01, ...) to Speaker 1, 2, ..."""
    label_map: dict[str, str] = {}
    counter = 1
    result = []

    for utt in utterances:
        if utt.speaker not in label_map:
            label_map[utt.speaker] = f"Speaker {counter}"
            counter += 1
        result.append(Utterance(
            speaker=label_map[utt.speaker],
            start_time=utt.start_time,
            end_time=utt.end_time,
            text=utt.text,
        ))

    return result


def diarize_transcript(
    transcript: Transcript,
    audio_path: Path,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    settings: Settings | None = None,
) -> Transcript:
    """Assign speaker labels to utterances using pyannote diarization."""
    if not transcript.utterances:
        return transcript

    if settings is None:
        from voxtract.config import get_settings
        settings = get_settings()

    device = resolve_device(settings)
    logger.info("Running pyannote diarization on device=%s", device)

    pipeline = _load_pipeline(device, settings=settings)

    diarize_kwargs = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    # Convert to 16kHz mono WAV if needed (pyannote requires WAV)
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = convert_to_wav16k(audio_path, output_dir=Path(tmp_dir))

        try:
            result = pipeline(str(wav_path), **diarize_kwargs)
        except Exception as exc:
            raise SpeakerError(
                f"Pyannote diarization failed: {exc}",
                code="SPEAKER_DIARIZE",
                recoverable=True,
            ) from exc

    # pyannote 4.x returns DiarizeOutput; use exclusive (no overlap) for ASR alignment
    if hasattr(result, "exclusive_speaker_diarization"):
        diarization = result.exclusive_speaker_diarization
    else:
        diarization = result

    # Extract segments once; used by both _assign_speakers and _split_by_speaker_change
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    new_utterances = _assign_speakers(transcript.utterances, segments)
    new_utterances = _split_by_speaker_change(new_utterances, segments)
    new_utterances = _normalize_speaker_labels(new_utterances)
    new_speakers = sorted({u.speaker for u in new_utterances})

    logger.info("Diarization complete: %d speakers detected", len(new_speakers))

    return Transcript(
        language=transcript.language,
        speakers=new_speakers,
        utterances=new_utterances,
        metadata=transcript.metadata,
    )

"""Speaker diarization using pyannote.audio pipeline.

Assigns speaker labels to STT utterances by running pyannote
diarization on the audio and matching speaker segments by time overlap.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from voxtract.audio.splitter import convert_to_wav16k
from voxtract.config import Settings, resolve_device_speaker
from voxtract.errors import SpeakerError
from voxtract.models import Transcript, Utterance, WordTimestamp

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


_PAUSE_THRESHOLD_S = 1.5  # Same-speaker gap > this → new utterance


def _assign_word_speaker(
    word: WordTimestamp,
    segments: list[tuple[float, float, str]],
) -> str:
    """Assign a single word to the speaker segment with greatest overlap."""
    best_speaker = "Speaker 0"
    best_overlap = 0.0
    mid = (word.start_time + word.end_time) / 2

    for seg_start, seg_end, speaker in segments:
        ov_start = max(word.start_time, seg_start)
        ov_end = min(word.end_time, seg_end)
        ov = ov_end - ov_start
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = speaker

    if best_overlap <= 0.0 and segments:
        # Zero-duration word or in a gap — check midpoint containment first
        for seg_start, seg_end, speaker in segments:
            if seg_start <= mid <= seg_end:
                return speaker
        # Fall back to nearest segment
        best_dist = float("inf")
        for seg_start, seg_end, speaker in segments:
            dist = min(abs(mid - seg_start), abs(mid - seg_end))
            if dist < best_dist:
                best_dist = dist
                best_speaker = speaker

    return best_speaker


def _build_utterances_from_words(
    utterances: list[Utterance],
    segments: list[tuple[float, float, str]],
    pause_threshold: float = _PAUSE_THRESHOLD_S,
) -> list[Utterance]:
    """Assign speakers at word level, then re-group into utterances.

    Flattens all word timestamps from input utterances, assigns each word
    to the best-matching diarizer segment, and groups consecutive words
    by speaker continuity and pause gaps.
    """
    if not utterances:
        return []

    # Flatten all words; fall back to utterance-level if no word timestamps
    all_words: list[WordTimestamp] = []
    has_word_timestamps = False
    for utt in utterances:
        if utt.words:
            all_words.extend(utt.words)
            has_word_timestamps = True
        else:
            all_words.append(WordTimestamp(
                text=utt.text,
                start_time=utt.start_time,
                end_time=utt.end_time,
            ))

    if not all_words:
        return []

    if not segments:
        # No diarizer segments — keep original speaker, re-group by pause
        word_speakers = [(w, utterances[0].speaker) for w in all_words]
    else:
        word_speakers = [(w, _assign_word_speaker(w, segments)) for w in all_words]

    # Group consecutive words by speaker + pause threshold
    result: list[Utterance] = []
    current_words: list[WordTimestamp] = []
    current_speaker = ""

    for word, speaker in word_speakers:
        if current_words:
            gap = word.start_time - current_words[-1].end_time
            if speaker != current_speaker or gap > pause_threshold:
                text = " ".join(w.text for w in current_words).strip()
                if text:
                    result.append(Utterance(
                        speaker=current_speaker,
                        start_time=current_words[0].start_time,
                        end_time=current_words[-1].end_time,
                        text=text,
                        words=current_words if has_word_timestamps else None,
                    ))
                current_words = []

        current_words.append(word)
        current_speaker = speaker

    if current_words:
        text = " ".join(w.text for w in current_words).strip()
        if text:
            result.append(Utterance(
                speaker=current_speaker,
                start_time=current_words[0].start_time,
                end_time=current_words[-1].end_time,
                text=text,
                words=current_words if has_word_timestamps else None,
            ))

    return result



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
            words=utt.words,
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

    device = resolve_device_speaker(settings)
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

    # Extract segments for word-level speaker assignment
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    new_utterances = _build_utterances_from_words(transcript.utterances, segments)
    new_utterances = _normalize_speaker_labels(new_utterances)
    new_speakers = sorted({u.speaker for u in new_utterances})

    logger.info("Diarization complete: %d speakers detected", len(new_speakers))

    return Transcript(
        language=transcript.language,
        speakers=new_speakers,
        utterances=new_utterances,
        metadata=transcript.metadata,
    )

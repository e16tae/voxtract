"""Speaker diarization using pyannote.audio pipeline.

Assigns speaker labels to Qwen3-ASR utterances by running pyannote
diarization on the audio and matching speaker segments to word timestamps.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from voxtract.config import Settings, resolve_device
from voxtract.errors import SpeakerError
from voxtract.models import Transcript, Utterance

logger = logging.getLogger(__name__)

_HF_MODEL = "pyannote/speaker-diarization-3.1"


def _load_pipeline(device: str):
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

    try:
        pipeline = Pipeline.from_pretrained(_HF_MODEL)
    except Exception as exc:
        raise SpeakerError(
            f"Failed to load pyannote pipeline '{_HF_MODEL}': {exc}. "
            "You may need to accept the model terms at "
            f"https://huggingface.co/{_HF_MODEL} and set HF_TOKEN.",
            code="SPEAKER_MODEL_LOAD",
            recoverable=False,
        ) from exc

    if device.startswith("cuda"):
        import torch
        pipeline.to(torch.device(device))

    return pipeline


def _assign_speakers(
    utterances: list[Utterance],
    diarization,
) -> list[Utterance]:
    """Assign pyannote speaker labels to utterances by time overlap."""
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

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

        if best_overlap == 0.0:
            for seg_start, seg_end, speaker in segments:
                if seg_start <= mid <= seg_end:
                    best_speaker = speaker
                    break

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

    pipeline = _load_pipeline(device)

    # pyannote requires WAV; convert if needed
    audio_path = Path(audio_path)
    tmp_dir = None
    if audio_path.suffix.lower() not in (".wav", ".wave"):
        tmp_dir = tempfile.mkdtemp()
        wav_path = Path(tmp_dir) / f"{audio_path.stem}.wav"
        logger.info("Converting %s to WAV for pyannote", audio_path.suffix)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1",
             "-loglevel", "error", str(wav_path)],
            check=True, timeout=120,
        )
        audio_path = wav_path

    diarize_kwargs = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers

    try:
        result = pipeline(str(audio_path), **diarize_kwargs)
    except Exception as exc:
        raise SpeakerError(
            f"Pyannote diarization failed: {exc}",
            code="SPEAKER_DIARIZE",
            recoverable=True,
        ) from exc
    finally:
        if tmp_dir is not None:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # pyannote 4.x returns DiarizeOutput; use exclusive (no overlap) for ASR alignment
    if hasattr(result, "exclusive_speaker_diarization"):
        diarization = result.exclusive_speaker_diarization
    else:
        diarization = result

    new_utterances = _assign_speakers(transcript.utterances, diarization)
    new_utterances = _normalize_speaker_labels(new_utterances)
    new_speakers = sorted({u.speaker for u in new_utterances})

    logger.info("Diarization complete: %d speakers detected", len(new_speakers))

    return Transcript(
        language=transcript.language,
        speakers=new_speakers,
        utterances=new_utterances,
        metadata=transcript.metadata,
    )

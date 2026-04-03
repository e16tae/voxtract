"""Pipeline orchestrator: audio → transcribe → diarize → transcript JSON."""

from __future__ import annotations

import logging
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

from voxtract.config import get_settings
from voxtract.errors import SpeakerError
from voxtract.models import Transcript, Utterance

logger = logging.getLogger(__name__)

_CHUNK_THRESHOLD_MINUTES = 30


def _merge_chunk_transcripts(
    chunks_info: list,
    transcripts: list[Transcript],
) -> Transcript:
    """Merge per-chunk transcripts into one, adjusting timestamps and deduplicating overlap."""
    all_utterances: list[Utterance] = []

    for chunk, transcript in zip(chunks_info, transcripts):
        offset = chunk.start_time

        for utt in transcript.utterances:
            adjusted = Utterance(
                speaker=utt.speaker,
                start_time=utt.start_time + offset,
                end_time=utt.end_time + offset,
                text=utt.text,
                words=utt.words,
            )

            if all_utterances:
                last = all_utterances[-1]
                if adjusted.start_time <= last.end_time and (
                    adjusted.text == last.text
                    or SequenceMatcher(
                        None, adjusted.text, last.text,
                    ).ratio() >= 0.85
                ):
                    continue

            all_utterances.append(adjusted)

    all_utterances.sort(key=lambda u: u.start_time)

    lang = transcripts[0].language if transcripts else "auto"
    speakers = sorted({u.speaker for u in all_utterances})
    duration = all_utterances[-1].end_time if all_utterances else 0.0

    return Transcript(
        language=lang,
        speakers=speakers,
        utterances=all_utterances,
        metadata={
            "audio_file": str(chunks_info[0].audio_path) if chunks_info else "",
            "duration": duration,
            "chunks": len(transcripts),
        },
    )


def run_pipeline(
    *,
    audio_path: Path,
    output: Path | None = None,
    output_dir: Path | None = None,
    language: str | None = None,
    stt_provider: str | None = None,
    context: str | None = None,
    chunk_minutes: int | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict:
    """Run the full transcribe → diarize pipeline.

    Long audio is automatically split into chunks for STT to avoid GPU OOM.
    Speaker diarization (pyannote) runs on the full audio for global consistency.
    """
    from voxtract.audio.splitter import get_duration, convert_to_wav16k
    from voxtract.stt import get_provider as get_stt

    settings = get_settings()
    if context is not None:
        settings = settings.model_copy(update={"stt_context": context})

    chunk_min = chunk_minutes or settings.chunk_minutes
    stt = get_stt(stt_provider or settings.stt_provider, settings=settings)

    # Pre-convert to 16kHz mono WAV (benchmark-matching conditions)
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = convert_to_wav16k(
            audio_path, output_dir=Path(tmp_dir),
            normalize=settings.audio_normalize,
            highpass=settings.audio_highpass,
        )

        # Step 1: Transcribe (with auto-split for long audio)
        duration = get_duration(audio_path)
        handles_long = getattr(stt, "handles_long_audio", False)
        if duration > _CHUNK_THRESHOLD_MINUTES * 60 and not handles_long:
            transcript = _transcribe_chunked(
                wav_path, stt, language, chunk_min, settings.overlap_seconds,
            )
        else:
            transcript = stt.transcribe(wav_path, language=language)

        # Step 2: Diarize (WAV already in correct format, no re-conversion needed)
        if len(transcript.speakers) <= 1:
            try:
                from voxtract.speaker.diarizer import diarize_transcript
                transcript = diarize_transcript(
                    transcript, wav_path,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    settings=settings,
                )
            except ImportError:
                logger.warning("Speaker diarization not available (speaker extras not installed)")
            except SpeakerError as exc:
                logger.warning("Speaker diarization failed: %s, continuing without", exc)

    # Write output (outside temp dir context — WAV no longer needed)
    if output is not None:
        out_path = Path(output)
    elif output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{audio_path.stem}_transcript.json"
    else:
        out_path = Path(f"{audio_path.stem}_transcript.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")

    return {
        "status": "success",
        "step": "process",
        "output_file": str(out_path.resolve()),
        "metadata": {
            "speakers_detected": len(transcript.speakers),
            "language": transcript.language,
            "audio_duration_seconds": duration,
        },
    }


def _transcribe_chunked(
    audio_path: Path,
    stt,
    language: str | None,
    chunk_minutes: int,
    overlap_seconds: int,
) -> Transcript:
    """Split long audio into chunks, transcribe each, and merge."""
    from voxtract.audio.splitter import split_audio

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        chunks = split_audio(
            audio_path,
            output_dir=tmp,
            chunk_minutes=chunk_minutes,
            overlap_seconds=overlap_seconds,
        )
        logger.info(
            "Audio split into %d chunks (%d min each, %ds overlap)",
            len(chunks), chunk_minutes, overlap_seconds,
        )

        transcripts: list[Transcript] = []
        for i, chunk in enumerate(chunks):
            logger.info(
                "Transcribing chunk %d/%d [%.0f-%.0fs]",
                i + 1, len(chunks), chunk.start_time, chunk.end_time,
            )
            t = stt.transcribe(Path(chunk.audio_path), language=language)
            transcripts.append(t)

    return _merge_chunk_transcripts(chunks, transcripts)

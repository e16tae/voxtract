"""Split long audio files into overlapping chunks using ffmpeg."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from voxtract.errors import AudioError
from voxtract.models import ChunkInfo

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_MINUTES = 55
_DEFAULT_OVERLAP_SECONDS = 60


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def get_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise AudioError(
            f"Audio file not found: {audio_path}",
            code="AUDIO_FILE_NOT_FOUND",
            recoverable=False,
        )
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as exc:
        raise AudioError(
            f"Failed to get audio duration: {exc}",
            code="AUDIO_PROBE",
            recoverable=False,
        ) from exc


def _is_wav16k_mono(audio_path: Path) -> bool:
    """Check if audio is already 16kHz mono WAV via ffprobe."""
    if audio_path.suffix.lower() not in (".wav", ".wave"):
        return False
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=sample_rate,channels,codec_name",
             "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True, timeout=10,
        )
        parts = result.stdout.strip().split(",")
        return len(parts) >= 3 and parts[0] == "pcm_s16le" and parts[1] == "16000" and parts[2] == "1"
    except Exception:
        return False


def convert_to_wav16k(
    audio_path: Path,
    *,
    output_dir: Path,
    normalize: bool = False,
) -> Path:
    """Convert audio to 16kHz mono PCM WAV.

    Returns original path if already matching (and normalize=False).
    When normalize=True, applies EBU R128 loudness normalization via ffmpeg loudnorm.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise AudioError(
            f"Audio file not found: {audio_path}",
            code="AUDIO_FILE_NOT_FOUND",
            recoverable=False,
        )

    if not normalize and _is_wav16k_mono(audio_path):
        return audio_path

    if not check_ffmpeg():
        raise AudioError(
            "ffmpeg not found on PATH. Install it with: apt install ffmpeg",
            code="AUDIO_FFMPEG_MISSING",
            recoverable=False,
        )

    wav_path = output_dir / f"{audio_path.stem}.wav"
    af_filters = []
    if normalize:
        af_filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-ar", "16000", "-ac", "1",
    ]
    if af_filters:
        cmd += ["-af", ",".join(af_filters)]
    cmd += ["-c:a", "pcm_s16le", "-loglevel", "error", str(wav_path)]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AudioError(
            f"WAV conversion failed: {exc.stderr}",
            code="AUDIO_CONVERT",
            recoverable=False,
        ) from exc

    label = "16kHz mono WAV + loudnorm" if normalize else "16kHz mono WAV"
    logger.info("Converted %s → %s (%s)", audio_path.name, wav_path.name, label)
    return wav_path


def split_audio(
    audio_path: Path,
    *,
    output_dir: Path,
    chunk_minutes: int = _DEFAULT_CHUNK_MINUTES,
    overlap_seconds: int = _DEFAULT_OVERLAP_SECONDS,
) -> list[ChunkInfo]:
    """Split an audio file into overlapping chunks using ffmpeg -c copy."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise AudioError(
            f"Audio file not found: {audio_path}",
            code="AUDIO_FILE_NOT_FOUND",
            recoverable=False,
        )

    if not check_ffmpeg():
        raise AudioError(
            "ffmpeg not found on PATH. Install it with: apt install ffmpeg",
            code="AUDIO_FFMPEG_MISSING",
            recoverable=False,
        )

    duration = get_duration(audio_path)
    chunk_seconds = chunk_minutes * 60

    if duration <= chunk_seconds:
        return [
            ChunkInfo(
                index=0,
                start_time=0.0,
                end_time=duration,
                audio_path=str(audio_path),
            )
        ]

    chunks: list[ChunkInfo] = []
    start = 0.0
    index = 0
    suffix = audio_path.suffix

    while start < duration:
        end = min(start + chunk_seconds, duration)
        chunk_path = output_dir / f"{audio_path.stem}_chunk{index:03d}{suffix}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(audio_path),
            "-ss", str(start),
            "-t", str(end - start),
            "-c", "copy",
            "-loglevel", "error",
            str(chunk_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        except subprocess.CalledProcessError as exc:
            raise AudioError(
                f"ffmpeg failed for chunk {index}: {exc.stderr}",
                code="AUDIO_SPLIT",
                recoverable=False,
            ) from exc

        chunks.append(
            ChunkInfo(
                index=index,
                start_time=start,
                end_time=end,
                audio_path=str(chunk_path),
            )
        )

        start += chunk_seconds - overlap_seconds
        index += 1

    logger.info(
        "Split %s into %d chunks (%.0fs each, %ds overlap)",
        audio_path.name, len(chunks), chunk_seconds, overlap_seconds,
    )
    return chunks

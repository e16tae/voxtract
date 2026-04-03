"""Tests for ffmpeg-based audio splitting."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from voxtract.audio.splitter import split_audio, check_ffmpeg, get_duration
from voxtract.errors import AudioError
from voxtract.models import ChunkInfo


def _create_silent_mp3(path: Path, duration_seconds: float = 120.0) -> None:
    """Create a silent MP3 file using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono",
            "-t", str(duration_seconds),
            "-c:a", "libmp3lame", "-b:a", "32k",
            "-loglevel", "error",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture()
def short_mp3(tmp_path: Path) -> Path:
    """A 2-minute silent MP3 -- shorter than chunk size, should return 1 chunk."""
    p = tmp_path / "short.mp3"
    _create_silent_mp3(p, duration_seconds=120.0)
    return p


@pytest.fixture()
def long_mp3(tmp_path: Path) -> Path:
    """A 10-minute silent MP3 -- tests basic splitting."""
    p = tmp_path / "long.mp3"
    _create_silent_mp3(p, duration_seconds=600.0)
    return p


class TestCheckFfmpeg:
    def test_ffmpeg_available(self) -> None:
        """ffmpeg should be available on the test machine."""
        assert check_ffmpeg() is True


class TestGetDuration:
    def test_returns_correct_duration(self, short_mp3: Path) -> None:
        dur = get_duration(short_mp3)
        assert abs(dur - 120.0) < 1.0

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AudioError):
            get_duration(tmp_path / "nope.mp3")


class TestSplitAudio:
    def test_short_audio_returns_single_chunk(self, short_mp3: Path, tmp_path: Path) -> None:
        chunks = split_audio(short_mp3, output_dir=tmp_path, chunk_minutes=55)
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].start_time == 0.0

    def test_long_audio_splits_correctly(self, long_mp3: Path, tmp_path: Path) -> None:
        chunks = split_audio(long_mp3, output_dir=tmp_path, chunk_minutes=3, overlap_seconds=60)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert Path(chunk.audio_path).exists()

    def test_chunk_overlap_is_correct(self, long_mp3: Path, tmp_path: Path) -> None:
        chunks = split_audio(long_mp3, output_dir=tmp_path, chunk_minutes=3, overlap_seconds=60)
        if len(chunks) >= 2:
            assert chunks[1].start_time == chunks[0].end_time - 60

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AudioError, match="not found"):
            split_audio(tmp_path / "nope.mp3", output_dir=tmp_path)

    def test_chunks_have_nonzero_duration(self, long_mp3: Path, tmp_path: Path) -> None:
        chunks = split_audio(long_mp3, output_dir=tmp_path, chunk_minutes=3, overlap_seconds=60)
        for chunk in chunks:
            assert Path(chunk.audio_path).stat().st_size > 0


class TestConvertToWav16k:
    def test_converts_mp3_to_wav(self, short_mp3: Path, tmp_path: Path) -> None:
        from voxtract.audio.splitter import convert_to_wav16k

        wav_path = convert_to_wav16k(short_mp3, output_dir=tmp_path)

        assert wav_path.suffix == ".wav"
        assert wav_path.exists()
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=sample_rate,channels", "-of", "csv=p=0",
             str(wav_path)],
            capture_output=True, text=True,
        )
        parts = result.stdout.strip().split(",")
        assert parts[0] == "16000"
        assert parts[1] == "1"

    def test_wav_already_16k_mono_returns_original(self, tmp_path: Path) -> None:
        from voxtract.audio.splitter import convert_to_wav16k

        wav_path = tmp_path / "already.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
             "-t", "1", "-c:a", "pcm_s16le", "-loglevel", "error", str(wav_path)],
            check=True, capture_output=True,
        )

        result = convert_to_wav16k(wav_path, output_dir=tmp_path, normalize=False, highpass=False)
        assert result == wav_path

    def test_normalize_applies_loudnorm(self, short_mp3: Path, tmp_path: Path) -> None:
        from voxtract.audio.splitter import convert_to_wav16k

        wav_path = convert_to_wav16k(short_mp3, output_dir=tmp_path, normalize=True)
        assert wav_path.suffix == ".wav"
        assert wav_path.exists()
        # Verify it's valid 16kHz mono WAV
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=sample_rate,channels", "-of", "csv=p=0",
             str(wav_path)],
            capture_output=True, text=True,
        )
        parts = result.stdout.strip().split(",")
        assert parts[0] == "16000"
        assert parts[1] == "1"

    def test_normalize_forces_reconversion(self, tmp_path: Path) -> None:
        """Even if already 16k mono WAV, normalize=True should re-encode."""
        from voxtract.audio.splitter import convert_to_wav16k

        wav_path = tmp_path / "already.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
             "-t", "1", "-c:a", "pcm_s16le", "-loglevel", "error", str(wav_path)],
            check=True, capture_output=True,
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = convert_to_wav16k(wav_path, output_dir=out_dir, normalize=True)
        # Should NOT return original — must re-encode with loudnorm
        assert result != wav_path

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        from voxtract.audio.splitter import convert_to_wav16k

        with pytest.raises(AudioError, match="not found"):
            convert_to_wav16k(tmp_path / "nope.mp3", output_dir=tmp_path)

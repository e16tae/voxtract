"""Tests for audio pre-conversion in the pipeline."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from voxtract.models import Transcript


class TestPipelinePreconvert:
    def test_converts_audio_before_stt_and_diarize(self, tmp_path: Path) -> None:
        """Pipeline should convert audio to WAV 16k, then pass it to both STT and diarizer."""
        fake_transcript = Transcript(
            language="ko", speakers=["Speaker 0"], utterances=[], metadata={},
        )
        mock_provider = MagicMock()
        mock_provider.handles_long_audio = True
        mock_provider.transcribe.return_value = fake_transcript

        wav_path = tmp_path / "audio.wav"
        wav_path.touch()

        with patch("voxtract.stt.get_provider", return_value=mock_provider), \
             patch("voxtract.audio.splitter.get_duration", return_value=60.0), \
             patch("voxtract.audio.splitter.convert_to_wav16k", return_value=wav_path) as mock_convert, \
             patch("voxtract.speaker.diarizer.diarize_transcript", return_value=fake_transcript) as mock_diarize:

            from voxtract.pipeline import run_pipeline

            audio = tmp_path / "input.m4a"
            audio.touch()
            run_pipeline(audio_path=audio, output=tmp_path / "out.json")

            mock_convert.assert_called_once()
            # STT receives the WAV path
            mock_provider.transcribe.assert_called_once()
            stt_audio_arg = mock_provider.transcribe.call_args[0][0]
            assert Path(stt_audio_arg) == wav_path
            # Diarizer receives the WAV path
            diarize_audio_arg = mock_diarize.call_args[1].get("audio_path", mock_diarize.call_args[0][1])
            assert Path(diarize_audio_arg) == wav_path

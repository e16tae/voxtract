"""Tests for audio pre-conversion and chunk merging in the pipeline."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from voxtract.models import ChunkInfo, Transcript, Utterance


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
            # Normalize flag is forwarded from settings
            _, kwargs = mock_convert.call_args
            assert "normalize" in kwargs
            # STT receives the WAV path
            mock_provider.transcribe.assert_called_once()
            stt_audio_arg = mock_provider.transcribe.call_args[0][0]
            assert Path(stt_audio_arg) == wav_path
            # Diarizer receives the WAV path
            diarize_audio_arg = mock_diarize.call_args[1].get("audio_path", mock_diarize.call_args[0][1])
            assert Path(diarize_audio_arg) == wav_path


class TestPipelineVadFilter:
    def test_vad_filters_hallucinations_before_diarize(self, tmp_path: Path) -> None:
        """VAD should run between STT and diarization, removing non-speech utterances."""
        stt_transcript = Transcript(
            language="ko",
            speakers=["Speaker 0"],
            utterances=[
                Utterance(speaker="Speaker 0", start_time=0.0, end_time=5.0, text="real speech"),
                Utterance(speaker="Speaker 0", start_time=20.0, end_time=25.0, text="hallucination"),
            ],
            metadata={},
        )
        # After VAD filter, only "real speech" remains
        diarized = Transcript(
            language="ko",
            speakers=["Speaker 1"],
            utterances=[
                Utterance(speaker="Speaker 1", start_time=0.0, end_time=5.0, text="real speech"),
            ],
            metadata={},
        )

        mock_provider = MagicMock()
        mock_provider.handles_long_audio = True
        mock_provider.transcribe.return_value = stt_transcript

        wav_path = tmp_path / "audio.wav"
        wav_path.touch()

        with patch("voxtract.stt.get_provider", return_value=mock_provider), \
             patch("voxtract.audio.splitter.get_duration", return_value=60.0), \
             patch("voxtract.audio.splitter.convert_to_wav16k", return_value=wav_path), \
             patch("voxtract.audio.vad.get_speech_segments", return_value=[(0.0, 10.0)]) as mock_vad, \
             patch("voxtract.speaker.diarizer.diarize_transcript", return_value=diarized) as mock_diarize:

            from voxtract.pipeline import run_pipeline

            audio = tmp_path / "input.m4a"
            audio.touch()
            run_pipeline(audio_path=audio, output=tmp_path / "out.json")

            # VAD was called
            mock_vad.assert_called_once()
            # Diarizer received filtered transcript (1 utterance, not 2)
            diarize_call_transcript = mock_diarize.call_args[0][0]
            assert len(diarize_call_transcript.utterances) == 1
            assert diarize_call_transcript.utterances[0].text == "real speech"

    def test_vad_disabled_skips_filter(self, tmp_path: Path) -> None:
        """When vad_filter=False, VAD should not run."""
        from voxtract.config import Settings

        settings_no_vad = Settings(vad_filter=False)

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
             patch("voxtract.audio.splitter.convert_to_wav16k", return_value=wav_path), \
             patch("voxtract.audio.vad.get_speech_segments") as mock_vad, \
             patch("voxtract.speaker.diarizer.diarize_transcript", return_value=fake_transcript), \
             patch("voxtract.pipeline.get_settings", return_value=settings_no_vad):

            from voxtract.pipeline import run_pipeline

            audio = tmp_path / "input.m4a"
            audio.touch()
            run_pipeline(audio_path=audio, output=tmp_path / "out.json")

            mock_vad.assert_not_called()


class TestMergeChunkDedup:
    """Tests for fuzzy deduplication in chunk overlap merging."""

    def test_exact_duplicate_removed(self) -> None:
        """Same utterance in overlap region of two chunks → deduplicated."""
        from voxtract.pipeline import _merge_chunk_transcripts

        # Chunk 0: global 0-120s, Chunk 1: global 60-180s → overlap at 60-120s
        chunks = [
            ChunkInfo(index=0, start_time=0.0, end_time=120.0, audio_path="/tmp/c0"),
            ChunkInfo(index=1, start_time=60.0, end_time=180.0, audio_path="/tmp/c1"),
        ]
        # Utterance at global ~90s appears in both chunks
        transcripts = [
            Transcript(language="ko", speakers=["Speaker 0"], utterances=[
                Utterance(speaker="Speaker 0", start_time=90.0, end_time=95.0, text="안녕하세요"),
            ], metadata={}),
            Transcript(language="ko", speakers=["Speaker 0"], utterances=[
                # local 30-35s in chunk1 → global 90-95s (same utterance)
                Utterance(speaker="Speaker 0", start_time=30.0, end_time=35.0, text="안녕하세요"),
            ], metadata={}),
        ]

        result = _merge_chunk_transcripts(chunks, transcripts)
        texts = [u.text for u in result.utterances]
        assert texts.count("안녕하세요") == 1

    def test_fuzzy_duplicate_removed(self) -> None:
        """Near-identical STT output in overlap region → deduplicated via fuzzy match."""
        from voxtract.pipeline import _merge_chunk_transcripts

        chunks = [
            ChunkInfo(index=0, start_time=0.0, end_time=120.0, audio_path="/tmp/c0"),
            ChunkInfo(index=1, start_time=60.0, end_time=180.0, audio_path="/tmp/c1"),
        ]
        transcripts = [
            Transcript(language="ko", speakers=["Speaker 0"], utterances=[
                Utterance(speaker="Speaker 0", start_time=90.0, end_time=95.0,
                          text="안녕하세요 반갑습니다"),
            ], metadata={}),
            Transcript(language="ko", speakers=["Speaker 0"], utterances=[
                # local 30-35s → global 90-95s, minor STT variance (extra period)
                Utterance(speaker="Speaker 0", start_time=30.0, end_time=35.0,
                          text="안녕하세요. 반갑습니다"),
            ], metadata={}),
        ]

        result = _merge_chunk_transcripts(chunks, transcripts)
        assert len(result.utterances) == 1

    def test_different_utterances_not_removed(self) -> None:
        """Completely different utterances at same time → both kept."""
        from voxtract.pipeline import _merge_chunk_transcripts

        chunks = [
            ChunkInfo(index=0, start_time=0.0, end_time=120.0, audio_path="/tmp/c0"),
            ChunkInfo(index=1, start_time=60.0, end_time=180.0, audio_path="/tmp/c1"),
        ]
        transcripts = [
            Transcript(language="ko", speakers=["Speaker 0"], utterances=[
                Utterance(speaker="Speaker 0", start_time=90.0, end_time=95.0,
                          text="첫 번째 문장입니다"),
            ], metadata={}),
            Transcript(language="ko", speakers=["Speaker 0"], utterances=[
                Utterance(speaker="Speaker 0", start_time=30.0, end_time=35.0,
                          text="완전히 다른 문장입니다"),
            ], metadata={}),
        ]

        result = _merge_chunk_transcripts(chunks, transcripts)
        assert len(result.utterances) == 2

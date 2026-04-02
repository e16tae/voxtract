"""Tests for pyannote-based speaker diarization."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voxtract.models import Transcript, Utterance
from voxtract.speaker.diarizer import (
    _assign_word_speaker,
    _load_pipeline,
    _normalize_speaker_labels,
    diarize_transcript,
)


def _make_transcript(n_utterances: int = 6) -> Transcript:
    utts = []
    t = 0.0
    for i in range(n_utterances):
        utts.append(Utterance(
            speaker="Speaker 0",
            start_time=t,
            end_time=t + 10.0,
            text=f"발언 {i + 1}",
        ))
        t += 12.0
    return Transcript(language="ko", speakers=["Speaker 0"], utterances=utts)


class _FakeTurn:
    """Mimics pyannote Segment."""
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class _FakeDiarization:
    """Mimics pyannote Annotation."""
    def __init__(self, tracks: list[tuple[float, float, str]]):
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False):
        for start, end, speaker in self._tracks:
            yield _FakeTurn(start, end), None, speaker


class TestAssignWordSpeaker:
    def test_assigns_by_overlap(self) -> None:
        from voxtract.models import WordTimestamp

        word = WordTimestamp(text="A", start_time=0.0, end_time=10.0)
        segments = [
            (0.0, 11.0, "SPEAKER_00"),
            (11.5, 23.0, "SPEAKER_01"),
        ]

        result = _assign_word_speaker(word, segments)
        assert result == "SPEAKER_00"

    def test_midpoint_containment_fallback(self) -> None:
        """Zero-duration word inside a segment should match via midpoint."""
        from voxtract.models import WordTimestamp

        word = WordTimestamp(text="네", start_time=5.0, end_time=5.0)
        segments = [
            (0.0, 3.0, "SPEAKER_00"),
            (4.0, 8.0, "SPEAKER_01"),
        ]

        result = _assign_word_speaker(word, segments)
        assert result == "SPEAKER_01"

    def test_nearest_segment_in_gap(self) -> None:
        """Word in a gap between segments should match the nearest one."""
        from voxtract.models import WordTimestamp

        word = WordTimestamp(text="갭", start_time=3.1, end_time=3.9)
        segments = [
            (0.0, 3.0, "SPEAKER_00"),
            (5.0, 10.0, "SPEAKER_01"),
        ]

        result = _assign_word_speaker(word, segments)
        # mid=3.5, dist to (0,3)=0.5, dist to (5,10)=1.5 → SPEAKER_00 is nearer
        assert result == "SPEAKER_00"


class TestNormalizeSpeakerLabels:
    def test_renames_in_order(self) -> None:
        utts = [
            Utterance(speaker="SPEAKER_00", start_time=0.0, end_time=5.0, text="A"),
            Utterance(speaker="SPEAKER_01", start_time=5.0, end_time=10.0, text="B"),
            Utterance(speaker="SPEAKER_00", start_time=10.0, end_time=15.0, text="C"),
        ]

        result = _normalize_speaker_labels(utts)

        assert result[0].speaker == "Speaker 1"
        assert result[1].speaker == "Speaker 2"
        assert result[2].speaker == "Speaker 1"


class TestDiarizeTranscript:
    def test_empty_transcript(self) -> None:
        transcript = Transcript(language="ko", speakers=[], utterances=[])
        result = diarize_transcript(transcript, Path("/tmp/audio.wav"))
        assert len(result.utterances) == 0

    def test_full_pipeline_with_mock(self) -> None:
        transcript = _make_transcript(4)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = _FakeDiarization([
            (0.0, 22.0, "SPEAKER_00"),
            (24.0, 46.0, "SPEAKER_01"),
        ])

        with patch("voxtract.speaker.diarizer._load_pipeline", return_value=mock_pipeline), \
             patch("voxtract.speaker.diarizer.convert_to_wav16k", return_value=Path("/tmp/audio.wav")):
            result = diarize_transcript(transcript, Path("/tmp/audio.wav"))

        assert len(result.speakers) == 2
        assert result.utterances[0].speaker == "Speaker 1"
        assert result.utterances[2].speaker == "Speaker 2"


class TestSpeakerCountArgs:
    def test_min_max_speakers_passed(self) -> None:
        transcript = _make_transcript(4)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = _FakeDiarization([
            (0.0, 22.0, "SPEAKER_00"),
            (24.0, 46.0, "SPEAKER_01"),
        ])

        with patch("voxtract.speaker.diarizer._load_pipeline", return_value=mock_pipeline), \
             patch("voxtract.speaker.diarizer.convert_to_wav16k", return_value=Path("/tmp/audio.wav")):
            diarize_transcript(
                transcript, Path("/tmp/audio.wav"),
                min_speakers=2, max_speakers=5,
            )

        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["min_speakers"] == 2
        assert call_kwargs["max_speakers"] == 5


class TestLoadPipeline:
    def test_uses_configured_model(self) -> None:
        """_load_pipeline should use the model name from settings."""
        from voxtract.config import Settings

        settings = Settings(speaker_model="pyannote/speaker-diarization-community-1")

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = MagicMock()

        with patch.dict(
            "sys.modules",
            {"pyannote": MagicMock(), "pyannote.audio": MagicMock(Pipeline=mock_pipeline_cls)},
        ), patch("torch.device"):
            _load_pipeline("cpu", settings=settings)

        mock_pipeline_cls.from_pretrained.assert_called_once_with(
            "pyannote/speaker-diarization-community-1"
        )

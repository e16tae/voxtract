"""Tests for Voice Activity Detection filtering."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voxtract.audio.vad import (
    _speech_overlap_ratio,
    filter_utterances_by_vad,
)
from voxtract.models import Utterance


def _utt(start: float, end: float, text: str = "test") -> Utterance:
    return Utterance(speaker="Speaker 0", start_time=start, end_time=end, text=text)


class TestSpeechOverlapRatio:
    def test_full_overlap(self) -> None:
        utt = _utt(1.0, 5.0)
        segments = [(0.0, 10.0)]
        assert _speech_overlap_ratio(utt, segments) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        utt = _utt(10.0, 15.0)
        segments = [(0.0, 5.0)]
        assert _speech_overlap_ratio(utt, segments) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        utt = _utt(4.0, 8.0)  # 4s duration
        segments = [(0.0, 6.0)]  # overlaps 4.0-6.0 = 2s
        assert _speech_overlap_ratio(utt, segments) == pytest.approx(0.5)

    def test_multiple_segments(self) -> None:
        utt = _utt(0.0, 10.0)  # 10s duration
        segments = [(0.0, 3.0), (7.0, 10.0)]  # 3s + 3s = 6s overlap
        assert _speech_overlap_ratio(utt, segments) == pytest.approx(0.6)

    def test_zero_duration_returns_one(self) -> None:
        utt = _utt(5.0, 5.0)
        segments = [(0.0, 10.0)]
        assert _speech_overlap_ratio(utt, segments) == pytest.approx(1.0)


class TestFilterUtterancesByVad:
    def test_keeps_speech_utterances(self) -> None:
        utts = [_utt(0.0, 5.0, "real speech")]
        segments = [(0.0, 10.0)]
        result = filter_utterances_by_vad(utts, segments)
        assert len(result) == 1

    def test_drops_non_speech_utterances(self) -> None:
        utts = [_utt(20.0, 25.0, "hallucination")]
        segments = [(0.0, 10.0)]
        result = filter_utterances_by_vad(utts, segments)
        assert len(result) == 0

    def test_mixed_keeps_and_drops(self) -> None:
        utts = [
            _utt(0.0, 5.0, "real"),
            _utt(15.0, 20.0, "hallucination"),
            _utt(25.0, 30.0, "also real"),
        ]
        segments = [(0.0, 10.0), (24.0, 35.0)]
        result = filter_utterances_by_vad(utts, segments)
        assert [u.text for u in result] == ["real", "also real"]

    def test_empty_segments_keeps_all(self) -> None:
        utts = [_utt(0.0, 5.0), _utt(10.0, 15.0)]
        result = filter_utterances_by_vad(utts, [])
        assert len(result) == 2

    def test_custom_threshold(self) -> None:
        utt = _utt(0.0, 10.0)
        segments = [(0.0, 3.0)]  # 30% overlap
        # Default threshold 0.5 → drop
        assert len(filter_utterances_by_vad([utt], segments)) == 0
        # Lower threshold 0.2 → keep
        assert len(filter_utterances_by_vad([utt], segments, min_overlap=0.2)) == 1

    def test_borderline_overlap_at_threshold(self) -> None:
        utt = _utt(0.0, 10.0)  # 10s
        segments = [(0.0, 5.0)]  # 50% overlap, exactly at threshold
        result = filter_utterances_by_vad([utt], segments, min_overlap=0.5)
        assert len(result) == 1  # >= threshold → keep


class TestGetSpeechSegments:
    def test_returns_segments_from_pyannote(self) -> None:
        """get_speech_segments should return (start, end) tuples from VAD pipeline."""
        from voxtract.audio.vad import get_speech_segments

        class _FakeTurn:
            def __init__(self, start: float, end: float):
                self.start = start
                self.end = end

        class _FakeVADResult:
            def itertracks(self, yield_label=False):
                yield _FakeTurn(0.0, 5.0), None, "SPEECH"
                yield _FakeTurn(8.0, 15.0), None, "SPEECH"

        mock_model = MagicMock()
        mock_vad_pipeline_cls = MagicMock()
        mock_vad_pipeline_instance = MagicMock()
        mock_vad_pipeline_instance.return_value = _FakeVADResult()
        mock_vad_pipeline_cls.return_value = mock_vad_pipeline_instance

        mock_pyannote_audio = MagicMock()
        mock_pyannote_audio.Model.from_pretrained.return_value = mock_model

        mock_pipelines = MagicMock()
        mock_pipelines.VoiceActivityDetection = mock_vad_pipeline_cls

        with patch.dict("sys.modules", {
            "pyannote": MagicMock(),
            "pyannote.audio": mock_pyannote_audio,
            "pyannote.audio.pipelines": mock_pipelines,
        }):
            import importlib
            import voxtract.audio.vad as vad_mod
            importlib.reload(vad_mod)
            segments = vad_mod.get_speech_segments(Path("/tmp/audio.wav"), device="cpu")

        assert segments == [(0.0, 5.0), (8.0, 15.0)]

    def test_raises_on_missing_pyannote(self) -> None:
        """Should raise AudioError when pyannote is not installed."""
        from voxtract.errors import AudioError

        with patch.dict("sys.modules", {
            "pyannote": None,
            "pyannote.audio": None,
            "pyannote.audio.pipelines": None,
        }):
            import importlib
            import voxtract.audio.vad as vad_mod
            importlib.reload(vad_mod)
            with pytest.raises(AudioError, match="not installed"):
                vad_mod.get_speech_segments(Path("/tmp/audio.wav"), device="cpu")

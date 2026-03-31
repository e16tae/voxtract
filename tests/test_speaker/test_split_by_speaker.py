"""Tests for splitting utterances at speaker change boundaries."""
from __future__ import annotations

import pytest

from voxtract.models import Utterance


class TestSplitBySpeakerChange:
    def test_no_split_when_single_speaker(self) -> None:
        """Utterance fully within one speaker segment -> no split."""
        from voxtract.speaker.diarizer import _split_by_speaker_change

        utts = [
            Utterance(speaker="SPEAKER_00", start_time=0.0, end_time=10.0, text="A B C"),
        ]
        segments = [(0.0, 15.0, "SPEAKER_00")]

        result = _split_by_speaker_change(utts, segments)
        assert len(result) == 1
        assert result[0].text == "A B C"

    def test_splits_at_speaker_boundary(self) -> None:
        """Utterance spanning two speaker segments -> split into two."""
        from voxtract.speaker.diarizer import _split_by_speaker_change

        utts = [
            Utterance(speaker="SPEAKER_00", start_time=0.0, end_time=20.0, text="A B C D"),
        ]
        # Speaker changes at 10.0s
        segments = [
            (0.0, 10.0, "SPEAKER_00"),
            (10.0, 25.0, "SPEAKER_01"),
        ]

        result = _split_by_speaker_change(utts, segments)
        assert len(result) == 2
        assert result[0].speaker == "SPEAKER_00"
        assert result[0].end_time == 10.0
        assert "A" in result[0].text
        assert result[1].speaker == "SPEAKER_01"
        assert result[1].start_time == 10.0
        assert "D" in result[1].text
        # All words accounted for
        all_words = result[0].text.split() + result[1].text.split()
        assert set(all_words) == {"A", "B", "C", "D"}

    def test_preserves_short_utterances(self) -> None:
        """Short utterance (<2s) should not be split even if spanning boundary."""
        from voxtract.speaker.diarizer import _split_by_speaker_change

        utts = [
            Utterance(speaker="SPEAKER_00", start_time=9.0, end_time=10.5, text="네"),
        ]
        segments = [
            (0.0, 10.0, "SPEAKER_00"),
            (10.0, 20.0, "SPEAKER_01"),
        ]

        result = _split_by_speaker_change(utts, segments)
        assert len(result) == 1  # Too short to split

    def test_empty_input(self) -> None:
        from voxtract.speaker.diarizer import _split_by_speaker_change

        result = _split_by_speaker_change([], [])
        assert result == []

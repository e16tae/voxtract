"""Tests for splitting utterances at speaker change boundaries."""
from __future__ import annotations

import pytest

from voxtract.models import Utterance, WordTimestamp


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

    def test_merges_consecutive_same_speaker(self) -> None:
        """Consecutive segments with the same speaker should be merged, not split."""
        from voxtract.speaker.diarizer import _split_by_speaker_change

        utts = [
            Utterance(speaker="SPEAKER_00", start_time=0.0, end_time=15.0, text="A B C D E"),
        ]
        # Two adjacent segments for SPEAKER_00, then one for SPEAKER_01
        segments = [
            (0.0, 5.0, "SPEAKER_00"),
            (5.0, 10.0, "SPEAKER_00"),
            (10.0, 15.0, "SPEAKER_01"),
        ]

        result = _split_by_speaker_change(utts, segments)
        # Should merge the two SPEAKER_00 segments → only 2 parts, not 3
        assert len(result) == 2
        assert result[0].speaker == "SPEAKER_00"
        assert result[1].speaker == "SPEAKER_01"

    def test_gap_does_not_skew_word_distribution(self) -> None:
        """Words should be distributed by overlap ratio, not biased by gaps."""
        from voxtract.speaker.diarizer import _split_by_speaker_change

        utts = [
            Utterance(
                speaker="SPEAKER_00", start_time=0.0, end_time=10.0,
                text="A B C D E F G H I J",
            ),
        ]
        # Equal overlap durations (4s each), with a 2s gap in the middle
        segments = [
            (0.0, 4.0, "SPEAKER_00"),
            (6.0, 10.0, "SPEAKER_01"),
        ]

        result = _split_by_speaker_change(utts, segments)
        assert len(result) == 2
        # Equal overlap → approximately equal word count (5 each)
        assert len(result[0].text.split()) == 5
        assert len(result[1].text.split()) == 5

    def test_empty_input(self) -> None:
        from voxtract.speaker.diarizer import _split_by_speaker_change

        result = _split_by_speaker_change([], [])
        assert result == []


class TestSplitByWordTimestamps:
    """Tests for word-timestamp-based splitting (precise boundaries)."""

    def test_splits_by_word_timestamps(self) -> None:
        """With word timestamps, words are assigned by their actual time position."""
        from voxtract.speaker.diarizer import _split_by_speaker_change

        utts = [
            Utterance(
                speaker="SPEAKER_00",
                start_time=0.0,
                end_time=10.0,
                text="A B C D",
                words=[
                    WordTimestamp(text="A", start_time=0.0, end_time=1.5),
                    WordTimestamp(text="B", start_time=2.0, end_time=3.5),
                    WordTimestamp(text="C", start_time=5.5, end_time=7.0),
                    WordTimestamp(text="D", start_time=7.5, end_time=9.5),
                ],
            ),
        ]
        # Speaker changes at 5.0s — A and B before, C and D after
        segments = [
            (0.0, 5.0, "SPEAKER_00"),
            (5.0, 10.0, "SPEAKER_01"),
        ]

        result = _split_by_speaker_change(utts, segments)
        assert len(result) == 2
        assert result[0].text == "A B"
        assert result[0].speaker == "SPEAKER_00"
        assert result[1].text == "C D"
        assert result[1].speaker == "SPEAKER_01"

    def test_word_timestamps_uneven_split(self) -> None:
        """Word timestamps should produce different split than ratio-based."""
        from voxtract.speaker.diarizer import _split_by_speaker_change

        # Scenario: speaker boundary at 3.0s, but word B spans 1.0–4.0
        # (midpoint 2.5, overlap with seg0=2.0, overlap with seg1=1.0 → goes to seg0)
        utts = [
            Utterance(
                speaker="SPEAKER_00",
                start_time=0.0,
                end_time=6.0,
                text="A B C",
                words=[
                    WordTimestamp(text="A", start_time=0.0, end_time=0.8),
                    WordTimestamp(text="B", start_time=1.0, end_time=4.0),
                    WordTimestamp(text="C", start_time=4.5, end_time=5.8),
                ],
            ),
        ]
        segments = [
            (0.0, 3.0, "SPEAKER_00"),
            (3.0, 6.0, "SPEAKER_01"),
        ]

        result = _split_by_speaker_change(utts, segments)
        assert len(result) == 2
        # B's overlap with SPEAKER_00 (1.0–3.0=2s) > overlap with SPEAKER_01 (3.0–4.0=1s)
        assert result[0].text == "A B"
        assert result[1].text == "C"

    def test_word_timestamps_preserved_in_output(self) -> None:
        """Split utterances should carry their word timestamps."""
        from voxtract.speaker.diarizer import _split_by_speaker_change

        utts = [
            Utterance(
                speaker="SPEAKER_00",
                start_time=0.0,
                end_time=10.0,
                text="A B",
                words=[
                    WordTimestamp(text="A", start_time=0.0, end_time=3.0),
                    WordTimestamp(text="B", start_time=6.0, end_time=9.0),
                ],
            ),
        ]
        segments = [
            (0.0, 5.0, "SPEAKER_00"),
            (5.0, 10.0, "SPEAKER_01"),
        ]

        result = _split_by_speaker_change(utts, segments)
        assert len(result) == 2
        assert result[0].words is not None
        assert len(result[0].words) == 1
        assert result[0].words[0].text == "A"
        assert result[1].words is not None
        assert len(result[1].words) == 1
        assert result[1].words[0].text == "B"

    def test_fallback_when_no_words(self) -> None:
        """Without word timestamps, ratio-based fallback should still work."""
        from voxtract.speaker.diarizer import _split_by_speaker_change

        utts = [
            Utterance(
                speaker="SPEAKER_00",
                start_time=0.0,
                end_time=20.0,
                text="A B C D",
                # words is None (default)
            ),
        ]
        segments = [
            (0.0, 10.0, "SPEAKER_00"),
            (10.0, 25.0, "SPEAKER_01"),
        ]

        result = _split_by_speaker_change(utts, segments)
        assert len(result) == 2
        all_words = result[0].text.split() + result[1].text.split()
        assert set(all_words) == {"A", "B", "C", "D"}

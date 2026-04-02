"""Tests for word-level speaker assignment and utterance reconstruction."""
from __future__ import annotations

import pytest

from voxtract.models import Utterance, WordTimestamp


class TestBuildUtterancesFromWords:
    """Word-level speaker assignment should replace utterance-level assignment."""

    def test_words_at_speaker_boundary_assigned_correctly(self) -> None:
        """Words near a speaker boundary should be assigned by overlap, not utterance."""
        from voxtract.speaker.diarizer import _build_utterances_from_words

        # Simulates the 29-32s problem:
        # STT produced one long utterance spanning a speaker change.
        # Words "이런 형식 이에요 센서가" should all go to SPEAKER_01
        # because they overlap with SPEAKER_01's segment.
        utts = [
            Utterance(
                speaker="Speaker 0",
                start_time=14.0,
                end_time=50.0,
                text="저희 가 이런 형식 이에요 센서가 있고",
                words=[
                    WordTimestamp(text="저희", start_time=14.0, end_time=14.5),
                    WordTimestamp(text="가", start_time=14.5, end_time=14.8),
                    WordTimestamp(text="이런", start_time=29.7, end_time=30.0),
                    WordTimestamp(text="형식", start_time=30.0, end_time=30.3),
                    WordTimestamp(text="이에요", start_time=30.3, end_time=30.6),
                    WordTimestamp(text="센서가", start_time=30.7, end_time=31.2),
                    WordTimestamp(text="있고", start_time=31.4, end_time=31.8),
                ],
            ),
        ]
        segments = [
            (14.0, 29.5, "SPEAKER_00"),
            (29.5, 31.3, "SPEAKER_01"),
            (31.3, 50.0, "SPEAKER_00"),
        ]

        result = _build_utterances_from_words(utts, segments)

        # "저희 가" → SPEAKER_00
        # "이런 형식 이에요 센서가" → SPEAKER_01 (all overlap with 29.5-31.3)
        # "있고" → SPEAKER_00
        assert len(result) == 3
        assert result[0].speaker == "SPEAKER_00"
        assert "저희" in result[0].text
        assert result[1].speaker == "SPEAKER_01"
        assert "이런" in result[1].text
        assert "형식" in result[1].text
        assert "이에요" in result[1].text
        assert "센서가" in result[1].text
        assert result[2].speaker == "SPEAKER_00"
        assert "있고" in result[2].text

    def test_no_single_word_orphan_utterances(self) -> None:
        """Consecutive words with same speaker should be grouped, not isolated."""
        from voxtract.speaker.diarizer import _build_utterances_from_words

        # Simulates the 814-816s problem:
        # "게 약간 트랜지스터랑 비슷하게 생겼거" are all from same speaker
        utts = [
            Utterance(
                speaker="Speaker 0",
                start_time=813.0,
                end_time=821.0,
                text="게 약간 트랜지스터랑 비슷 하게 생겼 거",
                words=[
                    WordTimestamp(text="게", start_time=814.0, end_time=814.3),
                    WordTimestamp(text="약간", start_time=814.7, end_time=815.0),
                    WordTimestamp(text="트랜지스터랑", start_time=815.0, end_time=815.6),
                    WordTimestamp(text="비슷", start_time=815.7, end_time=816.0),
                    WordTimestamp(text="하게", start_time=816.0, end_time=816.1),
                    WordTimestamp(text="생겼", start_time=816.1, end_time=816.3),
                    WordTimestamp(text="거", start_time=816.4, end_time=816.4),
                ],
            ),
        ]
        # All words fall within one speaker segment
        segments = [
            (810.0, 813.5, "SPEAKER_00"),
            (813.5, 821.0, "SPEAKER_01"),
        ]

        result = _build_utterances_from_words(utts, segments)

        # All words should be one utterance for SPEAKER_01
        assert len(result) == 1
        assert result[0].speaker == "SPEAKER_01"
        assert "게 약간 트랜지스터랑" in result[0].text

    def test_pause_threshold_creates_new_utterance(self) -> None:
        """Long pause between same-speaker words should start a new utterance."""
        from voxtract.speaker.diarizer import _build_utterances_from_words

        utts = [
            Utterance(
                speaker="Speaker 0",
                start_time=0.0,
                end_time=10.0,
                text="안녕 하세요 반갑습니다",
                words=[
                    WordTimestamp(text="안녕", start_time=0.0, end_time=0.5),
                    WordTimestamp(text="하세요", start_time=0.5, end_time=1.0),
                    # 2.0s gap — exceeds 1.5s pause threshold
                    WordTimestamp(text="반갑습니다", start_time=3.0, end_time=4.0),
                ],
            ),
        ]
        segments = [(0.0, 10.0, "SPEAKER_00")]

        result = _build_utterances_from_words(utts, segments)

        assert len(result) == 2
        assert result[0].text == "안녕 하세요"
        assert result[1].text == "반갑습니다"
        assert result[0].speaker == result[1].speaker == "SPEAKER_00"

    def test_word_timestamps_preserved(self) -> None:
        """Each output utterance should carry its word timestamps."""
        from voxtract.speaker.diarizer import _build_utterances_from_words

        utts = [
            Utterance(
                speaker="Speaker 0",
                start_time=0.0,
                end_time=10.0,
                text="A B C",
                words=[
                    WordTimestamp(text="A", start_time=0.0, end_time=1.0),
                    WordTimestamp(text="B", start_time=5.0, end_time=6.0),
                    WordTimestamp(text="C", start_time=6.5, end_time=7.5),
                ],
            ),
        ]
        segments = [
            (0.0, 4.0, "SPEAKER_00"),
            (4.0, 10.0, "SPEAKER_01"),
        ]

        result = _build_utterances_from_words(utts, segments)

        assert len(result) == 2
        assert result[0].words is not None
        assert len(result[0].words) == 1
        assert result[0].words[0].text == "A"
        assert result[1].words is not None
        assert len(result[1].words) == 2

    def test_utterance_times_from_word_times(self) -> None:
        """Utterance start/end should come from actual word timestamps, not segments."""
        from voxtract.speaker.diarizer import _build_utterances_from_words

        utts = [
            Utterance(
                speaker="Speaker 0",
                start_time=0.0,
                end_time=10.0,
                text="A B",
                words=[
                    WordTimestamp(text="A", start_time=1.0, end_time=2.0),
                    WordTimestamp(text="B", start_time=3.0, end_time=4.0),
                ],
            ),
        ]
        segments = [(0.0, 10.0, "SPEAKER_00")]

        result = _build_utterances_from_words(utts, segments)

        assert result[0].start_time == 1.0  # from first word
        assert result[0].end_time == 4.0  # from last word

    def test_multiple_utterances_flattened(self) -> None:
        """Words from multiple STT utterances should be flattened and re-grouped."""
        from voxtract.speaker.diarizer import _build_utterances_from_words

        utts = [
            Utterance(
                speaker="Speaker 0",
                start_time=0.0,
                end_time=3.0,
                text="A B",
                words=[
                    WordTimestamp(text="A", start_time=0.0, end_time=1.0),
                    WordTimestamp(text="B", start_time=1.5, end_time=2.5),
                ],
            ),
            Utterance(
                speaker="Speaker 0",
                start_time=4.0,
                end_time=7.0,
                text="C D",
                words=[
                    WordTimestamp(text="C", start_time=4.0, end_time=5.0),
                    WordTimestamp(text="D", start_time=5.5, end_time=6.5),
                ],
            ),
        ]
        # One speaker for all
        segments = [(0.0, 10.0, "SPEAKER_00")]

        result = _build_utterances_from_words(utts, segments)

        # Gap between B(2.5) and C(4.0) = 1.5s → exactly at threshold, stays same utterance
        # (threshold is strictly >, so 1.5 does not break)
        assert len(result) == 1
        assert result[0].text == "A B C D"

    def test_fallback_without_word_timestamps(self) -> None:
        """Without word timestamps, fall back to utterance-level assignment."""
        from voxtract.speaker.diarizer import _build_utterances_from_words

        utts = [
            Utterance(speaker="Speaker 0", start_time=0.0, end_time=5.0, text="A B"),
            Utterance(speaker="Speaker 0", start_time=6.0, end_time=11.0, text="C D"),
        ]
        segments = [
            (0.0, 5.5, "SPEAKER_00"),
            (5.5, 12.0, "SPEAKER_01"),
        ]

        result = _build_utterances_from_words(utts, segments)

        assert len(result) == 2
        assert result[0].speaker == "SPEAKER_00"
        assert result[1].speaker == "SPEAKER_01"

    def test_empty_input(self) -> None:
        from voxtract.speaker.diarizer import _build_utterances_from_words

        result = _build_utterances_from_words([], [])
        assert result == []

    def test_no_segments(self) -> None:
        """With no diarizer segments, all words default to Speaker 0."""
        from voxtract.speaker.diarizer import _build_utterances_from_words

        utts = [
            Utterance(
                speaker="Speaker 0",
                start_time=0.0,
                end_time=3.0,
                text="A B",
                words=[
                    WordTimestamp(text="A", start_time=0.0, end_time=1.0),
                    WordTimestamp(text="B", start_time=1.5, end_time=2.5),
                ],
            ),
        ]

        result = _build_utterances_from_words(utts, [])

        assert len(result) == 1
        assert result[0].speaker == "Speaker 0"

"""Tests for Pydantic data models — serialization/deserialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

from voxtract.models import (
    ChunkInfo,
    Transcript,
    Utterance,
)


class TestUtterance:
    def test_roundtrip(self) -> None:
        u = Utterance(speaker="Speaker 1", start_time=0.0, end_time=5.5, text="안녕하세요")
        data = json.loads(u.model_dump_json())
        restored = Utterance(**data)
        assert restored == u

    def test_required_fields(self) -> None:
        with pytest.raises(Exception):
            Utterance(speaker="Speaker 1")  # type: ignore[call-arg]


class TestTranscript:
    def test_roundtrip(self) -> None:
        t = Transcript(
            language="ko",
            speakers=["Speaker 1", "Speaker 2"],
            utterances=[
                Utterance(speaker="Speaker 1", start_time=0.0, end_time=3.0, text="첫 번째 발언"),
                Utterance(speaker="Speaker 2", start_time=3.5, end_time=7.0, text="두 번째 발언"),
            ],
            metadata={"duration": 7.0, "audio_file": "meeting.mp3"},
        )
        json_str = t.model_dump_json()
        restored = Transcript.model_validate_json(json_str)
        assert restored == t
        assert len(restored.utterances) == 2

    def test_default_metadata(self) -> None:
        t = Transcript(language="en", speakers=[])
        assert t.metadata == {}
        assert t.utterances == []

    def test_from_fixture(self) -> None:
        fixture_path = FIXTURES_DIR / "sample_transcript.json"
        with open(fixture_path, encoding="utf-8") as f:
            data = json.load(f)
        t = Transcript(**data)
        assert t.language == "ko"
        assert len(t.speakers) >= 1


class TestChunkInfo:
    def test_roundtrip(self) -> None:
        c = ChunkInfo(
            index=0,
            start_time=0.0,
            end_time=3360.0,
            audio_path="/tmp/chunk_0.mp3",
        )
        data = json.loads(c.model_dump_json())
        restored = ChunkInfo(**data)
        assert restored.index == 0
        assert restored.end_time == 3360.0

    def test_duration(self) -> None:
        c = ChunkInfo(index=0, start_time=0.0, end_time=3360.0, audio_path="/tmp/c.mp3")
        assert c.duration == 3360.0

"""Output formatters for Transcript — plain text, JSON, etc."""

from __future__ import annotations

from pathlib import Path

from voxtract.models import Transcript


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format (or H:MM:SS if >= 1 hour)."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def to_text(transcript: Transcript) -> str:
    """Format transcript as human-readable plain text.

    Example output:

        참석자 1 00:00
        법이 잘 있는 거 매출 취소가 되든지 ...

        참석자 2 00:47
        지금 앞에서 김수희 실장이랑 ...
    """
    lines: list[str] = []
    for utt in transcript.utterances:
        ts = _format_timestamp(utt.start_time)
        lines.append(f"{utt.speaker} {ts}")
        lines.append(utt.text)
        lines.append("")  # blank line between utterances
    return "\n".join(lines).rstrip("\n") + "\n"


def write_transcript(transcript: Transcript, path: Path, fmt: str = "json") -> None:
    """Write transcript to file in the requested format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "txt":
        path.write_text(to_text(transcript), encoding="utf-8")
    else:
        path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")

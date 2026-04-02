# Diarization Accuracy Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve speaker diarization accuracy by upgrading pyannote model, exposing speaker count controls, and splitting utterances at speaker change boundaries.

**Architecture:** Three sequential improvements (Tasks 2 and 3 depend on Task 1's changes to `_load_pipeline` and `diarize_transcript`): (1) upgrade pyannote pipeline from 3.1 to community-1 with configurable model name, (2) wire min_speakers/max_speakers from CLI through pipeline to diarizer (num_speakers already exists), (3) add post-diarization utterance splitting at speaker change points within a single STT utterance.

**Tech Stack:** pyannote.audio 4.0+ (community-1 model), click CLI, pydantic-settings, pytest

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/voxtract/config.py` | Add `speaker_model` setting |
| Modify | `src/voxtract/speaker/diarizer.py` | Upgrade model name, accept min/max speakers, add `_split_by_speaker_change()` |
| Modify | `src/voxtract/pipeline.py` | Pass speaker count args to diarizer |
| Modify | `src/voxtract/cli.py` | Add `--num-speakers`, `--min-speakers`, `--max-speakers` options |
| Modify | `tests/test_speaker/test_diarizer.py` | Test new features |
| Create | `tests/test_speaker/test_split_by_speaker.py` | Test speaker-change splitting |

---

### Task 1: Upgrade pyannote model to community-1 with configurable model name

**Files:**
- Modify: `src/voxtract/config.py:10-29`
- Modify: `src/voxtract/speaker/diarizer.py:19,22-50`
- Modify: `tests/test_speaker/test_diarizer.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_speaker/test_diarizer.py`:

```python
class TestLoadPipeline:
    def test_uses_configured_model(self) -> None:
        """_load_pipeline should use the model name from settings."""
        from voxtract.config import Settings

        settings = Settings(speaker_model="pyannote/speaker-diarization-community-1")

        with patch("voxtract.speaker.diarizer.Pipeline") as mock_pipeline_cls:
            mock_pipeline_cls.from_pretrained.return_value = MagicMock()
            from voxtract.speaker.diarizer import _load_pipeline
            _load_pipeline("cpu", settings=settings)

            mock_pipeline_cls.from_pretrained.assert_called_once_with(
                "pyannote/speaker-diarization-community-1"
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_speaker/test_diarizer.py::TestLoadPipeline -v`
Expected: FAIL (Settings has no `speaker_model` field, `_load_pipeline` doesn't accept `settings`)

- [ ] **Step 3: Add `speaker_model` to Settings**

In `src/voxtract/config.py`, add to the `Settings` class after the STT section:

```python
    # Speaker diarization
    speaker_model: str = "pyannote/speaker-diarization-community-1"
```

- [ ] **Step 4: Update `_load_pipeline` to accept settings and use configurable model**

In `src/voxtract/speaker/diarizer.py`, remove the module-level `_HF_MODEL` constant and change `_load_pipeline`:

```python
def _load_pipeline(device: str, *, settings: Settings | None = None):
    """Load pyannote diarization pipeline onto the given device."""
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        raise SpeakerError(
            "pyannote.audio is not installed. "
            "Install it with: uv pip install pyannote.audio",
            code="SPEAKER_DEPENDENCY_MISSING",
            recoverable=False,
        )

    if settings is None:
        from voxtract.config import get_settings
        settings = get_settings()

    model_name = settings.speaker_model

    try:
        pipeline = Pipeline.from_pretrained(model_name)
    except Exception as exc:
        raise SpeakerError(
            f"Failed to load pyannote pipeline '{model_name}': {exc}. "
            "You may need to accept the model terms at "
            f"https://huggingface.co/{model_name} and set HF_TOKEN.",
            code="SPEAKER_MODEL_LOAD",
            recoverable=False,
        ) from exc

    if device.startswith("cuda"):
        import torch
        pipeline.to(torch.device(device))

    return pipeline
```

Also update the call site in `diarize_transcript()` (line 130):

```python
    pipeline = _load_pipeline(device, settings=settings)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_speaker/test_diarizer.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/voxtract/config.py src/voxtract/speaker/diarizer.py tests/test_speaker/test_diarizer.py
git commit -m "feat: upgrade pyannote model to community-1 with configurable speaker_model setting"
```

---

### Task 2: Add num_speakers / min_speakers / max_speakers to CLI and pipeline

**Files:**
- Modify: `src/voxtract/cli.py:90-136`
- Modify: `src/voxtract/pipeline.py:63-112`
- Modify: `src/voxtract/speaker/diarizer.py:113-180`
- Modify: `tests/test_speaker/test_diarizer.py`
- Modify: `tests/test_stt/test_qwen3.py` (TestPipelineChunkingSkip)

- [ ] **Step 1: Write failing test**

Add to `tests/test_speaker/test_diarizer.py`:

```python
class TestSpeakerCountArgs:
    def test_min_max_speakers_passed(self) -> None:
        transcript = _make_transcript(4)

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = _FakeDiarization([
            (0.0, 22.0, "SPEAKER_00"),
            (24.0, 46.0, "SPEAKER_01"),
        ])

        with patch("voxtract.speaker.diarizer._load_pipeline", return_value=mock_pipeline):
            diarize_transcript(
                transcript, Path("/tmp/audio.wav"),
                min_speakers=2, max_speakers=5,
            )

        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["min_speakers"] == 2
        assert call_kwargs["max_speakers"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_speaker/test_diarizer.py::TestSpeakerCountArgs -v`
Expected: FAIL (`diarize_transcript` doesn't accept `min_speakers` or `max_speakers` — note: `num_speakers` already exists in the current signature)

- [ ] **Step 3: Add min_speakers/max_speakers to diarize_transcript()**

In `src/voxtract/speaker/diarizer.py`, update `diarize_transcript` signature and kwargs building:

```python
def diarize_transcript(
    transcript: Transcript,
    audio_path: Path,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    settings: Settings | None = None,
) -> Transcript:
```

Replace the existing kwargs section (currently lines 146-148 which only handles `num_speakers`) with this expanded version:

```python
    diarize_kwargs = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers
```

- [ ] **Step 4: Add CLI options to `process` command**

In `src/voxtract/cli.py`, add three options to the `process` command (after `--chunk-minutes`):

```python
@click.option("--num-speakers", type=int, default=None, help="Exact number of speakers (if known).")
@click.option("--min-speakers", type=int, default=None, help="Minimum number of speakers.")
@click.option("--max-speakers", type=int, default=None, help="Maximum number of speakers.")
```

Add corresponding parameters to the `process` function signature:

```python
def process(
    audio_path: Path,
    output: Path | None,
    output_dir: Path | None,
    language: str | None,
    stt_provider: str | None,
    context: str | None,
    chunk_minutes: int | None,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    as_json: bool,
) -> None:
```

Pass them to `run_pipeline`:

```python
        result = run_pipeline(
            audio_path=audio_path,
            output=output,
            output_dir=output_dir,
            language=language,
            stt_provider=stt_provider,
            context=context,
            chunk_minutes=chunk_minutes,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
```

- [ ] **Step 5: Add args to run_pipeline()**

In `src/voxtract/pipeline.py`, add to `run_pipeline` signature:

```python
def run_pipeline(
    *,
    audio_path: Path,
    output: Path | None = None,
    output_dir: Path | None = None,
    language: str | None = None,
    stt_provider: str | None = None,
    context: str | None = None,
    chunk_minutes: int | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict:
```

Pass to `diarize_transcript`:

```python
                transcript = diarize_transcript(
                    transcript, wav_path,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    settings=settings,
                )
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/voxtract/cli.py src/voxtract/pipeline.py src/voxtract/speaker/diarizer.py tests/test_speaker/test_diarizer.py
git commit -m "feat: add --num-speakers, --min-speakers, --max-speakers CLI options for diarization"
```

---

### Task 3: Split utterances at speaker change boundaries

**Files:**
- Modify: `src/voxtract/speaker/diarizer.py`
- Create: `tests/test_speaker/test_split_by_speaker.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_speaker/test_split_by_speaker.py`:

```python
"""Tests for splitting utterances at speaker change boundaries."""
from __future__ import annotations

import pytest

from voxtract.models import Utterance


class TestSplitBySpeakerChange:
    def test_no_split_when_single_speaker(self) -> None:
        """Utterance fully within one speaker segment → no split."""
        from voxtract.speaker.diarizer import _split_by_speaker_change

        utts = [
            Utterance(speaker="SPEAKER_00", start_time=0.0, end_time=10.0, text="A B C"),
        ]
        segments = [(0.0, 15.0, "SPEAKER_00")]

        result = _split_by_speaker_change(utts, segments)
        assert len(result) == 1
        assert result[0].text == "A B C"

    def test_splits_at_speaker_boundary(self) -> None:
        """Utterance spanning two speaker segments → split into two."""
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
            Utterance(speaker="SPEAKER_00", start_time=9.0, end_time=11.0, text="네"),
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_speaker/test_split_by_speaker.py -v`
Expected: FAIL with `ImportError: cannot import name '_split_by_speaker_change'`

- [ ] **Step 3: Implement `_split_by_speaker_change()`**

Add to `src/voxtract/speaker/diarizer.py`:

```python
_MIN_SPLIT_DURATION_S = 2.0  # Don't split utterances shorter than this


def _split_by_speaker_change(
    utterances: list[Utterance],
    segments: list[tuple[float, float, str]],
) -> list[Utterance]:
    """Split utterances that span speaker change boundaries.

    When an utterance's time range crosses from one speaker segment to another,
    split it at the boundary. Text is proportionally divided based on time.
    Short utterances (<_MIN_SPLIT_DURATION_S) are not split.
    """
    if not utterances or not segments:
        return utterances

    result: list[Utterance] = []

    for utt in utterances:
        duration = utt.end_time - utt.start_time
        if duration < _MIN_SPLIT_DURATION_S:
            result.append(utt)
            continue

        # Find all speaker segments that overlap this utterance
        overlapping = []
        for seg_start, seg_end, speaker in segments:
            overlap_start = max(utt.start_time, seg_start)
            overlap_end = min(utt.end_time, seg_end)
            if overlap_end > overlap_start:
                overlapping.append((overlap_start, overlap_end, speaker))

        if len(overlapping) <= 1:
            result.append(utt)
            continue

        # Split text proportionally by time
        words = utt.text.split()
        total_duration = utt.end_time - utt.start_time

        for seg_start, seg_end, speaker in overlapping:
            seg_duration = seg_end - seg_start
            ratio = seg_duration / total_duration
            word_count = max(1, round(len(words) * ratio))

            seg_words = words[:word_count]
            words = words[word_count:]

            text = " ".join(seg_words).strip()
            if text:
                result.append(Utterance(
                    speaker=speaker,
                    start_time=seg_start,
                    end_time=seg_end,
                    text=text,
                ))

        # Remaining words go to last segment
        if words:
            last = result[-1]
            result[-1] = Utterance(
                speaker=last.speaker,
                start_time=last.start_time,
                end_time=last.end_time,
                text=last.text + " " + " ".join(words),
            )

    return result
```

- [ ] **Step 4: Wire into diarize_transcript()**

In `diarize_transcript()`, replace the block that currently reads (after the `exclusive_speaker_diarization` check, around lines 169-170):

```python
    new_utterances = _assign_speakers(transcript.utterances, diarization)
    new_utterances = _normalize_speaker_labels(new_utterances)
```

With:

```python
    # Extract segments for speaker-change splitting
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    new_utterances = _assign_speakers(transcript.utterances, diarization)
    new_utterances = _split_by_speaker_change(new_utterances, segments)
    new_utterances = _normalize_speaker_labels(new_utterances)
```

Note: `diarization` is already assigned at this point (lines 164-167). `_assign_speakers` also extracts segments internally — this small duplication keeps the functions independent.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_speaker/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/voxtract/speaker/diarizer.py tests/test_speaker/test_split_by_speaker.py
git commit -m "feat: split utterances at speaker change boundaries for finer-grained diarization"
```

---

### Task 4: Run full test suite

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Verify CLI help**

Run: `voxtract process --help`
Expected: Shows `--num-speakers`, `--min-speakers`, `--max-speakers` options

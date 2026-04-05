# Transcription Quality Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maximize Korean transcription quality with zero VRAM overhead by setting language default "ko", adding repetition_penalty=1.2, and increasing max_new_tokens to 1024.

**Architecture:** All three changes modify `config.py` defaults and wire through existing code paths. The repetition_penalty is applied to the HF transformers `generation_config` after qwen-asr model loading — no qwen-asr library modification needed. Language fallback flows through CLI → pipeline → STT provider.

**Tech Stack:** Python 3.11+, pydantic-settings, qwen-asr, HF transformers, pytest

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/voxtract/config.py` | Add `language`, `stt_repetition_penalty`; change `stt_max_tokens` default |
| Modify | `src/voxtract/stt/qwen3.py` | Apply `repetition_penalty` to `generation_config` after model load |
| Modify | `src/voxtract/cli.py` | Use `settings.language` as fallback for `--language` in both commands |
| Modify | `src/voxtract/pipeline.py` | Use `settings.language` as fallback when `language` param is None |
| Modify | `tests/test_stt/test_qwen3.py` | Tests for repetition_penalty, language fallback, max_tokens default |
| Modify | `tests/test_pipeline/test_preconvert.py` | Update existing tests for language fallback |

---

### Task 1: Add config settings (language, repetition_penalty, max_tokens)

**Files:**
- Modify: `src/voxtract/config.py:10-38`
- Modify: `tests/test_stt/test_qwen3.py`

- [ ] **Step 1: Write failing test for new config defaults**

Add to `tests/test_stt/test_qwen3.py` at the end of the file:

```python
class TestConfigDefaults:
    """Verify new quality-optimization config defaults."""

    def test_language_default_ko(self) -> None:
        from voxtract.config import Settings
        settings = Settings()
        assert settings.language == "ko"

    def test_repetition_penalty_default(self) -> None:
        from voxtract.config import Settings
        settings = Settings()
        assert settings.stt_repetition_penalty == 1.2

    def test_max_tokens_default_1024(self) -> None:
        from voxtract.config import Settings
        settings = Settings()
        assert settings.stt_max_tokens == 1024

    def test_language_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("VOXTRACT_LANGUAGE", "en")
        from voxtract.config import Settings
        settings = Settings()
        assert settings.language == "en"

    def test_repetition_penalty_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("VOXTRACT_STT_REPETITION_PENALTY", "1.5")
        from voxtract.config import Settings
        settings = Settings()
        assert settings.stt_repetition_penalty == 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stt/test_qwen3.py::TestConfigDefaults -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'language'`

- [ ] **Step 3: Add settings to config.py**

In `src/voxtract/config.py`, add three lines to the `Settings` class. After the `device` field (line 16) and before the `stt_provider` field (line 19), add:

```python
    # Language
    language: str = "ko"  # default language for STT (ISO 639-1)
```

After the `stt_context` field (line 22), add:

```python
    stt_repetition_penalty: float = 1.2  # suppress repeated token hallucinations
```

Change the `stt_max_tokens` default on line 23:

```python
    stt_max_tokens: int = 1024  # max generated tokens per inference batch
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stt/test_qwen3.py::TestConfigDefaults -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/voxtract/config.py tests/test_stt/test_qwen3.py
git commit -m "feat: add language, repetition_penalty settings; increase max_tokens to 1024"
```

---

### Task 2: Apply repetition_penalty to generation_config

**Files:**
- Modify: `src/voxtract/stt/qwen3.py:139-141`
- Modify: `tests/test_stt/test_qwen3.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_stt/test_qwen3.py` inside the existing `TestQwen3Provider` class, after the `test_use_cache_false` method:

```python
    @patch("qwen_asr.Qwen3ASRModel")
    @patch("voxtract.stt.qwen3.resolve_device", return_value="cpu")
    def test_repetition_penalty_set(self, mock_device, mock_model_cls) -> None:
        """Repetition penalty should be set on generation_config after model load."""
        from voxtract.stt.qwen3 import Qwen3Provider

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        provider = Qwen3Provider()
        provider._load_model()

        assert mock_model.model.generation_config.repetition_penalty == 1.2

    @patch("qwen_asr.Qwen3ASRModel")
    @patch("voxtract.stt.qwen3.resolve_device", return_value="cpu")
    def test_repetition_penalty_custom(self, mock_device, mock_model_cls) -> None:
        """Custom repetition_penalty from settings should be applied."""
        from voxtract.config import Settings
        from voxtract.stt.qwen3 import Qwen3Provider

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        settings = Settings(stt_repetition_penalty=1.5)
        provider = Qwen3Provider(settings=settings)
        provider._load_model()

        assert mock_model.model.generation_config.repetition_penalty == 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stt/test_qwen3.py::TestQwen3Provider::test_repetition_penalty_set tests/test_stt/test_qwen3.py::TestQwen3Provider::test_repetition_penalty_custom -v`
Expected: FAIL (generation_config.repetition_penalty not set to expected value)

- [ ] **Step 3: Add repetition_penalty to _load_model()**

In `src/voxtract/stt/qwen3.py`, after the `use_cache = False` line (line 141), add:

```python
            # Apply repetition penalty to suppress hallucinated repetitions
            if hasattr(self._model, 'model') and hasattr(self._model.model, 'generation_config'):
                self._model.model.generation_config.repetition_penalty = (
                    self._settings.stt_repetition_penalty
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stt/test_qwen3.py::TestQwen3Provider -v`
Expected: PASS (all TestQwen3Provider tests)

- [ ] **Step 5: Commit**

```bash
git add src/voxtract/stt/qwen3.py tests/test_stt/test_qwen3.py
git commit -m "feat: apply repetition_penalty to Qwen3 generation_config after model load"
```

---

### Task 3: Wire language fallback in CLI and pipeline

**Files:**
- Modify: `src/voxtract/cli.py:48-97` (transcribe command) and `src/voxtract/cli.py:111-153` (process command)
- Modify: `src/voxtract/pipeline.py:68-80`
- Modify: `tests/test_stt/test_qwen3.py`
- Modify: `tests/test_pipeline/test_preconvert.py`

- [ ] **Step 1: Write failing tests for language fallback**

Add to `tests/test_stt/test_qwen3.py` at the end of the file:

```python
class TestLanguageFallback:
    """Verify language fallback: CLI --language > settings.language."""

    @patch("qwen_asr.Qwen3ASRModel")
    @patch("voxtract.stt.qwen3.resolve_device", return_value="cpu")
    def test_transcribe_uses_settings_language_when_not_specified(
        self, mock_device, mock_model_cls, tmp_path,
    ) -> None:
        """When language=None, provider should receive settings.language."""
        from voxtract.config import Settings
        from voxtract.stt.qwen3 import Qwen3Provider

        mock_model = MagicMock()
        mock_model.transcribe.return_value = [
            MagicMock(language="Korean", text="테스트", time_stamps=None),
        ]
        mock_model_cls.from_pretrained.return_value = mock_model

        wav = tmp_path / "test.wav"
        wav.touch()

        settings = Settings(language="ko")
        provider = Qwen3Provider(settings=settings)
        provider.transcribe(wav, language="ko")

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "Korean"

    def test_pipeline_passes_settings_language(self, tmp_path) -> None:
        """Pipeline should use settings.language when language param is None."""
        from voxtract.models import Transcript
        from voxtract.pipeline import run_pipeline

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
             patch("voxtract.speaker.diarizer.diarize_transcript", return_value=fake_transcript):

            audio = tmp_path / "input.m4a"
            audio.touch()
            # language=None → should fallback to settings.language ("ko")
            run_pipeline(audio_path=audio, output=tmp_path / "out.json")

        call_kwargs = mock_provider.transcribe.call_args[1]
        assert call_kwargs["language"] == "ko"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stt/test_qwen3.py::TestLanguageFallback -v`
Expected: FAIL (`language` will be `None` since pipeline doesn't apply fallback yet)

- [ ] **Step 3: Modify pipeline.py to apply language fallback**

In `src/voxtract/pipeline.py`, in `run_pipeline()`, after line 93 (`chunk_min = chunk_minutes or settings.chunk_minutes`), add:

```python
    lang = language or settings.language or None
```

Then replace `language` with `lang` in the two places it's used:

Line 112 — change:
```python
            transcript = stt.transcribe(wav_path, language=language)
```
to:
```python
            transcript = stt.transcribe(wav_path, language=lang)
```

Lines 108-110 — change:
```python
            transcript = _transcribe_chunked(
                wav_path, stt, language, chunk_min, settings.overlap_seconds,
            )
```
to:
```python
            transcript = _transcribe_chunked(
                wav_path, stt, lang, chunk_min, settings.overlap_seconds,
            )
```

- [ ] **Step 4: Modify cli.py transcribe command**

In `src/voxtract/cli.py`, in the `transcribe` function, after line 59 (`settings = settings.model_copy(...)` block), add:

```python
    lang = language or settings.language or None
```

Then on line 70, change:
```python
            transcript = provider.transcribe(wav_path, language=language)
```
to:
```python
            transcript = provider.transcribe(wav_path, language=lang)
```

- [ ] **Step 5: Modify cli.py process command**

In `src/voxtract/cli.py`, in the `process` function, after line 127 (`try:`), change the `run_pipeline` call to pass the resolved language:

```python
    lang = language or settings.language or None
```

Add this line before the `try:` block (before line 127), then change line 132:
```python
            language=language,
```
to:
```python
            language=lang,
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_stt/test_qwen3.py::TestLanguageFallback tests/test_pipeline/test_preconvert.py tests/test_stt/test_qwen3.py::TestPipelineChunkingSkip -v`
Expected: PASS

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/voxtract/cli.py src/voxtract/pipeline.py tests/test_stt/test_qwen3.py
git commit -m "feat: wire language fallback — settings.language used when --language not provided"
```

---

### Task 4: Update README documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update Configuration table**

In `README.md`, replace the Configuration table with:

```markdown
## Configuration

Environment variables (prefix `VOXTRACT_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `VOXTRACT_DEVICE` | `auto` | `auto`, `cuda`, `cuda:0`, `cuda:1`, `cpu` |
| `VOXTRACT_LANGUAGE` | `ko` | Default language for STT (ISO 639-1) |
| `VOXTRACT_STT_CONTEXT` | `""` | Contextual hints for ASR |
| `VOXTRACT_STT_MAX_TOKENS` | `1024` | Max generated tokens per inference batch |
| `VOXTRACT_STT_REPETITION_PENALTY` | `1.2` | Suppress repeated token hallucinations |
| `VOXTRACT_CHUNK_MINUTES` | `25` | Audio chunk size for long files |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with new quality config options"
```

---

### Task 5: Run full test suite and verify

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Verify config loads without errors**

Run: `uv run python -c "from voxtract.config import Settings; s = Settings(); print(f'language={s.language} rep_penalty={s.stt_repetition_penalty} max_tokens={s.stt_max_tokens}')"`
Expected: `language=ko rep_penalty=1.2 max_tokens=1024`

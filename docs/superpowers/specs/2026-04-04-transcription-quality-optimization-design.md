# Transcription Quality Optimization

**Goal:** Maximize Korean transcription quality with zero VRAM overhead by tuning STT defaults and generation parameters.

**Context:** voxtract uses Qwen3-ASR-1.7B via qwen-asr library. The model's `generate()` call only receives `max_new_tokens`; all other generation parameters fall through to HF transformers' `generation_config`. We exploit this to inject `repetition_penalty` without modifying qwen-asr.

---

## Changes

### 1. Language default "ko"

**Problem:** Language defaults to `None` (auto-detect). Qwen3-ASR wastes tokens on language identification and may produce language tags in output instead of pure transcription text.

**Solution:** Add `language` setting with default `"ko"`. CLI `--language` overrides it. When language is specified, Qwen3-ASR appends `language Korean<asr_text>` to the prompt, forcing text-only output.

**Files:**
- `src/voxtract/config.py` — add `language: str = "ko"`
- `src/voxtract/cli.py` — both `transcribe` and `process` commands: use `settings.language` as fallback when `--language` is not provided
- `src/voxtract/pipeline.py` — use `settings.language` as fallback when `language` param is None

### 2. Repetition penalty

**Problem:** Greedy decoding occasionally produces repeated phrases (hallucination), especially in noisy audio or silence gaps that pass VAD filtering.

**Solution:** Set `repetition_penalty=1.2` on the model's `generation_config` after loading. This biases against generating tokens that already appeared, suppressing repetitive hallucinations. No VRAM overhead — it only modifies logit scores during token selection.

**Files:**
- `src/voxtract/config.py` — add `stt_repetition_penalty: float = 1.2`
- `src/voxtract/stt/qwen3.py` — in `_load_model()`, after model creation: set `self._model.model.generation_config.repetition_penalty`

### 3. Increase max_new_tokens

**Problem:** Default `max_new_tokens=512` may truncate long utterances. qwen-asr chunks audio internally, so each chunk generates independently — 512 is usually sufficient but leaves no safety margin.

**Solution:** Increase default from 512 to 1024. Already exposed as `VOXTRACT_STT_MAX_TOKENS` environment variable.

**Files:**
- `src/voxtract/config.py` — change `stt_max_tokens: int = 512` to `stt_max_tokens: int = 1024`

---

## What is NOT changing

- **Device configuration** — stays `"auto"` default, user sets `VOXTRACT_DEVICE` env var as needed
- **Beam search** — not included; VRAM cost outweighs marginal quality gain for well-trained models
- **Context handling** — stays as-is (optional `--context` flag)
- **Audio preprocessing** — already well-configured (loudnorm + highpass)
- **VAD filtering** — no changes
- **Speaker diarization** — no changes
- **use_cache** — stays `False` (no beam search means no penalty for disabled KV cache)

---

## Testing strategy

- Unit tests for new config defaults (`language`, `stt_repetition_penalty`, `stt_max_tokens`)
- Unit test: verify `generation_config.repetition_penalty` is set after model load
- Unit test: verify language fallback chain (CLI flag > settings default)
- Existing pipeline/STT tests must continue to pass

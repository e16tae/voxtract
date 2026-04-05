# Voxtract

Audio to speaker-attributed transcript — extract who said what, when.

## Installation

```bash
uv pip install -e ".[all]"
```

## Usage

```bash
# Full pipeline: transcribe + diarize
voxtract process audio.m4a --json

# Transcribe only (no speaker diarization)
voxtract transcribe audio.m4a --json

# With context hints for better accuracy
voxtract process audio.m4a --context "교합력 센서, 울산대학교" --json
```

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

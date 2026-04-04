# tests/test_stt/test_qwen3.py
"""Tests for Qwen3 ASR STT provider."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest

from voxtract.models import Transcript


@dataclass
class FakeAlignItem:
    text: str
    start_time: float  # seconds
    end_time: float

@dataclass
class FakeAlignResult:
    items: list

@dataclass
class FakeASRTranscription:
    language: str
    text: str
    time_stamps: FakeAlignResult | None = None


class TestQwen3Registration:
    def test_registered(self) -> None:
        from voxtract.stt import _PROVIDERS, _ensure_builtins
        _ensure_builtins()
        assert "qwen3" in _PROVIDERS


class TestQwen3Provider:
    def test_implements_protocol(self) -> None:
        from voxtract.stt.base import STTProvider
        from voxtract.stt.qwen3 import Qwen3Provider
        provider = Qwen3Provider.__new__(Qwen3Provider)
        assert isinstance(provider, STTProvider)

    def test_handles_long_audio(self) -> None:
        from voxtract.stt.qwen3 import Qwen3Provider
        provider = Qwen3Provider()
        assert provider.handles_long_audio is True

    @patch("qwen_asr.Qwen3ASRModel")
    @patch("voxtract.stt.qwen3.resolve_device", return_value="cpu")
    def test_use_cache_false(self, mock_device, mock_model_cls) -> None:
        """Model should be loaded with use_cache=False set on model.config."""
        from voxtract.stt.qwen3 import Qwen3Provider

        mock_model = MagicMock()
        mock_model.model.config.use_cache = True
        mock_model_cls.from_pretrained.return_value = mock_model

        provider = Qwen3Provider()
        provider._load_model()

        assert mock_model.model.config.use_cache is False

    @patch("qwen_asr.Qwen3ASRModel")
    @patch("voxtract.stt.qwen3.resolve_device", return_value="cpu")
    def test_max_tokens_from_settings(self, mock_device, mock_model_cls) -> None:
        """max_new_tokens should be read from settings.stt_max_tokens."""
        from voxtract.config import Settings
        from voxtract.stt.qwen3 import Qwen3Provider

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        settings = Settings(stt_max_tokens=1024)
        provider = Qwen3Provider(settings=settings)
        provider._load_model()

        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert call_kwargs["max_new_tokens"] == 1024

    def test_resolve_language_maps_iso_codes(self) -> None:
        from voxtract.stt.qwen3 import _resolve_language

        assert _resolve_language("ko") == "Korean"
        assert _resolve_language("en") == "English"
        assert _resolve_language("KO") == "Korean"  # case insensitive
        assert _resolve_language("Korean") == "Korean"  # passthrough
        assert _resolve_language("unknown") == "unknown"  # passthrough
        assert _resolve_language(None) is None

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

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        from voxtract.errors import STTError
        from voxtract.stt.qwen3 import Qwen3Provider
        provider = Qwen3Provider()
        with pytest.raises(STTError, match="not found"):
            provider.transcribe(tmp_path / "nonexistent.mp3")


class TestBuildTranscript:
    """Test _build_transcript with mock qwen-asr output."""

    def test_builds_from_timestamps(self) -> None:
        from voxtract.stt.qwen3 import Qwen3Provider

        provider = Qwen3Provider.__new__(Qwen3Provider)
        provider._model_repo = "test-model"

        result = FakeASRTranscription(
            language="Korean",
            text="안녕하세요. 회의를 시작하겠습니다.",
            time_stamps=FakeAlignResult(items=[
                FakeAlignItem(text="안녕하세요.", start_time=0.0, end_time=1.2),
                FakeAlignItem(text="회의를", start_time=1.3, end_time=1.8),
                FakeAlignItem(text="시작하겠습니다.", start_time=1.8, end_time=3.5),
            ]),
        )

        transcript = provider._build_transcript(result, Path("test.mp3"))

        assert transcript.language == "Korean"
        assert len(transcript.utterances) >= 1
        for utt in transcript.utterances:
            assert utt.start_time >= 0
            assert utt.end_time > utt.start_time
            assert utt.speaker == "Speaker 0"

    def test_groups_words_into_utterances(self) -> None:
        """Words with small gaps should be grouped; large gaps create new utterances."""
        from voxtract.stt.qwen3 import Qwen3Provider

        provider = Qwen3Provider.__new__(Qwen3Provider)
        provider._model_repo = "test-model"

        result = FakeASRTranscription(
            language="Korean",
            text="첫 문장. 두 번째 문장.",
            time_stamps=FakeAlignResult(items=[
                FakeAlignItem(text="첫", start_time=0.0, end_time=0.5),
                FakeAlignItem(text="문장.", start_time=0.5, end_time=1.5),
                # 2-second pause → should split into new utterance
                FakeAlignItem(text="두", start_time=3.5, end_time=4.0),
                FakeAlignItem(text="번째", start_time=4.0, end_time=4.5),
                FakeAlignItem(text="문장.", start_time=4.5, end_time=5.5),
            ]),
        )

        transcript = provider._build_transcript(result, Path("test.mp3"))
        assert len(transcript.utterances) == 2
        assert "첫 문장." in transcript.utterances[0].text
        assert "두 번째 문장." in transcript.utterances[1].text

    def test_utterances_contain_word_timestamps(self) -> None:
        """Each utterance should carry per-word timestamps from ForcedAligner."""
        from voxtract.stt.qwen3 import Qwen3Provider

        provider = Qwen3Provider.__new__(Qwen3Provider)
        provider._model_repo = "test-model"

        result = FakeASRTranscription(
            language="Korean",
            text="안녕 세상",
            time_stamps=FakeAlignResult(items=[
                FakeAlignItem(text="안녕", start_time=0.0, end_time=0.8),
                FakeAlignItem(text="세상", start_time=0.9, end_time=1.5),
            ]),
        )

        transcript = provider._build_transcript(result, Path("test.mp3"))
        assert len(transcript.utterances) == 1
        utt = transcript.utterances[0]
        assert utt.words is not None
        assert len(utt.words) == 2
        assert utt.words[0].text == "안녕"
        assert utt.words[0].start_time == 0.0
        assert utt.words[0].end_time == 0.8
        assert utt.words[1].text == "세상"

    def test_word_timestamps_split_across_utterances(self) -> None:
        """Word timestamps should be partitioned correctly when pause splits utterances."""
        from voxtract.stt.qwen3 import Qwen3Provider

        provider = Qwen3Provider.__new__(Qwen3Provider)
        provider._model_repo = "test-model"

        result = FakeASRTranscription(
            language="Korean",
            text="A B C",
            time_stamps=FakeAlignResult(items=[
                FakeAlignItem(text="A", start_time=0.0, end_time=0.5),
                # 2-second pause
                FakeAlignItem(text="B", start_time=2.5, end_time=3.0),
                FakeAlignItem(text="C", start_time=3.0, end_time=3.5),
            ]),
        )

        transcript = provider._build_transcript(result, Path("test.mp3"))
        assert len(transcript.utterances) == 2
        assert len(transcript.utterances[0].words) == 1
        assert len(transcript.utterances[1].words) == 2

    def test_no_timestamps_uses_full_text(self) -> None:
        from voxtract.stt.qwen3 import Qwen3Provider

        provider = Qwen3Provider.__new__(Qwen3Provider)
        provider._model_repo = "test-model"

        result = FakeASRTranscription(
            language="Korean",
            text="타임스탬프 없는 전체 텍스트.",
            time_stamps=None,
        )

        transcript = provider._build_transcript(result, Path("test.mp3"))
        assert len(transcript.utterances) == 1
        assert transcript.utterances[0].text == "타임스탬프 없는 전체 텍스트."

    def test_empty_text_returns_empty(self) -> None:
        from voxtract.stt.qwen3 import Qwen3Provider

        provider = Qwen3Provider.__new__(Qwen3Provider)
        provider._model_repo = "test-model"

        result = FakeASRTranscription(language="Korean", text="", time_stamps=None)

        transcript = provider._build_transcript(result, Path("test.mp3"))
        assert len(transcript.utterances) == 0


class TestPipelineChunkingSkip:
    """Verify pipeline skips external chunking for handles_long_audio providers."""

    @patch("voxtract.audio.splitter.get_duration", return_value=3600)
    @patch("voxtract.pipeline._transcribe_chunked")
    def test_skips_chunking_for_qwen3(self, mock_chunked, mock_duration, tmp_path):
        from voxtract.pipeline import run_pipeline

        fake_transcript = Transcript(
            language="ko", speakers=["Speaker 0"], utterances=[], metadata={},
        )
        mock_provider = MagicMock()
        mock_provider.handles_long_audio = True
        mock_provider.transcribe.return_value = fake_transcript

        wav_path = tmp_path / "long.wav"
        wav_path.touch()

        with patch("voxtract.stt.get_provider", return_value=mock_provider), \
             patch("voxtract.audio.splitter.convert_to_wav16k", return_value=wav_path), \
             patch("voxtract.speaker.diarizer.diarize_transcript", return_value=fake_transcript):
            audio = tmp_path / "long.wav"
            audio.touch()
            run_pipeline(audio_path=audio, output=tmp_path / "out.json")

        mock_chunked.assert_not_called()
        mock_provider.transcribe.assert_called_once()


class TestConfigDefaults:
    """Verify new quality-optimization config defaults."""

    def test_language_default_ko(self, monkeypatch) -> None:
        monkeypatch.delenv("VOXTRACT_LANGUAGE", raising=False)
        from voxtract.config import Settings
        settings = Settings()
        assert settings.language == "ko"

    def test_repetition_penalty_default(self, monkeypatch) -> None:
        monkeypatch.delenv("VOXTRACT_STT_REPETITION_PENALTY", raising=False)
        from voxtract.config import Settings
        settings = Settings()
        assert settings.stt_repetition_penalty == 1.2

    def test_max_tokens_default_1024(self, monkeypatch) -> None:
        monkeypatch.delenv("VOXTRACT_STT_MAX_TOKENS", raising=False)
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

    def test_max_tokens_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("VOXTRACT_STT_MAX_TOKENS", "2048")
        from voxtract.config import Settings
        settings = Settings()
        assert settings.stt_max_tokens == 2048

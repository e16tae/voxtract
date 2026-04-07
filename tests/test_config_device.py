"""Tests for GPU device resolver functions in voxtract.config."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from voxtract.config import (
    Settings,
    _pick_best_gpu,
    _pick_secondary_gpu,
    _resolve_auto,
    resolve_device_stt,
    resolve_device_speaker,
)


def _make_device_props(total_memory: int) -> MagicMock:
    props = MagicMock()
    props.total_memory = total_memory
    return props


# ---------------------------------------------------------------------------
# _resolve_auto
# ---------------------------------------------------------------------------

class TestResolveAuto:
    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_available(self, _mock: MagicMock) -> None:
        assert _resolve_auto() == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_not_available(self, _mock: MagicMock) -> None:
        assert _resolve_auto() == "cpu"

    def test_torch_not_installed(self) -> None:
        import builtins
        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            assert _resolve_auto() == "cpu"


# ---------------------------------------------------------------------------
# _pick_best_gpu
# ---------------------------------------------------------------------------

class TestPickBestGpu:
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.device_count", return_value=3)
    @patch("torch.cuda.is_available", return_value=True)
    def test_multi_gpu_picks_largest(self, _avail, _count, mock_props) -> None:
        mock_props.side_effect = lambda i: _make_device_props(
            {0: 8_000, 1: 24_000, 2: 12_000}[i]
        )
        assert _pick_best_gpu() == "cuda:1"

    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.is_available", return_value=True)
    def test_single_gpu(self, _avail, _count, mock_props) -> None:
        mock_props.return_value = _make_device_props(8_000)
        assert _pick_best_gpu() == "cuda:0"

    @patch("torch.cuda.is_available", return_value=False)
    def test_no_cuda(self, _avail) -> None:
        assert _pick_best_gpu() == "cpu"


# ---------------------------------------------------------------------------
# _pick_secondary_gpu
# ---------------------------------------------------------------------------

class TestPickSecondaryGpu:
    @patch("torch.cuda.device_count", return_value=3)
    def test_returns_different_gpu(self, _count) -> None:
        # Primary is cuda:1, should pick cuda:0 (first != 1)
        assert _pick_secondary_gpu("cuda:1") == "cuda:0"

    @patch("torch.cuda.device_count", return_value=2)
    def test_returns_other_of_two(self, _count) -> None:
        assert _pick_secondary_gpu("cuda:0") == "cuda:1"

    @patch("torch.cuda.device_count", return_value=1)
    def test_single_gpu_returns_primary(self, _count) -> None:
        assert _pick_secondary_gpu("cuda:0") == "cuda:0"

    @patch("torch.cuda.device_count", return_value=2)
    def test_handles_cuda_no_index(self, _count) -> None:
        # "cuda" with no index → treated as index 0
        result = _pick_secondary_gpu("cuda")
        assert result == "cuda:1"


# ---------------------------------------------------------------------------
# resolve_device_stt
# ---------------------------------------------------------------------------

class TestResolveDeviceStt:
    def test_explicit_device_stt(self) -> None:
        settings = Settings(device_stt="cuda:2")
        assert resolve_device_stt(settings) == "cuda:2"

    def test_explicit_device_stt_auto(self) -> None:
        settings = Settings(device_stt="auto")
        with patch("voxtract.config._resolve_auto", return_value="cuda"):
            assert resolve_device_stt(settings) == "cuda"

    @patch("voxtract.config._pick_best_gpu", return_value="cuda:1")
    def test_auto_picks_best_gpu(self, _mock) -> None:
        settings = Settings(device="auto", device_stt="")
        assert resolve_device_stt(settings) == "cuda:1"

    def test_fallback_to_device(self) -> None:
        settings = Settings(device="cuda:3", device_stt="")
        assert resolve_device_stt(settings) == "cuda:3"


# ---------------------------------------------------------------------------
# resolve_device_speaker
# ---------------------------------------------------------------------------

class TestResolveDeviceSpeaker:
    def test_explicit_device_speaker(self) -> None:
        settings = Settings(device_speaker="cuda:2")
        assert resolve_device_speaker(settings) == "cuda:2"

    def test_explicit_device_speaker_auto(self) -> None:
        settings = Settings(device_speaker="auto")
        with patch("voxtract.config._resolve_auto", return_value="cpu"):
            assert resolve_device_speaker(settings) == "cpu"

    @patch("voxtract.config._pick_secondary_gpu", return_value="cuda:1")
    @patch("voxtract.config._pick_best_gpu", return_value="cuda:0")
    def test_auto_picks_secondary(self, _best, _sec) -> None:
        settings = Settings(device="auto", device_speaker="")
        assert resolve_device_speaker(settings) == "cuda:1"

    @patch("voxtract.config._pick_secondary_gpu", return_value="cuda:0")
    @patch("voxtract.config._pick_best_gpu", return_value="cuda:0")
    def test_single_gpu_same_as_stt(self, _best, _sec) -> None:
        settings = Settings(device="auto", device_speaker="")
        assert resolve_device_speaker(settings) == "cuda:0"

    def test_fallback_to_device(self) -> None:
        settings = Settings(device="cpu", device_speaker="")
        assert resolve_device_speaker(settings) == "cpu"

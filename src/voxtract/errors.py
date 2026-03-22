"""Error types and exit codes for the voxtract CLI."""

from __future__ import annotations

# Exit codes
EXIT_OK = 0
EXIT_INPUT_ERROR = 1
EXIT_STT_ERROR = 2
EXIT_CONFIG_ERROR = 5
EXIT_AUDIO_ERROR = 6
EXIT_SPEAKER_ERROR = 7


class VoxtractError(Exception):
    """Base exception with an associated exit code."""

    exit_code: int = 1

    def __init__(self, message: str, *, code: str = "UNKNOWN", recoverable: bool = False):
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": str(self),
            "recoverable": self.recoverable,
        }


class InputError(VoxtractError):
    exit_code = EXIT_INPUT_ERROR


class STTError(VoxtractError):
    exit_code = EXIT_STT_ERROR


class ConfigError(VoxtractError):
    exit_code = EXIT_CONFIG_ERROR


class AudioError(VoxtractError):
    exit_code = EXIT_AUDIO_ERROR


class SpeakerError(VoxtractError):
    exit_code = EXIT_SPEAKER_ERROR

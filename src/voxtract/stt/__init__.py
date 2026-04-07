"""STT provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voxtract.stt.base import STTProvider

_PROVIDERS: dict[str, type] = {}


def register(name: str, cls: type) -> None:
    """Register an STT provider class by name."""
    _PROVIDERS[name] = cls


def get_provider(name: str, **kwargs) -> STTProvider:
    """Instantiate a registered STT provider by name."""
    _ensure_builtins()
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown STT provider: {name!r}. Available: {list(_PROVIDERS)}"
        )
    return _PROVIDERS[name](**kwargs)


def _ensure_builtins() -> None:
    """Import built-in providers so they self-register."""
    if _PROVIDERS:
        return
    try:
        import voxtract.stt.whisper  # noqa: F401
    except ImportError:
        pass

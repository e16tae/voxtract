"""CLI interface using Click — subcommands: transcribe, process."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from voxtract.errors import VoxtractError


def _output_json(data: dict) -> None:
    """Write JSON to stdout."""
    click.echo(json.dumps(data, ensure_ascii=False, indent=2))


def _output_error(step: str, err: VoxtractError) -> None:
    """Write error JSON to stderr and exit."""
    click.echo(
        json.dumps({"status": "error", "step": step, "error": err.to_dict()}, ensure_ascii=False, indent=2),
        err=True,
    )
    sys.exit(err.exit_code)


@click.group()
@click.version_option(package_name="voxtract")
def cli() -> None:
    """Voxtract — audio to speaker-attributed transcript."""


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--format", "-f", "fmt", type=click.Choice(["json", "txt"]), default="json", help="Output format.")
@click.option("--language", "-l", default=None, help="Language code (e.g. 'ko'). Auto-detected if omitted.")
@click.option("--stt-provider", default=None, help="STT provider name.")
@click.option("--context", "-c", default=None, help="Contextual hints for ASR (topic, names, jargon).")
@click.option("--json", "as_json", is_flag=True, help="Output JSON for agent consumption.")
def transcribe(
    audio_path: Path,
    output: Path | None,
    fmt: str,
    language: str | None,
    stt_provider: str | None,
    context: str | None,
    as_json: bool,
) -> None:
    """Transcribe audio file to a transcript (no diarization)."""
    import tempfile

    from voxtract.audio.splitter import convert_to_wav16k
    from voxtract.config import get_settings
    from voxtract.stt import get_provider
    from voxtract.errors import STTError

    settings = get_settings()
    if context is not None:
        settings = settings.model_copy(update={"stt_context": context})
    provider_name = stt_provider or settings.stt_provider
    lang = language or settings.language or None

    try:
        provider = get_provider(provider_name, settings=settings)
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = convert_to_wav16k(
                audio_path, output_dir=Path(tmp_dir),
                normalize=settings.audio_normalize,
                highpass=settings.audio_highpass,
            )
            transcript = provider.transcribe(wav_path, language=lang)
    except VoxtractError as e:
        if as_json:
            _output_error("transcribe", e)
        raise click.ClickException(str(e)) from e
    except Exception as e:
        err = STTError(str(e), code="STT_PROVIDER_FAILED", recoverable=True)
        if as_json:
            _output_error("transcribe", err)
        raise click.ClickException(str(e)) from e

    from voxtract.formatter import write_transcript

    ext = "txt" if fmt == "txt" else "json"
    output = output or Path(f"{audio_path.stem}_transcript.{ext}")
    write_transcript(transcript, output, fmt)

    if as_json:
        _output_json({
            "status": "success",
            "step": "transcribe",
            "output_file": str(output.resolve()),
            "metadata": {
                "speakers_detected": len(transcript.speakers),
                "duration_seconds": transcript.metadata.get("duration"),
                "language": transcript.language,
            },
        })
    else:
        click.echo(f"Transcript saved to {output}")


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--format", "-f", "fmt", type=click.Choice(["json", "txt"]), default="json", help="Output format.")
@click.option("--language", "-l", default=None, help="Language code.")
@click.option("--stt-provider", default=None)
@click.option("--context", "-c", default=None, help="Contextual hints for ASR (topic, names, jargon).")
@click.option("--chunk-minutes", type=int, default=None, help="Audio chunk size in minutes (default: 25).")
@click.option("--num-speakers", type=int, default=None, help="Exact number of speakers (if known).")
@click.option("--min-speakers", type=int, default=None, help="Minimum number of speakers.")
@click.option("--max-speakers", type=int, default=None, help="Maximum number of speakers.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON for agent consumption.")
def process(
    audio_path: Path,
    output: Path | None,
    output_dir: Path | None,
    fmt: str,
    language: str | None,
    stt_provider: str | None,
    context: str | None,
    chunk_minutes: int | None,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    as_json: bool,
) -> None:
    """Full pipeline: audio → transcribe → diarize → transcript."""
    from voxtract.config import get_settings
    from voxtract.pipeline import run_pipeline

    settings = get_settings()
    lang = language or settings.language or None

    try:
        result = run_pipeline(
            audio_path=audio_path,
            output=output,
            output_dir=output_dir,
            fmt=fmt,
            language=lang,
            stt_provider=stt_provider,
            context=context,
            chunk_minutes=chunk_minutes,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except VoxtractError as e:
        if as_json:
            _output_error("process", e)
        raise click.ClickException(str(e)) from e
    except Exception as e:
        err = VoxtractError(str(e), code="PIPELINE_FAILED", recoverable=False)
        if as_json:
            _output_error("process", err)
        raise click.ClickException(str(e)) from e

    if as_json:
        _output_json(result)
    else:
        click.echo(f"Transcript saved to {result.get('output_file', 'unknown')}")

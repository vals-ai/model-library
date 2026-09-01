from typing import cast

import pytest
from rich.console import Console

from model_library.base import LLM, TranscriptionMetadata, TranscriptionResult
from model_library.registry_utils import CLI_ONLY_PROVIDERS
from scripts import run_models


async def test_run_transcription_smoke() -> None:
    seen: dict[str, object] = {}

    class TranscriptionModel:
        supports_transcription = True
        text = ""

        async def query(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("transcription models must not use query")

        async def transcribe_audio(
            self,
            *,
            name: str,
            mime: str,
            audio: bytes,
            language: str | None = None,
        ) -> TranscriptionResult:
            seen.update(
                {"name": name, "mime": mime, "audio": audio, "language": language}
            )
            return TranscriptionResult(
                text=self.text,
                metadata=TranscriptionMetadata(
                    audio_bytes=len(audio),
                    request_duration_seconds=0.1,
                    input_tokens=8,
                    output_tokens=6,
                    total_tokens=14,
                    cost_usd=0.00008,
                ),
            )

    model = TranscriptionModel()
    with pytest.raises(Exception, match="Unexpected transcription"):
        await run_models._run_transcription_smoke(  # pyright: ignore[reportPrivateUsage]
            cast(LLM, model)
        )

    model.text = "The capital of France is Paris."
    await run_models._run_transcription_smoke(  # pyright: ignore[reportPrivateUsage]
        cast(LLM, model)
    )

    assert seen["name"] == "smoke.webm"
    assert seen["mime"] == "audio/webm"
    assert cast(bytes, seen["audio"]).startswith(b"\x1aE\xdf\xa3")
    assert seen["language"] is None


def test_select_models_skips_cli_only_providers() -> None:
    runnable, skipped = run_models.select_models(
        run_models.model_registry, research=False
    )

    assert not [
        key
        for key in runnable
        if run_models.model_registry[key].provider_name in CLI_ONLY_PROVIDERS
    ]
    assert set(skipped) <= CLI_ONLY_PROVIDERS
    assert "devin/adaptive" in skipped["devin"]
    assert "factory/router" in skipped["factory"]


def test_dashboard_reports_skips_without_counting_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(run_models.skipped_model_names, "devin", ["devin/adaptive"])
    monkeypatch.setattr(run_models, "providers", {"devin"})

    console = Console(width=120)
    with console.capture() as capture:
        console.print(run_models.create_dashboard(total=0, completed_count=0))
    output = capture.get()

    assert "devin: 1 models, 0 failed, 1 skipped (cli-only)" in output

from types import SimpleNamespace

import pytest
from model_library import registry_utils
from model_library.base import TranscriptionMetadata
from model_library.register_models import (
    CostProperties,
    TranscriptionCostProperties,
)
from model_library.registry_utils import (
    compute_transcription_cost,
    get_transcription_registry_config,
)


def _metadata(**values: object) -> TranscriptionMetadata:
    return TranscriptionMetadata(
        audio_bytes=1,
        request_duration_seconds=99.0,
        **values,
    )


@pytest.mark.parametrize(
    ("pricing", "metadata", "expected"),
    [
        pytest.param(
            TranscriptionCostProperties(usd_per_minute=0.6, billing_basis="session"),
            _metadata(billable_duration_seconds=30.0),
            0.3,
            id="per-minute",
        ),
        pytest.param(
            TranscriptionCostProperties(
                usd_per_minute=60.0, billing_basis="audio", increment_seconds=6.0
            ),
            _metadata(audio_duration_seconds=0.1),
            6.0,
            id="increment",
        ),
        pytest.param(
            TranscriptionCostProperties(
                usd_per_minute=60.0,
                billing_basis="session",
                minimum_billable_seconds=10.0,
                increment_seconds=6.0,
            ),
            _metadata(billable_duration_seconds=1.0),
            12.0,
            id="minimum-before-increment",
        ),
        pytest.param(
            TranscriptionCostProperties(usd_per_minute=1.0, billing_basis="session"),
            _metadata(audio_duration_seconds=30.0),
            None,
            id="missing-billable-measurement",
        ),
        pytest.param(
            None,
            _metadata(audio_duration_seconds=60.0, billable_duration_seconds=60.0),
            None,
            id="unpriced",
        ),
    ],
)
def test_transcription_duration_cost(
    monkeypatch: pytest.MonkeyPatch,
    pricing: TranscriptionCostProperties | None,
    metadata: TranscriptionMetadata,
    expected: float | None,
) -> None:
    """Duration pricing respects billing rules without changing measurements."""
    config = SimpleNamespace(transcription_cost=pricing, costs_per_million_token=None)
    monkeypatch.setattr(
        registry_utils, "get_transcription_registry_config", lambda _key: config
    )
    original = metadata.model_dump()

    cost = compute_transcription_cost("test/duration", metadata)

    assert cost == (pytest.approx(expected) if expected is not None else None)
    assert metadata.model_dump() == original


def test_transcription_token_cost_requires_returned_input_and_output_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        transcription_cost=None,
        costs_per_million_token=CostProperties(input=2.5, output=10.0),
    )
    monkeypatch.setattr(
        registry_utils, "get_transcription_registry_config", lambda _key: config
    )

    assert compute_transcription_cost(
        "test/token", _metadata(input_tokens=200, output_tokens=100)
    ) == pytest.approx(0.0015)
    assert compute_transcription_cost("test/token", _metadata(input_tokens=200)) is None


def test_gemini_live_registry_uses_reported_token_cost() -> None:
    config = get_transcription_registry_config("google/gemini-3.5-transcribe-live")

    assert config is not None
    assert config.transcription_cost is None
    assert config.costs_per_million_token is not None
    assert config.costs_per_million_token.input == 3.5
    assert config.costs_per_million_token.output == 21.0
    assert compute_transcription_cost(
        "google/gemini-3.5-transcribe-live",
        _metadata(
            input_tokens=1_500,
            output_tokens=175,
            billable_duration_seconds=60,
        ),
    ) == pytest.approx(0.008925)


@pytest.mark.parametrize(
    ("registry_key", "hourly_rate"),
    [
        ("meta/muse_voice_transcribe", 0.18),
        ("elevenlabs/scribe_v2_realtime", 0.39),
    ],
)
def test_registry_duration_rates_match_official_hourly_prices(
    registry_key: str,
    hourly_rate: float,
) -> None:
    config = get_transcription_registry_config(registry_key)

    assert config is not None
    assert config.transcription_cost is not None
    assert config.transcription_cost.usd_per_minute == pytest.approx(hourly_rate / 60)

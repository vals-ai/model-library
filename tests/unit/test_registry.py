"""Unit tests for model registry operations"""

import asyncio
import time
from datetime import UTC, datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import pytest

import model_library
import model_library.register_models as register_models_module
import model_library.registry_utils as registry_utils
from model_library.base import GatewayLLM, LLM
from model_library.providers.delegates.fireworks import FireworksModel
from model_library.providers.google import GoogleModel
from model_library.register_models import (
    ICONS_DIR,
    CompanyIcon,
    Metadata,
    ModelRegistry,
    get_company_icon,
    get_deprecated_model_registry,
    get_model_registry,
    get_provider_registry,
    register_provider,
)
from model_library.registry_utils import TokenDict, get_registry_model, recompute_cost


@pytest.mark.parametrize("concurrency_type", ["threads", "async"])
async def test_concurrent_access(concurrency_type: Literal["threads", "async"]) -> None:
    """
    Test that get_model_registry() is safe under concurrency
    Only one registry object is to be created
    """

    async def _test_async() -> list[tuple[int, int, int]]:
        results: list[tuple[int, int, int]] = []

        async def get_registry_worker(worker_id: int) -> ModelRegistry:
            await asyncio.sleep(0.01)
            registry = get_model_registry()
            results.append((worker_id, id(registry), len(registry)))
            return registry

        tasks = [get_registry_worker(i) for i in range(10)]
        await asyncio.gather(*tasks)
        return results

    def _test_threads() -> list[tuple[int, int, int]]:
        results: list[tuple[int, int, int]] = []

        def get_registry_worker(worker_id: int) -> ModelRegistry:
            time.sleep(0.01)
            registry = get_model_registry()
            results.append((worker_id, id(registry), len(registry)))
            return registry

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_registry_worker, i) for i in range(10)]
            for future in futures:
                future.result()
        return results

    match concurrency_type:
        case "async":
            results = await _test_async()
        case "threads":
            results = _test_threads()

    # All registries should be the same object (same ID)
    registry_ids = {result[1] for result in results}
    assert len(registry_ids) == 1, (
        f"Multiple registry objects created in {concurrency_type} mode"
    )

    # All registries should have the same size
    registry_sizes = {result[2] for result in results}
    assert len(registry_sizes) == 1, f"Registry sizes vary in {concurrency_type} mode"

    # Registry should not be empty
    registry_size = registry_sizes.pop()
    assert registry_size > 0, f"Registry is empty in {concurrency_type} mode"


async def test_registry_is_singleton():
    """Test that repeated calls return the same dict object"""
    registry1 = get_provider_registry()
    registry2 = get_provider_registry()
    assert registry1 is registry2


@pytest.mark.parametrize(
    ("company", "expected"),
    [
        ("OpenAI", CompanyIcon(light="OpenAI.svg", dark="OpenAI-dark.svg")),
        ("Inworld", CompanyIcon(light="Inworld.png")),
        ("AssemblyAI", CompanyIcon(light="AssemblyAI.svg")),
        ("AWS", CompanyIcon(light="AWS.svg", dark="AWS-dark.svg")),
        (
            "Google Cloud",
            CompanyIcon(light="Google Cloud.svg", dark="Google Cloud-dark.svg"),
        ),
        (
            "MiniMax AI",
            CompanyIcon(light="MiniMax AI.svg", dark="MiniMax AI-dark.svg"),
        ),
        ("Tencent", CompanyIcon(light="Tencent.svg", dark="Tencent-dark.svg")),
        ("Reson8", CompanyIcon(light="Reson8.png")),
        ("Unknown Company", None),
    ],
)
def test_get_company_icon_uses_exact_bundled_filenames(
    company: str, expected: CompanyIcon | None
) -> None:
    icon = get_company_icon(company)

    assert icon == expected
    if icon is not None:
        assert (ICONS_DIR / icon.light).is_file()
        if icon.dark is not None:
            assert (ICONS_DIR / icon.dark).is_file()


def test_get_company_icon_rejects_ambiguous_light_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ = (tmp_path / "Duplicate.svg").write_text("<svg></svg>")
    _ = (tmp_path / "Duplicate.png").write_bytes(b"png")
    monkeypatch.setattr(register_models_module, "ICONS_DIR", tmp_path)

    with pytest.raises(ValueError, match="Multiple light icons found for Duplicate"):
        get_company_icon("Duplicate")


def test_deprecated_registry_loads_nullable_costs():
    config = get_deprecated_model_registry()["poolside/laguna-xs.2"]

    assert config.costs_per_million_token is None


async def test_register_provider_decorator():
    """Decorator registers a provider correctly."""

    @register_provider("test")
    class DummyProvider(LLM):
        pass

    registry = get_provider_registry()
    assert "test" in registry
    assert registry["test"] is DummyProvider


async def test_registry_contains_expected_providers():
    registry = get_provider_registry()
    expected = ["openai", "zai", "fireworks", "azure"]
    for name in expected:
        assert name in registry


def test_gateway_registry_model_skips_provider_resolution(monkeypatch):
    registry_config = next(
        config
        for config in get_model_registry().values()
        if config.provider_name == "openai"
    )
    provider_registry = MagicMock(
        side_effect=AssertionError("Gateway clients must not load local providers")
    )
    gateway_settings = MagicMock()
    gateway_settings.MODEL_GATEWAY_API_KEY = "test-key"
    gateway_settings.get.return_value = "https://gateway.test"

    monkeypatch.setattr(model_library, "model_library_settings", gateway_settings)
    monkeypatch.setattr(registry_utils, "get_provider_registry", provider_registry)
    monkeypatch.setattr(
        registry_utils,
        "get_registry_config",
        lambda _model: registry_config,
    )

    model = registry_utils.get_registry_model(registry_config.full_key)

    assert isinstance(model, GatewayLLM)
    provider_registry.assert_not_called()


async def test_manual_and_registry_instantiation():
    """
    Test manual and registry instantiation of a provider.
    """
    model1 = FireworksModel("test-model-1")
    assert model1 is not None

    model2 = get_registry_model("google/gemini-2.5-flash")
    assert model2 is not None

    model3 = FireworksModel("test-model-3")
    assert model3 is not None

    assert isinstance(model1, FireworksModel)
    assert isinstance(model2, GoogleModel)
    assert isinstance(model3, FireworksModel)


@pytest.mark.parametrize(
    ("metadata", "ignored"),
    [
        ({}, False),
        ({"internal_only": True}, True),
        ({"internal_only": True, "ignored_for_cost": False}, False),
        ({"internal_only": False, "ignored_for_cost": True}, True),
    ],
)
def test_ignored_for_cost_defaults_to_internal_only(
    metadata: dict[str, bool], ignored: bool
) -> None:
    assert Metadata.model_validate(metadata).ignored_for_cost is ignored


def test_registry_metadata_is_available_but_not_rendered():
    model = get_registry_model("google/gemini-2.5-flash")

    assert model.metadata is not None
    rendered = repr(model)
    assert "_metadata" not in rendered
    assert "ModelConfig" not in rendered


async def test_cli_models_not_instantiable():
    with pytest.raises(ValueError, match="Factory CLI"):
        get_registry_model("factory/router")


@pytest.mark.parametrize(
    ("model_key", "label"),
    [("devin/adaptive", "Devin Adaptive"), ("factory/router", "Factory Router")],
)
def test_agent_router_models_are_publicly_selectable(
    model_key: str, label: str
) -> None:
    config = get_model_registry()[model_key]

    assert config.label == label
    assert config.metadata.deprecated is False
    assert config.metadata.internal_only is False
    assert config.metadata.available_for_everyone is True
    assert config.metadata.available_as_evaluator is False


@pytest.mark.parametrize("model_key", ["devin/adaptive", "factory/router"])
def test_agent_router_models_carry_nominal_prices_excluded_from_cost(
    model_key: str,
) -> None:
    config = get_model_registry()[model_key]
    costs = config.costs_per_million_token

    assert costs is not None
    assert costs.input == 3.0
    assert costs.output == 15.0
    assert config.metadata.ignored_for_cost is True
async def test_none_costs():
    """Models without costs should have costs_per_million_token=None and return None cost."""
    from model_library.base import QueryResultMetadata
    from model_library.registry_utils import get_model_cost, get_registry_config

    # poolside/laguna-m.1 has no costs defined in its YAML config
    config = get_registry_config("poolside/laguna-m.1")
    assert config is not None
    assert config.costs_per_million_token is None

    cost = get_model_cost("poolside/laguna-m.1")
    assert cost is None

    # _calculate_cost should return None when costs are missing
    model = get_registry_model("poolside/laguna-m.1")
    metadata = QueryResultMetadata(in_tokens=100, out_tokens=50)
    result = await model._calculate_cost(metadata)
    assert result is None


def test_discount_follows_context_batch_and_cache_rate_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from model_library.base import QueryResultMetadata
    from model_library.register_models import (
        BatchCost,
        CacheCost,
        ContextCost,
        CostProperties,
    )
    from model_library.registry_utils import compute_model_cost, discount_model_cost

    costs = CostProperties(
        input=2.0,
        output=8.0,
        cache=CacheCost(read=1.0, write=3.0, write_1h_markup=2.0),
        context=ContextCost(
            threshold=1500,
            input=4.0,
            output=12.0,
            cache=CacheCost(read=2.0, write=5.0, write_1h_markup=2.0),
        ),
        batch=BatchCost(
            input_discount=0.5, output_discount=0.5, cache_discount=0.5
        ),
        current_discount=50,
    )
    monkeypatch.setattr(registry_utils, "get_model_cost", lambda _key: costs)
    usage = QueryResultMetadata(
        in_tokens=1000,
        out_tokens=2000,
        reasoning_tokens=250,
        cache_read_tokens=500,
        cache_write_tokens=100,
        cache_write_1h_tokens=25,
    )

    cost = compute_model_cost("test/context-batch", usage, batch=True)

    assert cost is not None
    assert cost.input == pytest.approx(0.002)
    assert cost.output == pytest.approx(0.012)
    assert cost.reasoning == pytest.approx(0.0015)
    assert cost.cache_read == pytest.approx(0.0005)
    assert cost.cache_write == pytest.approx(0.0002875)

    discounted_cost = discount_model_cost("test/context-batch", cost)
    assert discounted_cost.input == pytest.approx(0.001)
    assert discounted_cost.output == pytest.approx(0.006)
    assert discounted_cost.reasoning == pytest.approx(0.00075)
    assert discounted_cost.cache_read == pytest.approx(0.00025)
    assert discounted_cost.cache_write == pytest.approx(0.00014375)


@pytest.mark.parametrize(
    "model_key,fee",
    [
        ("openai/gpt-5.5", 0.04),
        ("anthropic/claude-sonnet-4-6", 0.04),
        ("grok/grok-4.5", 0.02),
        ("google/gemini-3.5-flash", 0.056),
        ("google/gemini-2.5-pro", 0.14),
    ],
)
async def test_web_search_fee_hardcoded_per_provider(model_key: str, fee: float):
    from model_library.base import QueryResultMetadata

    model = get_registry_model(model_key)
    cost = await model._calculate_cost(  # pyright: ignore[reportPrivateUsage]
        QueryResultMetadata(in_tokens=0, out_tokens=0, web_search_requests=4)
    )
    assert cost is not None
    assert cost.web_search == pytest.approx(fee)
    assert cost.total == pytest.approx(fee)


def test_registry_cost_has_no_web_search_fee():
    from model_library.base import QueryResultMetadata
    from model_library.registry_utils import compute_model_cost

    cost = compute_model_cost(
        "openai/gpt-5.5",
        QueryResultMetadata(in_tokens=0, out_tokens=0, web_search_requests=1),
    )
    assert cost is not None
    assert cost.web_search is None


async def test_web_search_cost_absent_without_searches():
    from model_library.base import QueryResultMetadata

    model = get_registry_model("openai/gpt-5.5")
    cost = await model._calculate_cost(  # pyright: ignore[reportPrivateUsage]
        QueryResultMetadata(in_tokens=1000, out_tokens=0)
    )
    assert cost is not None
    assert cost.web_search is None


@pytest.mark.parametrize(
    ("model_key", "at", "factor"),
    [
        ("deepseek/deepseek-v4-flash", datetime(2026, 11, 4, 1, 0, tzinfo=UTC), 1.0),
        ("deepseek/deepseek-v4-flash", datetime(2026, 11, 4, 3, 59, tzinfo=UTC), 1.0),
        ("deepseek/deepseek-v4-flash", datetime(2026, 11, 4, 4, 0, tzinfo=UTC), 0.5),
        ("deepseek/deepseek-v4-flash", datetime(2026, 11, 4, 6, 0, tzinfo=UTC), 1.0),
        ("deepseek/deepseek-v4-flash", datetime(2026, 11, 4, 10, 0, tzinfo=UTC), 0.5),
        ("deepseek/deepseek-v4-flash", datetime(2026, 11, 4, 16, 0, tzinfo=UTC), 0.5),
        ("deepseek/deepseek-v4-flash", datetime(2026, 11, 7, 2, 0, tzinfo=UTC), 0.5),
        ("deepseek/deepseek-v4-pro-medium", datetime(2026, 11, 4, 16, 0, tzinfo=UTC), 0.5),
        ("deepseek/deepseek-v4.1-flash-1017", datetime(2026, 11, 8, 8, 0, tzinfo=UTC), 0.5),
        ("deepseek/deepseek-chat", datetime(2026, 11, 4, 16, 0, tzinfo=UTC), 1.0),
    ],
)
def test_deepseek_off_peak_halves_token_cost_outside_weekday_peak_hours(
    model_key: str, at: datetime, factor: float
) -> None:
    from model_library.registry_utils import QueryResultCost, discount_model_cost

    cost = QueryResultCost(input=1.0, output=2.0, cache_read=0.5, web_search=0.1)

    discounted = discount_model_cost(model_key, cost, at=at)

    assert discounted.input == pytest.approx(1.0 * factor)
    assert discounted.output == pytest.approx(2.0 * factor)
    assert discounted.cache_read == pytest.approx(0.5 * factor)
    assert discounted.web_search == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("completed_at", "factor"),
    [
        (datetime(2026, 11, 7, 2, 0, tzinfo=UTC), 0.5),
        (datetime(2026, 11, 9, 2, 0, tzinfo=UTC), 1.0),
        (None, 1.0),
    ],
)
async def test_recompute_cost_prices_off_peak_from_completed_at(
    completed_at: datetime | None, factor: float
) -> None:
    tokens: TokenDict = {"in_tokens": 1_000_000, "out_tokens": 1_000_000}
    standard = await recompute_cost("deepseek/deepseek-v4.1-flash-1017", tokens)

    cost = await recompute_cost(
        "deepseek/deepseek-v4.1-flash-1017", tokens, completed_at=completed_at
    )

    assert cost.total == pytest.approx(standard.total * factor)


async def test_a_query_can_be_repriced_from_its_returned_metadata_alone() -> None:
    from unittest.mock import AsyncMock, patch

    from model_library.base import QueryResult, QueryResultMetadata

    model_key = "deepseek/deepseek-v4.1-flash-1017"
    model = get_registry_model(model_key)
    usage = QueryResultMetadata(in_tokens=1_000_000, out_tokens=1_000_000)
    off_peak = datetime(2026, 11, 7, 2, 0, tzinfo=UTC)
    with (
        patch.object(
            type(model),
            "_query_impl",
            new=AsyncMock(return_value=QueryResult(output_text="hi", metadata=usage)),
        ),
        patch("model_library.base.base.datetime", wraps=datetime) as clock,
    ):
        clock.now.return_value = off_peak
        out = await model.query("hi")

    saved = QueryResultMetadata.model_validate_json(out.metadata.model_dump_json())
    tokens: TokenDict = {"in_tokens": saved.in_tokens, "out_tokens": saved.out_tokens}
    repriced = await recompute_cost(model_key, tokens, completed_at=saved.completed_at)

    assert saved.completed_at == off_peak
    assert out.metadata.effective_cost is not None and out.metadata.cost is not None
    assert repriced.total == pytest.approx(out.metadata.effective_cost.total)
    assert repriced.total == pytest.approx(out.metadata.cost.total * 0.5)


async def test_recompute_cost_takes_off_the_current_discount() -> None:
    """Solar Decide lists input at $0.10/M with 50% off, so a call is charged $0.05/M."""
    tokens: TokenDict = {"in_tokens": 1_000_000, "out_tokens": 0}

    cost = await recompute_cost("decision/solar-decide", tokens)

    assert cost.input == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("completed_at", "factor"),
    [
        (datetime(2026, 8, 16, 15, 59, tzinfo=UTC), 1.0),
        (datetime(2026, 8, 16, 16, 0, tzinfo=UTC), 0.5),
    ],
)
async def test_recompute_cost_gives_no_off_peak_before_it_started(
    completed_at: datetime, factor: float
) -> None:
    tokens: TokenDict = {"in_tokens": 1_000_000, "out_tokens": 1_000_000}
    standard = await recompute_cost("deepseek/deepseek-v4.1-flash-1017", tokens)

    cost = await recompute_cost(
        "deepseek/deepseek-v4.1-flash-1017", tokens, completed_at=completed_at
    )

    assert cost.total == pytest.approx(standard.total * factor)


@pytest.mark.parametrize(
    ("completed_at", "factor"),
    [
        (datetime(2026, 10, 1, 2, 0, tzinfo=UTC), 0.5),
        (datetime(2026, 10, 7, 8, 0, tzinfo=UTC), 0.5),
        (datetime(2026, 10, 8, 2, 0, tzinfo=UTC), 1.0),
    ],
)
async def test_recompute_cost_treats_china_holidays_as_off_peak(
    completed_at: datetime, factor: float
) -> None:
    tokens: TokenDict = {"in_tokens": 1_000_000, "out_tokens": 1_000_000}
    standard = await recompute_cost("deepseek/deepseek-v4.1-flash-1017", tokens)

    cost = await recompute_cost(
        "deepseek/deepseek-v4.1-flash-1017", tokens, completed_at=completed_at
    )

    assert cost.total == pytest.approx(standard.total * factor)


def test_off_peak_stacks_with_current_discount(monkeypatch: pytest.MonkeyPatch) -> None:
    from model_library.register_models import (
        CostProperties,
        OffPeakDiscount,
    )
    from model_library.registry_utils import QueryResultCost, discount_model_cost

    costs = CostProperties(
        input=1.0,
        output=2.0,
        current_discount=20,
        off_peak=OffPeakDiscount(
            discount=50,
            discount_hours_utc={"mon-fri": [(0, 1), (4, 24)]},
        ),
    )
    monkeypatch.setattr(registry_utils, "get_model_cost", lambda _key: costs)
    cost = QueryResultCost(input=1.0, output=2.0)

    peak = discount_model_cost("test/both", cost, at=datetime(2026, 11, 4, 2, tzinfo=UTC))
    off_peak = discount_model_cost(
        "test/both", cost, at=datetime(2026, 11, 4, 12, tzinfo=UTC)
    )

    assert peak.input == pytest.approx(0.8)
    assert off_peak.input == pytest.approx(0.4)
    assert off_peak.output == pytest.approx(0.8)


@pytest.mark.parametrize(
    "hours",
    [
        {"mon-fri": [(0, 24)], "fri": [(0, 24)]},
        {"monday": [(0, 24)]},
        {"fri-mon": [(0, 24)]},
        {"mon": [(4, 4)]},
    ],
)
def test_off_peak_rejects_bad_discount_hours(hours: dict[str, list[tuple[int, int]]]) -> None:
    from model_library.register_models import OffPeakDiscount

    with pytest.raises(ValueError):
        OffPeakDiscount(discount=50, discount_hours_utc=hours)


def test_off_peak_gives_no_hourly_discount_on_an_unmapped_day() -> None:
    from model_library.register_models import OffPeakDiscount

    off_peak = OffPeakDiscount(discount=50, discount_hours_utc={"sat": [(0, 24)]})

    assert off_peak.applies_at(datetime(2026, 11, 7, 12, tzinfo=UTC))
    assert not off_peak.applies_at(datetime(2026, 11, 8, 12, tzinfo=UTC))

"""Unit tests for model registry operations"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import pytest

from model_library.base.base import LLM
from model_library.providers.fireworks import FireworksModel
from model_library.providers.google import GoogleModel
from model_library.register_models import (
    ModelRegistry,
    get_model_registry,
    get_provider_registry,
    register_provider,
)
from model_library.registry_utils import get_registry_model


@pytest.mark.parametrize("concurrency_type", ["threads", "async"])
def test_concurrent_access(concurrency_type: Literal["threads", "async"]) -> None:
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
            results = asyncio.run(_test_async())
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


def test_registry_is_singleton():
    """Test that repeated calls return the same dict object"""
    registry1 = get_provider_registry()
    registry2 = get_provider_registry()
    assert registry1 is registry2


def test_register_provider_decorator():
    """Decorator registers a provider correctly."""

    @register_provider("test")
    class DummyProvider(LLM):
        pass

    registry = get_provider_registry()
    assert "test" in registry
    assert registry["test"] is DummyProvider


def test_registry_contains_expected_providers():
    registry = get_provider_registry()
    expected = ["openai", "zai", "fireworks", "azure"]
    for name in expected:
        assert name in registry


def test_manual_and_registry_instantiation():
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

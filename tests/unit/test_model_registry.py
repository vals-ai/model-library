"""Unit tests for model registry operations"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import pytest

from model_library.register_models import ModelRegistry, get_model_registry


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

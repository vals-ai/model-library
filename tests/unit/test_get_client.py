"""Unit tests for get_client functionality"""

import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from model_library.base import LLMConfig, client_registry
from model_library.providers.openai import OpenAIModel


@pytest.fixture(autouse=True)
async def clear_client_registry():
    """Clear the global client registry before each test"""
    client_registry.clear()
    yield
    client_registry.clear()


@pytest.mark.parametrize("concurrency_type", ["threads", "async"])
async def test_concurrent_get_client(
    concurrency_type: Literal["threads", "async"],
) -> None:
    """
    Test that get_client() is safe under concurrency.
    Only one client should be created per provider/api_key combination.
    """

    async def _test_async() -> list[tuple[int, int, tuple[str, str]]]:
        results: list[tuple[int, int, tuple[str, str]]] = []

        async def get_client_worker(worker_id: int) -> None:
            await asyncio.sleep(0.01)
            model = OpenAIModel(
                "gpt-4", config=LLMConfig(custom_api_key=SecretStr("test_key"))
            )
            client = model.get_client()
            results.append((worker_id, id(client), model._client_registry_key))  # pyright: ignore[reportPrivateUsage]

        tasks = [get_client_worker(i) for i in range(10)]
        await asyncio.gather(*tasks)
        return results

    def _test_threads() -> list[tuple[int, int, tuple[str, str]]]:
        results: list[tuple[int, int, tuple[str, str]]] = []

        def get_client_worker(worker_id: int) -> None:
            time.sleep(0.01)
            model = OpenAIModel(
                "gpt-4", config=LLMConfig(custom_api_key=SecretStr("test_key"))
            )
            client = model.get_client()
            results.append((worker_id, id(client), model._client_registry_key))  # pyright: ignore[reportPrivateUsage]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_client_worker, i) for i in range(10)]
            for future in futures:
                future.result()
        return results

    match concurrency_type:
        case "async":
            results = await _test_async()
        case "threads":
            results = _test_threads()

    registry_keys = {result[2] for result in results}
    assert len(registry_keys) == 1, (
        f"All workers should have same registry key in {concurrency_type} mode"
    )


async def test_client_caching_same_instance():
    """Test that calling get_client() multiple times returns the same cached instance"""
    model = OpenAIModel("gpt-4", config=LLMConfig(custom_api_key=SecretStr("test_key")))

    client1 = model.get_client()
    client2 = model.get_client()

    assert client1 is client2, "get_client() should return the same cached instance"


async def test_client_registry_key_consistency():
    """Test that client registry key is consistent for same provider/api_key"""
    api_key = "test_key_123"
    config = LLMConfig(custom_api_key=SecretStr(api_key))

    model1 = OpenAIModel("gpt-4", config=config)
    model2 = OpenAIModel("gpt-3.5-turbo", config=config)

    assert model1._client_registry_key == model2._client_registry_key  # pyright: ignore[reportPrivateUsage]

    client1 = model1.get_client()
    client2 = model2.get_client()

    assert client1 is client2


async def test_different_api_keys_different_registry_keys():
    """Test that different API keys result in different registry keys"""
    model1 = OpenAIModel("gpt-4", config=LLMConfig(custom_api_key=SecretStr("key_1")))
    model2 = OpenAIModel("gpt-4", config=LLMConfig(custom_api_key=SecretStr("key_2")))

    assert model1._client_registry_key != model2._client_registry_key  # pyright: ignore[reportPrivateUsage]


async def test_has_client():
    """Test that has_client() correctly indicates client presence"""
    model = OpenAIModel("gpt-4", config=LLMConfig(custom_api_key=SecretStr("test_key")))

    test_client = MagicMock()
    model.assign_client(test_client)

    assert model.has_client(), "Client should exist after assign_client"


async def test_assign_client_idempotent():
    """Test that assign_client is idempotent (double-checked locking works)"""
    model = OpenAIModel("gpt-4", config=LLMConfig(custom_api_key=SecretStr("test_key")))

    client1 = MagicMock(name="client1")
    client2 = MagicMock(name="client2")

    key = model._client_registry_key  # pyright: ignore[reportPrivateUsage]
    if key in client_registry:
        del client_registry[key]

    model.assign_client(client1)
    first_client = client_registry[key]

    model.assign_client(client2)
    second_client = client_registry[key]

    assert first_client is client1
    assert second_client is client1
    assert second_client is not client2


async def test_client_registry_key_hash():
    """Test that client registry key uses hashed API key"""
    api_key = "my_secret_api_key"
    config = LLMConfig(custom_api_key=SecretStr(api_key))

    model = OpenAIModel("gpt-4", config=config)

    expected_hash = hashlib.sha256(api_key.encode()).hexdigest()
    registry_key = model._client_registry_key  # pyright: ignore[reportPrivateUsage]

    assert registry_key[0] == "openai"
    assert registry_key[1] == expected_hash


async def test_provider_and_api_key_in_registry_key():
    """Test that registry key contains both provider and api_key hash"""
    model = OpenAIModel("gpt-4", config=LLMConfig(custom_api_key=SecretStr("test_key")))

    registry_key = model._client_registry_key  # pyright: ignore[reportPrivateUsage]
    assert isinstance(registry_key, tuple)
    assert len(registry_key) == 2
    assert registry_key[0] == "openai"
    assert isinstance(registry_key[1], str)
    assert len(registry_key[1]) == 64


async def test_separate_registry_entries_for_different_keys():
    """Test that different API keys maintain separate registry entries"""
    model1 = OpenAIModel("gpt-4", config=LLMConfig(custom_api_key=SecretStr("key_1")))
    model2 = OpenAIModel("gpt-3.5", config=LLMConfig(custom_api_key=SecretStr("key_2")))

    assert model1._client_registry_key[0] == "openai"  # pyright: ignore[reportPrivateUsage]
    assert model2._client_registry_key[0] == "openai"  # pyright: ignore[reportPrivateUsage]
    assert model1._client_registry_key != model2._client_registry_key  # pyright: ignore[reportPrivateUsage]

    client1 = MagicMock(name="client1")
    client2 = MagicMock(name="client2")

    model1.assign_client(client1)
    model2.assign_client(client2)

    assert model1.has_client()
    assert model2.has_client()
    assert client_registry[model1._client_registry_key] is client1  # pyright: ignore[reportPrivateUsage]
    assert client_registry[model2._client_registry_key] is client2  # pyright: ignore[reportPrivateUsage]

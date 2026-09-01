"""Tests for Kimi provider configuration."""

import asyncio
from types import SimpleNamespace
from pydantic import SecretStr
import pytest

from model_library.base import LLMConfig
from model_library.base.input import FileWithId, RawInput, RawResponse, TextInput
from model_library.providers.delegates.kimi import KimiConfig, KimiModel
from model_library.providers.openai import OpenAIConfig, OpenAIModel
from model_library.registry_utils import get_registry_model

_INPUT = [TextInput(text="")]

class TestKimiConfig:
    async def test_parallel_tool_calls_configures_openai_delegate(self):
        config = LLMConfig(
            custom_api_key=SecretStr("sk-test"),
            provider_config=KimiConfig(parallel_tool_calls=False),
        )
        model = KimiModel("kimi-k2", config=config)

        assert config.custom_endpoint is None
        assert isinstance(config.provider_config, KimiConfig)
        assert isinstance(model.provider_config, KimiConfig)
        assert model.provider_config.parallel_tool_calls is False
        assert isinstance(model.delegate, OpenAIModel)
        assert isinstance(model.delegate.provider_config, OpenAIConfig)
        assert model.delegate.provider_config.parallel_tool_calls is False

        body = await model.build_body(_INPUT, tools=[])
        assert body["parallel_tool_calls"] is False

    async def test_preprocess_files_fetches_file_contents_concurrently_and_preserves_order(
        self,
    ):
        model = KimiModel(
            "kimi-k2",
            config=LLMConfig(custom_api_key=SecretStr("sk-test")),
        )
        assert model.delegate is not None
        client = model.delegate.get_client()
        started: list[str] = []
        release = asyncio.Event()
        active_requests = 0
        max_active_requests = 0

        async def content(*, file_id: str):
            nonlocal active_requests, max_active_requests
            started.append(file_id)
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            if len(started) == 2:
                release.set()
            await asyncio.wait_for(release.wait(), timeout=1.0)
            active_requests -= 1
            return SimpleNamespace(text=f"content:{file_id}")

        client.files.content = content
        middle = TextInput(text="middle")
        inputs = [
            FileWithId(
                type="file",
                name="first.txt",
                mime="text/plain",
                file_id="file-first",
            ),
            middle,
            FileWithId(
                type="file",
                name="second.txt",
                mime="text/plain",
                file_id="file-second",
            ),
        ]

        preprocessed = await asyncio.wait_for(
            model._preprocess_files(inputs),
            timeout=1.0,  # pyright: ignore[reportPrivateUsage]
        )

        assert set(started) == {"file-first", "file-second"}
        assert max_active_requests == 2
        assert isinstance(preprocessed[0], RawInput)
        assert preprocessed[0].input == {
            "role": "system",
            "content": "content:file-first",
        }
        assert preprocessed[1] is middle
        assert isinstance(preprocessed[2], RawInput)
        assert preprocessed[2].input == {
            "role": "system",
            "content": "content:file-second",
        }

    async def test_preprocess_files_cancels_sibling_fetches_after_failure(self):
        model = KimiModel(
            "kimi-k2",
            config=LLMConfig(custom_api_key=SecretStr("sk-test")),
        )
        assert model.delegate is not None
        client = model.delegate.get_client()
        slow_started = asyncio.Event()
        slow_cancel_requested = asyncio.Event()
        allow_cancel_cleanup = asyncio.Event()

        async def content(*, file_id: str):
            if file_id == "file-fail":
                await asyncio.wait_for(slow_started.wait(), timeout=1.0)
                raise RuntimeError("fetch failed")

            slow_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                slow_cancel_requested.set()
                await asyncio.wait_for(allow_cancel_cleanup.wait(), timeout=1.0)
                raise
            raise AssertionError("slow fetch should be cancelled")

        client.files.content = content
        inputs = [
            FileWithId(
                type="file",
                name="fail.txt",
                mime="text/plain",
                file_id="file-fail",
            ),
            FileWithId(
                type="file",
                name="slow.txt",
                mime="text/plain",
                file_id="file-slow",
            ),
        ]

        task = asyncio.create_task(
            model._preprocess_files(inputs)  # pyright: ignore[reportPrivateUsage]
        )
        await asyncio.wait_for(slow_cancel_requested.wait(), timeout=1.0)
        assert not task.done()

        allow_cancel_cleanup.set()
        with pytest.raises(RuntimeError, match="fetch failed"):
            await asyncio.wait_for(task, timeout=1.0)

"""Shared fixtures for provider transcription adapter tests."""

from collections.abc import AsyncIterator, Iterator
from typing import Generic, TypeVar

import pytest
from pydantic import SecretStr

from examples.data.audio import tone_wav
from model_library.base import LLMConfig, client_registry

AUDIO = tone_wav()


def config(registry_key: str) -> LLMConfig:
    return LLMConfig(
        supports_transcription=True,
        custom_api_key=SecretStr("provider-key"),
        registry_key=registry_key,
    )


@pytest.fixture(autouse=True)
def clear_transcription_client_registry() -> Iterator[None]:
    client_registry.clear()
    yield
    client_registry.clear()


class ScriptedStream:
    def __init__(self, events: list[object]) -> None:
        self._script = events
        self.sends: list[object] = []

    def _record_send(self, payload: object) -> None:
        self.sends.append(payload)

    async def __aiter__(self) -> AsyncIterator[object]:
        for event in self._script:
            yield event


StreamT = TypeVar("StreamT")


class Connection(Generic[StreamT]):
    def __init__(self, stream: StreamT) -> None:
        self.stream = stream

    async def __aenter__(self) -> StreamT:
        return self.stream

    async def __aexit__(self, *_args: object) -> None:
        return None

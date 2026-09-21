"""Inworld transcription adapter tests."""

import asyncio
import json
from typing import cast
from unittest.mock import patch

import pytest

from model_library.providers.voice.inworld import InworldModel
from tests.unit.providers.transcription_test_support import (
    AUDIO,
    Connection,
    ScriptedStream,
    config,
)


class _InworldStream(ScriptedStream):
    def __init__(
        self,
        responses: list[dict[str, object]],
        *,
        remain_open_after_responses: bool = False,
    ) -> None:
        super().__init__(cast(list[object], responses))
        self._remain_open_after_responses = remain_open_after_responses
        self.messages: list[dict[str, object]] = []

    async def send(self, message: str) -> None:
        payload = json.loads(message)
        self._record_send(payload)
        self.messages.append(payload)
        await asyncio.sleep(0)

    async def __aiter__(self):
        for response in self._script:
            yield json.dumps(response)
        if self._remain_open_after_responses:
            await asyncio.Event().wait()


async def test_inworld_streaming_assembles_disjoint_finals_and_usage() -> None:
    stream = _InworldStream(
        [
            {
                "result": {
                    "transcription": {"isFinal": True, "transcript": "first segment"}
                }
            },
            {
                "result": {
                    "transcription": {
                        "isFinal": True,
                        "transcript": "second segment",
                    }
                }
            },
            {
                "result": {
                    "usage": {
                        "transcribedAudioMs": 2_400,
                        "modelId": "inworld/inworld-stt-1",
                    }
                }
            },
        ],
        remain_open_after_responses=True,
    )
    model = InworldModel(
        "inworld/inworld-stt-1", config=config("inworld/inworld-stt-1")
    )

    with patch(
        "model_library.providers.voice.inworld.connect",
        return_value=Connection(stream),
    ):
        result = await asyncio.wait_for(
            model.transcribe_audio(name="clip.wav", mime="audio/wav", audio=AUDIO),
            timeout=2,
        )

    assert result.text == "first segment second segment"
    assert result.metadata.billable_duration_seconds == 2.4


async def test_inworld_requires_terminal_usage() -> None:
    stream = _InworldStream(
        [{"result": {"transcription": {"isFinal": True, "transcript": "hello world"}}}]
    )
    model = InworldModel(
        "inworld/inworld-stt-1", config=config("inworld/inworld-stt-1")
    )

    with (
        patch(
            "model_library.providers.voice.inworld.connect",
            return_value=Connection(stream),
        ),
        pytest.raises(RuntimeError, match="did not include terminal usage"),
    ):
        await model.transcribe_audio(name="clip.wav", mime="audio/wav", audio=AUDIO)


class _RejectingStream(ScriptedStream):
    def __init__(self) -> None:
        super().__init__([])
        self._rejected = asyncio.Event()

    async def send(self, message: str) -> None:
        await asyncio.sleep(0)
        self._record_send(message)
        if self._rejected.is_set():
            raise ConnectionError("socket closed")
        if next(iter(json.loads(message))) == "audioChunk":
            self._rejected.set()

    async def __aiter__(self):
        await self._rejected.wait()
        yield json.dumps({"error": {"message": "unsupported model"}})


async def test_inworld_surfaces_server_rejection_over_send_failure() -> None:
    model = InworldModel(
        "inworld/inworld-stt-1", config=config("inworld/inworld-stt-1")
    )

    with (
        patch(
            "model_library.providers.voice.inworld.connect",
            return_value=Connection(_RejectingStream()),
        ),
        pytest.raises(
            RuntimeError, match="Transcription failed: unsupported model"
        ) as raised,
    ):
        await model.transcribe_audio(name="clip.wav", mime="audio/wav", audio=AUDIO)

    group = raised.value.__cause__
    assert isinstance(group, ExceptionGroup)
    assert {type(error) for error in group.exceptions} == {
        RuntimeError,
        ConnectionError,
    }

"""Cross-provider transcription client-registry tests."""

from unittest.mock import patch

import pytest
from pydantic import SecretStr

from model_library.base import LLMConfig, TranscriptionOnly
from model_library.providers.amazon.transcribe import AWSTranscribeModel
from model_library.providers.voice.assemblyai import AssemblyAIModel
from model_library.providers.voice.azure_speech import AzureSpeechModel
from model_library.providers.voice.cartesia import CartesiaModel
from model_library.providers.voice.groq import GroqModel
from model_library.providers.voice.inworld import InworldModel
from model_library.providers.voice.xai import XAITranscriptionModel
from tests.unit.providers.transcription_test_support import (
    clear_transcription_client_registry as clear_transcription_client_registry,
    config,
)


@pytest.mark.parametrize(
    ("model_type", "model_name", "registry_key"),
    [
        (AWSTranscribeModel, "transcribe-streaming", "amazon/transcribe-streaming"),
        (
            AzureSpeechModel,
            "universal-language-model",
            "azure_speech/universal-language-model",
        ),
        (
            AssemblyAIModel,
            "universal-3-5-pro",
            "assemblyai/universal-3-5-pro",
        ),
        (XAITranscriptionModel, "grok-voice-transcribe-2.0", "xai/grok-voice-transcribe-2.0"),
        (InworldModel, "inworld-stt-1", "inworld/inworld-stt-1"),
        (GroqModel, "whisper-large-v3-turbo", "groq/whisper-large-v3-turbo"),
    ],
)
def test_request_scoped_models_skip_shared_client_initialization(
    model_type: type[TranscriptionOnly],
    model_name: str,
    registry_key: str,
) -> None:
    with patch.object(
        model_type,
        "get_client",
        side_effect=AssertionError("request-scoped model initialized a root client"),
    ):
        model_type(model_name, config=config(registry_key))


def test_transcription_client_registry_distinguishes_connections() -> None:
    configs = [
        LLMConfig(custom_api_key=SecretStr("first-key")),
        LLMConfig(custom_api_key=SecretStr("second-key")),
        LLMConfig(
            custom_api_key=SecretStr("first-key"),
            custom_endpoint="https://proxy.example/cartesia",
        ),
    ]

    keys = {
        CartesiaModel("ink-2", config=model_config)._client_registry_key  # pyright: ignore[reportPrivateUsage]
        for model_config in configs
    }

    assert len(keys) == 3
    assert {namespace for namespace, _key_hash in keys} == {"cartesia.transcription"}

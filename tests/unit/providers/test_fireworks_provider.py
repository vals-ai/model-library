from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from model_library.base import LLMConfig, QueryResult, TextInput
from model_library.providers.fireworks import FireworksModel


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fireworks_delegate_query_passes_kwargs():
    delegate_impl = AsyncMock(return_value=QueryResult(output_text="ok"))
    delegate = MagicMock()
    delegate._query_impl = delegate_impl

    with (
        patch(
            "model_library.providers.fireworks.create_openai_client_with_defaults",
            return_value=MagicMock(),
        ) as mock_client_factory,
        patch(
            "model_library.providers.fireworks.model_library_settings"
        ) as mock_settings,
        patch(
            "model_library.providers.fireworks.OpenAIModel", return_value=delegate
        ) as mock_openai,
    ):
        mock_settings.FIREWORKS_API_KEY = "test-key"

        model = FireworksModel("glm-4p5", config=LLMConfig())

        result = await model._query_impl(
            [TextInput(text="hello")],
            tools=[],
            extra="value",
        )

    mock_client_factory.assert_called_once()
    mock_openai.assert_called_once()

    delegate_impl.assert_awaited_once()
    await_args = delegate_impl.await_args
    assert len(await_args.args) == 1
    assert isinstance(await_args.args[0], list)
    assert await_args.kwargs["tools"] == []
    assert await_args.kwargs["extra"] == "value"
    assert result.output_text == "ok"

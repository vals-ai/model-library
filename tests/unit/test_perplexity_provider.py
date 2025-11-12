from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from model_library.base import LLMConfig, QueryResult, TextInput
from model_library.providers.perplexity import PerplexityModel
from model_library.register_models import get_model_registry


@pytest.mark.unit
def test_perplexity_models_registered():
    registry = get_model_registry()
    expected_models = {
        "perplexity/sonar",
        "perplexity/sonar-pro",
        "perplexity/sonar-reasoning",
        "perplexity/sonar-reasoning-pro",
        "perplexity/sonar-deep-research",
    }

    assert expected_models.issubset(registry.keys())
    for name in expected_models:
        model = registry[name]
        assert model.provider_name == "perplexity"
        assert model.provider_endpoint in {
            "sonar",
            "sonar-pro",
            "sonar-reasoning",
            "sonar-reasoning-pro",
            "sonar-deep-research",
        }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_perplexity_delegate_query():
    mock_client = MagicMock()
    mock_client.base_url = "https://api.perplexity.ai"

    with (
        patch(
            "model_library.providers.perplexity.create_openai_client_with_defaults",
            return_value=mock_client,
        ) as mock_factory,
        patch(
            "model_library.providers.perplexity.model_library_settings"
        ) as mock_settings,
    ):
        mock_settings.PERPLEXITY_API_KEY = "test-key"

        model = PerplexityModel("sonar", config=LLMConfig())

        assert model.delegate is not None
        assert model.native is False
        assert model.delegate.get_client() is mock_client
        mock_factory.assert_called_once_with(
            api_key="test-key", base_url="https://api.perplexity.ai"
        )

        query_result = QueryResult(output_text="ok")
        model.delegate._query_impl = AsyncMock(return_value=query_result)  # type: ignore[method-assign]

        result = await model._query_impl([TextInput(text="hello")], tools=[])

        delegate_impl = model.delegate._query_impl  # type: ignore[attr-defined]
        delegate_impl.assert_awaited_once()
        await_args = delegate_impl.await_args
        # delegate should receive normalized arguments
        assert isinstance(await_args.args[0], list)
        assert await_args.kwargs["tools"] == []
        assert result is query_result

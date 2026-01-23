import pytest
from unittest.mock import AsyncMock, patch

from model_library.base import LLMConfig, QueryResult
from model_library.providers.zai import ZAIConfig, ZAIModel
from model_library.registry_utils import get_registry_model


@pytest.mark.parametrize("clear_thinking", [True, False])
async def test_query_impl_passes_extra_body(clear_thinking: bool):
    """Test that _query_impl passes extra_body to delegate_query."""
    model = ZAIModel(
        "glm-4.7-flashx",
        config=LLMConfig(
            reasoning=True,
            provider_config=ZAIConfig(clear_thinking=clear_thinking),
        ),
    )

    with patch.object(
        model,
        "delegate_query",
        new_callable=AsyncMock,
    ) as mock_delegate_query:
        mock_delegate_query.return_value = QueryResult()

        await model.query("test prompt")

        mock_delegate_query.assert_called_once()
        call_kwargs = mock_delegate_query.call_args.kwargs
        assert call_kwargs["extra_body"] == model.get_extra_body()
        assert call_kwargs["extra_body"]["thinking"]["clear_thinking"] == clear_thinking
        assert call_kwargs["extra_body"]["thinking"]["type"] == "enabled"


async def test_clear_thinking_default_true_for_glm_4_7():
    """Test that clear_thinking defaults to True for zai/glm-4.7."""
    model = get_registry_model("zai/glm-4.7")

    assert model.clear_thinking is True
    assert model.get_extra_body()["thinking"]["clear_thinking"] is True


async def test_clear_thinking_false_for_glm_4_7_flashx():
    """Test that clear_thinking is False for zai/glm-4.7-flashx (set in YAML config)."""
    model = get_registry_model("zai/glm-4.7-flashx")

    assert model.clear_thinking is False
    assert model.get_extra_body()["thinking"]["clear_thinking"] is False

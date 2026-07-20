from pydantic import SecretStr

from model_library.base import LLMConfig
from model_library.providers.delegates.thinkingmachines import ThinkingMachinesModel
def test_reasoning_effort_preserves_numeric_string():
    model = ThinkingMachinesModel(
        "thinkingmachines/test-model",
        config=LLMConfig(
            reasoning_effort="0.99",
            custom_api_key=SecretStr("test-key"),
        ),
    )

    assert model._get_extra_body()["reasoning_effort"] == "0.99"

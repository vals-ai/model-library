from model_library.base import LLM
from model_library.base.input import ToolResult
from model_library.registry_utils import get_registry_model
from tests.conftest import parametrize_all_models
from tests.test_helpers import get_example_tool_input


@parametrize_all_models
async def test_serialize_deserialize_history_roundtrip(model_key: str):
    """Test serialize/deserialize roundtrip"""

    model = get_registry_model(model_key)

    tools = []
    if model.supports_tools:
        input, system_prompt, tools = get_example_tool_input()

        response = await model.query(input, tools=tools, system_prompt=system_prompt)
    else:
        response = await model.query("Tell me a joke")

    serialized = LLM.serialize_input(response.history)

    assert isinstance(serialized, str)
    assert len(serialized) > 0

    dinput = LLM.deserialize_input(serialized)

    print(response.history)
    print(dinput)

    assert dinput == response.history

    if response.tool_calls:
        tool_results = [
            ToolResult(tool_call=tool_call, result="low")
            for tool_call in response.tool_calls
        ]
        await model.query([*dinput, *tool_results], tools=tools)

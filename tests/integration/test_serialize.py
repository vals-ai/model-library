from typing import Any

from model_library.base import LLM
from model_library.base.input import ToolResult
from model_library.registry_utils import get_registry_model
from tests.conftest import parametrize_all_models
from tests.test_helpers import get_example_tool_input


def deep_diff(a: Any, b: Any, path: str = "root") -> list[str]:
    from google.protobuf.message import Message
    from pydantic import BaseModel

    diffs: list[str] = []

    if type(a) is not type(b):
        diffs.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
        return diffs

    if isinstance(a, Message):
        if a.SerializeToString() != b.SerializeToString():
            diffs.append(f"{path}: protobuf bytes differ")
        return diffs

    if isinstance(a, BaseModel):
        for f in a.model_fields:
            diffs.extend(deep_diff(getattr(a, f), getattr(b, f), f"{path}.{f}"))
        return diffs

    if hasattr(a, "__dict__"):
        all_keys = set(a.__dict__) | set(b.__dict__)
        for k in all_keys:
            if k not in a.__dict__:
                diffs.append(f"{path}.{k}: missing in first")
            elif k not in b.__dict__:
                diffs.append(f"{path}.{k}: missing in second")
            else:
                diffs.extend(deep_diff(a.__dict__[k], b.__dict__[k], f"{path}.{k}"))
        return diffs

    if isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"{path}: list length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(deep_diff(x, y, f"{path}[{i}]"))
        return diffs

    if isinstance(a, dict):
        all_keys = set(a) | set(b)
        for k in all_keys:
            if k not in a:
                diffs.append(f"{path}[{k!r}]: missing in first")
            elif k not in b:
                diffs.append(f"{path}[{k!r}]: missing in second")
            else:
                diffs.extend(deep_diff(a[k], b[k], f"{path}[{k!r}]"))
        return diffs

    if a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")

    return diffs


@parametrize_all_models
async def test_serialize_deserialize_history_roundtrip(model_key: str):
    """Test serialize/deserialize roundtrip"""

    model = get_registry_model(model_key)

    if model.supports_tools:
        input, system_prompt, tools = get_example_tool_input()

        response = await model.query(input, tools=tools, system_prompt=system_prompt)
    else:
        response = await model.query("Tell me a joke")

    serialized = LLM.serialize_input(response.history)

    assert isinstance(serialized, bytes)
    assert len(serialized) > 0

    dinput = LLM.deserialize_input(serialized)

    print(response.history)
    print(dinput)

    if "grok" in model_key:
        diffs = deep_diff(dinput, response.history)
        if diffs:
            for d in diffs:
                print(d)
            assert False, f"Found {len(diffs)} differences"
    else:
        assert dinput == response.history

    if response.tool_calls:
        tool_result = ToolResult(tool_call=response.tool_calls[0], result="low")
        await model.query([*dinput, tool_result])

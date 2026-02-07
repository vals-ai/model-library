"""
Unit tests for serialize/deserialize functionality
"""

import tempfile
from pathlib import Path

from model_library.base.base import LLM
from tests.test_helpers import get_example_tool_input


async def test_serialize_deserialize_roundtrip():
    """Test serialize/deserialize roundtrip"""

    input, _, _ = get_example_tool_input()

    serialized = LLM.serialize_input(input)

    assert isinstance(serialized, bytes)
    assert len(serialized) > 0

    dinput = LLM.deserialize_input(serialized)

    assert dinput == input


async def test_serialize_empty_input():
    """Test serialization with empty input"""

    serialized = LLM.serialize_input([])

    dinput = LLM.deserialize_input(serialized)

    assert dinput == []


async def test_deserialize_from_file():
    """Test deserializing from a file path"""

    input, _, _ = get_example_tool_input()

    serialized = LLM.serialize_input(input)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        f.write(serialized)
        temp_path = f.name

    temp_path = Path(temp_path)

    dinput = LLM.deserialize_input(temp_path)

    assert dinput == input

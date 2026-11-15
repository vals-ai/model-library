import json

import pytest

from model_library.base.input import RawResponse, SystemInput, TextInput
from model_library.base import LLM


async def test_sign_verify_roundtrip_preserves_types():
    """History round-trip must preserve InputItem subtypes."""
    items = [SystemInput(text="sys"), TextInput(text="hello")]
    signed = LLM.serialize_input(items, secret=b"test-secret")
    restored = LLM.deserialize_input(signed, secret=b"test-secret")

    assert isinstance(restored[0], SystemInput)
    assert isinstance(restored[1], TextInput)
    assert restored[1].text == "hello"


async def test_sign_verify_roundtrip_with_raw_response():
    """RawResponse objects survive the sign/verify round-trip."""

    provider_obj = {"role": "assistant", "content": "hi"}
    items = [RawResponse(response=provider_obj), TextInput(text="hello")]
    signed = LLM.serialize_input(items, secret=b"test-secret")
    restored = LLM.deserialize_input(signed, secret=b"test-secret")

    assert isinstance(restored[0], RawResponse)
    assert restored[0].response == provider_obj


async def test_tampered_pickle_rejected():
    """Tampering with a pickled blob's HMAC tag is detected."""

    items = [RawResponse(response={"key": "value"})]
    signed = LLM.serialize_input(items, secret=b"test-secret")

    # Tamper with the HMAC tag on the pickled field
    data = json.loads(signed)
    data[0]["response"]["hmac"] = "deadbeef" * 8
    tampered = json.dumps(data)

    with pytest.raises(ValueError, match="HMAC verification failed"):
        LLM.deserialize_input(tampered, secret=b"test-secret")


async def test_wrong_secret_rejected():
    """Different secrets produce different HMACs — verification fails."""

    items = [RawResponse(response={"key": "value"})]
    signed = LLM.serialize_input(items, secret=b"secret-a")

    with pytest.raises(ValueError, match="HMAC verification failed"):
        LLM.deserialize_input(signed, secret=b"secret-b")


async def test_plain_text_items_work_without_hmac_fields():
    """Items without pickled fields (TextInput, SystemInput) don't need HMAC."""

    items = [TextInput(text="hello"), SystemInput(text="sys")]
    signed = LLM.serialize_input(items, secret=b"test-secret")

    # Verify it's valid JSON with no pickle/hmac fields
    data = json.loads(signed)
    assert "pickle" not in str(data[0])

    restored = LLM.deserialize_input(signed, secret=b"test-secret")
    assert len(restored) == 2

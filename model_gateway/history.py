"""History serialization for the model proxy server.

Uses LLM.serialize_input/deserialize_input with an HMAC secret so that
pickled RawResponse/RawInput fields are signed and verified.
"""

# TODO: Key rotation — currently uses a fixed HMAC secret from Secrets Manager.
# Future improvement: derive a new signing key on each server boot (e.g.,
# HMAC-SHA256(aws_secret, boot_id)), sign with the newest key, but verify
# against a rolling window of previous keys (e.g., last 2 hours).  This lets
# us rotate secrets without invalidating in-flight histories.

from collections.abc import Sequence

from model_library.base.base import LLM
from model_library.base.input import InputItem


def sign_history(items: Sequence[InputItem], *, secret: bytes | None = None) -> str:
    """Serialize history items to a JSON string, optionally signing pickle blobs."""
    return LLM.serialize_input(items, secret=secret)


def verify_and_load_history(
    signed_json: str | None, *, secret: bytes | None = None
) -> list[InputItem]:
    """Deserialize history, verifying HMAC when a secret is provided."""
    if not signed_json:
        return []
    return LLM.deserialize_input(signed_json, secret=secret)

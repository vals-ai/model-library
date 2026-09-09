"""Botocore ConverseStream raw-byte capture."""

import logging

from collections.abc import Iterable
from contextvars import ContextVar
from typing import Any

from model_library.failure_capture.core import Attempt, current_capture

logger = logging.getLogger(__name__)

_botocore_attempt: ContextVar[Attempt | None] = ContextVar(
    "provider_exchange_botocore_attempt", default=None
)


class CapturingBotocoreRawStream:
    def __init__(self, stream: Any, attempt: Attempt) -> None:
        self._stream = stream
        self._attempt = attempt

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def stream(self, *args: object, **kwargs: object) -> Iterable[bytes]:
        try:
            for chunk in self._stream.stream(*args, **kwargs):
                self._attempt.response_bytes(
                    chunk, representation="bedrock_aws_event_stream_bytes"
                )
                yield chunk
            self._attempt.finish()
        except BaseException as exc:
            self._attempt.fail(exc)
            raise


def install_botocore_capture(client: Any) -> None:
    """Install instance-local ConverseStream event hooks."""
    events = client.meta.events

    def clear() -> Attempt | None:
        attempt = _botocore_attempt.get()
        _botocore_attempt.set(None)
        return attempt

    def before_send(*, request: Any, **_kwargs: object) -> None:
        active = current_capture()
        if active is None:
            return
        attempt = active.attempt("botocore")
        _botocore_attempt.set(attempt)
        attempt.request_bytes(
            request.body,
            representation="bedrock_serialized_request_body_bytes",
        )

    def before_parse(*, response_dict: dict[str, Any], **_kwargs: object) -> None:
        attempt = clear()
        if attempt is None:
            return
        status = response_dict["status_code"]
        attempt.response_start(status)
        body = response_dict["body"]
        if status < 300:
            response_dict["body"] = CapturingBotocoreRawStream(body, attempt)
            return
        attempt.response_bytes(
            body, representation="bedrock_http_error_response_entity_bytes"
        )
        attempt.finish()

    def needs_retry(
        *,
        caught_exception: BaseException | None = None,
        **_kwargs: object,
    ) -> None:
        attempt = clear()
        if attempt is not None and caught_exception is not None:
            attempt.fail(caught_exception)

    try:
        for event, handler in (
            ("before-send", before_send),
            ("before-parse", before_parse),
            ("needs-retry", needs_retry),
        ):
            events.register(f"{event}.bedrock-runtime.ConverseStream", handler)
    except Exception:
        active = current_capture()
        if active is not None:
            active.record_error()
        logger.warning("Failed to register Bedrock ConverseStream capture hook")

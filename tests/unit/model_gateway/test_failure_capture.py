import asyncio
import base64
import gzip
import io
import json
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock

import pytest

import model_gateway.route_helpers as route_helpers
from model_gateway.metrics import model_dimensions
from model_gateway.route_helpers import GatewayOperation
from model_gateway.types import ProviderError
from model_library.base import (
    FileWithBase64,
    FileWithBytes,
    FileWithUrl,
    normalize_query_input,
)
from model_library.exceptions import RetryException
from model_library.failure_capture import core
import model_library.telemetry as telemetry
from model_library.retriers.backoff import ExponentialBackoffRetrier


def _operation(operation: str = "query") -> GatewayOperation:
    return GatewayOperation(
        operation=operation,
        dimensions=model_dimensions(operation="query", model="test/model", config={}),
        start=0.0,
        provider="test",
    )


def _records(data: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in gzip.decompress(data).splitlines()]


def _payload(records: list[dict[str, object]], event: str) -> bytes:
    return b"".join(
        base64.b64decode(str(record["bytes_base64"]))
        for record in records
        if record["event"] == event
    )


class _Scope:
    def __init__(self, attachments: list[bytes]) -> None:
        self.attachments = attachments

    def add_attachment(self, *, bytes: bytes, filename: str, content_type: str) -> None:
        assert filename == core.ATTACHMENT_FILENAME
        assert content_type == "application/gzip"
        self.attachments.append(bytes)


class _Sentry:
    def __init__(self) -> None:
        self.attachments: list[bytes] = []
        self.exceptions: list[BaseException] = []
        self.fail = False

    @contextmanager
    def new_scope(self):
        yield _Scope(self.attachments)

    def capture_exception(self, exc: BaseException) -> None:
        self.exceptions.append(exc)
        if self.fail:
            raise RuntimeError("Sentry unavailable")


@pytest.fixture
def runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_Sentry, list[core.CaptureStats]]:
    sentry = _Sentry()
    stats: list[core.CaptureStats] = []
    monkeypatch.setattr(
        telemetry, "start_span", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(telemetry, "_enabled", True)
    monkeypatch.setattr(telemetry, "_current_recording_span", lambda: None)
    monkeypatch.setattr(telemetry, "import_module", lambda _name: sentry)
    monkeypatch.setattr(route_helpers, "record_gateway_phase", lambda **_kwargs: None)
    monkeypatch.setattr(
        route_helpers, "emit_model_error", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        route_helpers,
        "emit_failure_capture_metrics",
        lambda _dims, value: stats.append(value),
    )
    return sentry, stats


async def _captured_success(payload: bytes = b"response") -> str:
    active = core.current_capture()
    assert active is not None
    attempt = active.attempt("test")
    attempt.request_bytes(b"request", representation="test_request_bytes")
    attempt.response_start(200)
    attempt.response_bytes(payload, representation="test_response_bytes")
    attempt.finish()
    return "ok"


async def _captured_failure(exc: Exception, calls: list[int]) -> str:
    active = core.current_capture()
    assert active is not None
    attempt = active.attempt("test")
    calls.append(attempt.id)
    attempt.request_bytes(f"request-{attempt.id}".encode(), representation="test")
    attempt.response_bytes(f"response-{attempt.id}".encode(), representation="test")
    attempt.fail(exc)
    raise exc


async def test_flag_off_creates_no_spool_and_thrown_failure_has_one_event(
    monkeypatch: pytest.MonkeyPatch,
    runtime: tuple[_Sentry, list[core.CaptureStats]],
) -> None:
    monkeypatch.delenv("GATEWAY_FAILED_EXCHANGE_CAPTURE_ENABLED", raising=False)
    monkeypatch.setattr(
        core.tempfile,
        "SpooledTemporaryFile",
        lambda **_kwargs: pytest.fail("disabled capture created a spool"),
    )
    sentry, _ = runtime
    retrier = ExponentialBackoffRetrier(logging.getLogger("test"), max_tries=1)
    result = await _operation().provider_call(
        retrier.execute(AsyncMock(side_effect=RetryException("failed"))),
        span_attrs={},
    )
    assert isinstance(result, ProviderError)
    assert len(sentry.exceptions) == 1
    assert sentry.attachments == []


async def test_direct_retrier_reporting_is_unchanged_outside_gateway(
    monkeypatch: pytest.MonkeyPatch,
    runtime: tuple[_Sentry, list[core.CaptureStats]],
) -> None:
    monkeypatch.setenv("GATEWAY_FAILED_EXCHANGE_CAPTURE_ENABLED", "true")
    sentry, _ = runtime
    retrier = ExponentialBackoffRetrier(logging.getLogger("test"), max_tries=1)
    with pytest.raises(RetryException):
        await retrier.execute(AsyncMock(side_effect=RetryException("failed")))
    assert len(sentry.exceptions) == 1


async def test_enabled_success_discards_without_compression_or_attachment(
    monkeypatch: pytest.MonkeyPatch,
    runtime: tuple[_Sentry, list[core.CaptureStats]],
) -> None:
    monkeypatch.setenv("GATEWAY_FAILED_EXCHANGE_CAPTURE_ENABLED", "true")
    sentry, stats = runtime
    named_tempfile = AsyncMock()
    monkeypatch.setattr(core.tempfile, "NamedTemporaryFile", named_tempfile)
    result = await _operation().provider_call(_captured_success(), span_attrs={})
    assert result == "ok"
    assert sentry.exceptions == sentry.attachments == []
    assert named_tempfile.call_count == 0
    assert len(stats) == 1
    assert stats[0].compressed_bytes == 0


async def test_retry_final_failure_has_ordered_attempts_and_one_artifact(
    monkeypatch: pytest.MonkeyPatch,
    runtime: tuple[_Sentry, list[core.CaptureStats]],
) -> None:
    monkeypatch.setenv("GATEWAY_FAILED_EXCHANGE_CAPTURE_ENABLED", "true")
    monkeypatch.setattr("model_library.retriers.base.asyncio.sleep", AsyncMock())
    sentry, stats = runtime
    calls: list[int] = []
    error = RetryException("failed")
    retrier = ExponentialBackoffRetrier(
        logging.getLogger("test"), max_tries=2, initial=0
    )
    result = await _operation().provider_call(
        retrier.execute(_captured_failure, error, calls), span_attrs={}
    )
    assert isinstance(result, ProviderError)
    assert calls == [1, 2]
    assert sentry.exceptions == [error]
    assert len(sentry.attachments) == 1
    assert sentry.attachments[0][4:9] == b"\0\0\0\0\x04"
    records = _records(sentry.attachments[0])
    assert [r["attempt"] for r in records if r["event"] == "attempt.start"] == [1, 2]
    assert _payload(records, "request.bytes") == b"request-1request-2"
    assert _payload(records, "response.bytes") == b"response-1response-2"
    assert stats[0].attempts == 2


async def test_cancellation_attaches_partial_capture_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
    runtime: tuple[_Sentry, list[core.CaptureStats]],
) -> None:
    monkeypatch.setenv("GATEWAY_FAILED_EXCHANGE_CAPTURE_ENABLED", "true")
    sentry, _ = runtime

    async def cancel() -> None:
        active = core.current_capture()
        assert active is not None
        attempt = active.attempt("test")
        attempt.response_bytes(b"partial", representation="test")
        attempt.fail(asyncio.CancelledError())
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await _operation().provider_call(cancel(), span_attrs={})

    assert len(sentry.exceptions) == 1
    assert _payload(_records(sentry.attachments[0]), "response.bytes") == b"partial"


async def test_cancellation_records_event_when_capture_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
    runtime: tuple[_Sentry, list[core.CaptureStats]],
) -> None:
    monkeypatch.setenv("GATEWAY_FAILED_EXCHANGE_CAPTURE_ENABLED", "true")
    monkeypatch.setattr(
        core.tempfile,
        "SpooledTemporaryFile",
        MagicMock(side_effect=OSError("spool")),
    )
    sentry, _ = runtime
    cancelled = asyncio.CancelledError()

    async def cancel() -> None:
        raise cancelled

    with pytest.raises(asyncio.CancelledError) as raised:
        await _operation().provider_call(cancel(), span_attrs={})

    assert raised.value is cancelled
    assert sentry.exceptions == [cancelled]
    assert sentry.attachments == []


class _FailingSpool:
    def __init__(self) -> None:
        self._spool = io.BytesIO()

    def write(self, _data: bytes) -> int:
        raise OSError("write failed")

    def __getattr__(self, name: str):
        return getattr(self._spool, name)


@pytest.mark.parametrize(
    "boundary", ["spool", "write", "finalize", "attachment", "sentry"]
)
async def test_capture_boundary_failures_preserve_provider_outcome(
    boundary: str,
    monkeypatch: pytest.MonkeyPatch,
    runtime: tuple[_Sentry, list[core.CaptureStats]],
) -> None:
    monkeypatch.setenv("GATEWAY_FAILED_EXCHANGE_CAPTURE_ENABLED", "true")
    sentry, _ = runtime
    sentry.fail = boundary == "sentry"
    if boundary == "spool":
        monkeypatch.setattr(
            core.tempfile,
            "SpooledTemporaryFile",
            MagicMock(side_effect=OSError("spool")),
        )
    elif boundary == "write":
        monkeypatch.setattr(
            core.tempfile, "SpooledTemporaryFile", lambda **_kwargs: _FailingSpool()
        )
    elif boundary == "finalize":
        monkeypatch.setattr(
            core.tempfile, "NamedTemporaryFile", MagicMock(side_effect=OSError("gzip"))
        )
    elif boundary == "attachment":
        monkeypatch.setattr(
            _Scope,
            "add_attachment",
            MagicMock(side_effect=OSError("attachment")),
        )

    original = RuntimeError("provider failed")

    async def fail() -> None:
        active = core.current_capture()
        if active is not None:
            attempt = active.attempt("test")
            attempt.request_bytes(b"request", representation="test")
            attempt.fail(original)
        raise original

    result = await _operation().provider_call(fail(), span_attrs={})

    assert isinstance(result, ProviderError)
    assert sentry.exceptions == [original]


async def test_request_redaction_preserves_non_file_bytes_across_chunks(
    monkeypatch: pytest.MonkeyPatch,
    runtime: tuple[_Sentry, list[core.CaptureStats]],
) -> None:
    monkeypatch.setenv("GATEWAY_FAILED_EXCHANGE_CAPTURE_ENABLED", "true")
    sentry, stats = runtime
    image = b"private-image" * 20
    encoded_image = base64.b64encode(image)
    raw_file = b"private-file" * 20
    unrelated_base64 = base64.b64encode(b"unrelated-public-value")
    body = (
        b'{"prompt":"keep this prompt","image":"data:image/png;base64,'
        + encoded_image
        + b'","file":"'
        + raw_file
        + b'","unrelated":"'
        + unrelated_base64
        + b'","temperature":0}'
    )
    inputs = [
        FileWithBase64(
            type="image",
            name="image.png",
            mime="png",
            base64=encoded_image.decode(),
        ),
        FileWithBytes(
            type="file",
            name="file.bin",
            mime="application/octet-stream",
            data=raw_file,
        ),
        FileWithUrl(
            type="image",
            name="remote.png",
            mime="png",
            url="https://example.com/image.png",
        ),
    ]

    async def fail() -> None:
        active = core.current_capture()
        assert active is not None
        normalize_query_input(inputs)
        normalize_query_input(inputs)
        attempt = active.attempt("test")
        split = body.index(encoded_image) + len(encoded_image) // 2
        attempt.request_bytes(body[:split], representation="test_request")
        attempt.request_bytes(body[split:], representation="test_request")
        attempt.response_bytes(b"provider response", representation="test_response")
        attempt.fail(RuntimeError("failed"))
        raise RuntimeError("failed")

    result = await _operation().provider_call(fail(), span_attrs={})

    assert isinstance(result, ProviderError)
    records = _records(sentry.attachments[0])
    request = _payload(records, "request.bytes")
    request_record = next(
        record for record in records if record["event"] == "request.bytes"
    )
    assert len(request) == len(body)
    assert request_record["byte_count"] == len(body)
    assert request_record["redacted_files"] == 2
    assert b"keep this prompt" in request
    assert b'"temperature":0' in request
    assert b"<image omitted;" in request
    assert b"<file omitted;" in request
    assert b'"unrelated":"' + unrelated_base64 + b'"' in request
    assert encoded_image not in request
    assert image not in request
    assert raw_file not in request
    assert _payload(records, "response.bytes") == b"provider response"
    assert stats[0].request_bytes == len(body)
    assert stats[0].errors == 0


@pytest.mark.parametrize("encoded", ["not-valid-base64!", "Zh=="])
async def test_malformed_typed_base64_fails_closed(
    encoded: str,
    monkeypatch: pytest.MonkeyPatch,
    runtime: tuple[_Sentry, list[core.CaptureStats]],
) -> None:
    monkeypatch.setenv("GATEWAY_FAILED_EXCHANGE_CAPTURE_ENABLED", "true")
    sentry, stats = runtime
    body = b'{"image":"' + encoded.encode() + b'"}'

    async def fail() -> None:
        active = core.current_capture()
        assert active is not None
        normalize_query_input(
            [
                FileWithBase64(
                    type="image",
                    name="image.png",
                    mime="png",
                    base64=encoded,
                )
            ]
        )
        attempt = active.attempt("test")
        attempt.request_bytes(body, representation="test_request")
        attempt.response_bytes(b"provider response", representation="test_response")
        attempt.fail(RuntimeError("failed"))
        raise RuntimeError("failed")

    result = await _operation().provider_call(fail(), span_attrs={})

    assert isinstance(result, ProviderError)
    records = _records(sentry.attachments[0])
    request_record = next(
        record for record in records if record["event"] == "request.bytes"
    )
    assert "bytes_base64" not in request_record
    assert request_record["byte_count"] == len(body)
    assert request_record["payload_omitted"] == "file_redaction_unverified"
    assert _payload(records, "response.bytes") == b"provider response"
    assert stats[0].request_bytes == len(body)
    assert stats[0].errors == 1

def test_registered_files_redact_raw_and_canonical_wire_forms() -> None:
    raw_from_base64 = b"raw-from-base64"
    raw_file = b"raw-file-value" * 3
    encoded_file = base64.b64encode(raw_file)
    inputs = [
        FileWithBase64(
            type="image",
            name="image.png",
            mime="png",
            base64=base64.b64encode(raw_from_base64).decode(),
        ),
        FileWithBytes(
            type="file",
            name="file.bin",
            mime="application/octet-stream",
            data=raw_file,
        ),
    ]
    body = b"keep:" + raw_from_base64 + b";" + encoded_file + b":end"

    with core.capture(enabled=True) as active:
        assert active is not None
        normalize_query_input(inputs)
        attempt = active.attempt("test")
        attempt.request_bytes(body, representation="test_request")
        attempt.finish()
        artifact = active.finalize()
        assert artifact is not None
        records = _records(artifact.path.read_bytes())

    request = _payload(records, "request.bytes")
    assert request.startswith(b"keep:")
    assert request.endswith(b":end")
    assert len(request) == len(body)
    assert raw_from_base64 not in request
    assert encoded_file not in request
    assert b"<image omitted;" in request
    assert b"<file omitted;" in request


def test_overlapping_file_payloads_are_fully_redacted() -> None:
    with core.capture(enabled=True) as active:
        assert active is not None
        active.register_request_payload(b"ABC", "short")
        active.register_request_payload(b"ABCDEF", "long")
        attempt = active.attempt("test")
        body = b'[{"data":"QUJDREVG"}]'
        attempt.request_bytes(body, representation="test_request")
        attempt.finish()
        artifact = active.finalize()
        assert artifact is not None
        records = _records(artifact.path.read_bytes())

    request = _payload(records, "request.bytes")
    request_record = next(record for record in records if record["event"] == "request.bytes")
    assert len(request) == len(body)
    assert b"QUJD" not in request
    assert b"QUJDREVG" not in request
    assert b"<long" in request
    assert request_record["redacted_files"] == 2



def test_redaction_preserves_separate_retry_attempts() -> None:
    payload = b"private-image"
    encoded = base64.b64encode(payload)
    bodies = [b"attempt-one:" + encoded, b"attempt-two:" + encoded]

    with core.capture(enabled=True) as active:
        assert active is not None
        active.register_request_payload(payload, "image")
        for attempt_number, body in enumerate(bodies, start=1):
            attempt = active.attempt("test")
            split = body.index(encoded) + 3
            attempt.request_bytes(body[:split], representation="test_request")
            attempt.request_bytes(body[split:], representation="test_request")
            attempt.fail(RuntimeError(f"attempt {attempt_number}"))
        artifact = active.finalize()
        assert artifact is not None
        records = _records(artifact.path.read_bytes())

    request_records = [record for record in records if record["event"] == "request.bytes"]
    assert [record["attempt"] for record in request_records] == [1, 2]
    assert [record["byte_count"] for record in request_records] == list(map(len, bodies))
    redacted = [_payload([record], "request.bytes") for record in request_records]
    assert all(len(body) == len(original) for body, original in zip(redacted, bodies))
    assert all(encoded not in body for body in redacted)




@pytest.mark.parametrize("operation", ["files_upload", "audio_transcriptions"])
async def test_file_operations_record_request_lengths_without_payloads(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
    runtime: tuple[_Sentry, list[core.CaptureStats]],
) -> None:
    monkeypatch.setenv("GATEWAY_FAILED_EXCHANGE_CAPTURE_ENABLED", "true")
    sentry, stats = runtime
    chunks = [b"private-file-one", b"private-file-two"]

    async def fail() -> None:
        active = core.current_capture()
        assert active is not None
        attempt = active.attempt("test")
        for chunk in chunks:
            attempt.request_bytes(chunk, representation="test_request")
        attempt.response_bytes(b"provider response", representation="test_response")
        attempt.fail(RuntimeError("failed"))
        raise RuntimeError("failed")

    result = await _operation(operation).provider_call(fail(), span_attrs={})

    assert isinstance(result, ProviderError)
    records = _records(sentry.attachments[0])
    request_records = [
        record for record in records if record["event"] == "request.bytes"
    ]
    assert all("bytes_base64" not in record for record in request_records)
    assert [record["byte_count"] for record in request_records] == list(
        map(len, chunks)
    )
    assert {
        record["payload_omitted"] for record in request_records
    } == {"inline_file_operation"}
    assert _payload(records, "response.bytes") == b"provider response"
    assert stats[0].request_bytes == sum(map(len, chunks))


async def test_unmatched_request_redaction_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    runtime: tuple[_Sentry, list[core.CaptureStats]],
) -> None:
    monkeypatch.setenv("GATEWAY_FAILED_EXCHANGE_CAPTURE_ENABLED", "true")
    sentry, stats = runtime
    body = b'{"prompt":"keep but omit capture"}'

    async def fail() -> None:
        active = core.current_capture()
        assert active is not None
        active.register_request_payload("c2VjcmV0", "file")
        attempt = active.attempt("test")
        attempt.request_bytes(body, representation="test_request")
        attempt.response_start(400)
        attempt.response_bytes(b"provider response", representation="test_response")
        attempt.fail(RuntimeError("failed"))
        raise RuntimeError("failed")

    result = await _operation().provider_call(fail(), span_attrs={})

    assert isinstance(result, ProviderError)
    records = _records(sentry.attachments[0])
    request_record = next(
        record for record in records if record["event"] == "request.bytes"
    )
    assert "bytes_base64" not in request_record
    assert request_record["byte_count"] == len(body)
    assert request_record["payload_omitted"] == "file_redaction_unverified"
    assert _payload(records, "response.bytes") == b"provider response"
    assert stats[0].request_bytes == len(body)
    assert stats[0].errors == 1


async def test_oversize_metrics_are_content_free(
    monkeypatch: pytest.MonkeyPatch,
    runtime: tuple[_Sentry, list[core.CaptureStats]],
) -> None:
    monkeypatch.setenv("GATEWAY_FAILED_EXCHANGE_CAPTURE_ENABLED", "true")
    monkeypatch.setattr(core, "ATTACHMENT_LIMIT_BYTES", 1)
    sentry, stats = runtime
    sentinel = b"payload-sentinel"

    async def fail() -> None:
        active = core.current_capture()
        assert active is not None
        attempt = active.attempt("test")
        attempt.request_bytes(sentinel, representation="test")
        raise RuntimeError("failed")

    result = await _operation().provider_call(fail(), span_attrs={})

    assert isinstance(result, ProviderError)
    assert sentry.attachments == []
    assert "payload-sentinel" not in json.dumps(asdict(stats[0]))


async def test_concurrent_captures_are_isolated() -> None:
    ready = asyncio.Event()

    async def run(payload: bytes) -> bytes:
        with core.capture(enabled=True) as active:
            assert active is not None
            attempt = active.attempt("test")
            attempt.response_bytes(payload, representation="test")
            ready.set()
            await asyncio.sleep(0)
            assert core.current_capture() is active
            artifact = active.finalize()
            assert artifact is not None
            return artifact.path.read_bytes()

    first = asyncio.create_task(run(b"first"))
    await ready.wait()
    second = asyncio.create_task(run(b"second"))
    first_data, second_data = await asyncio.gather(first, second)

    assert _payload(_records(first_data), "response.bytes") == b"first"
    assert _payload(_records(second_data), "response.bytes") == b"second"
    assert core.current_capture() is None

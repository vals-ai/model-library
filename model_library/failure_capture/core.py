from __future__ import annotations

# Attempt and capture() are the only collaborators for Capture internals.
# pyright: reportPrivateUsage=false

import base64
import gzip
import json
import shutil
import tempfile
import time
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import IO

SPOOL_MEMORY_LIMIT_BYTES = 1 * 1024 * 1024
ATTACHMENT_LIMIT_BYTES = 20 * 1024 * 1024
ATTACHMENT_FILENAME = "provider-failed-exchange.ndjson.gz"


@dataclass(frozen=True)
class Artifact:
    path: Path
    compressed_bytes: int
    compression_ms: float

    @property
    def attachable(self) -> bool:
        return self.compressed_bytes <= ATTACHMENT_LIMIT_BYTES


@dataclass(frozen=True)
class CaptureStats:
    attempts: int
    request_bytes: int
    response_bytes: int
    response_chunks: int
    raw_bytes: int
    compressed_bytes: int
    compression_ms: float
    errors: int
    transports: tuple[str, ...]


def _request_patterns(value: str | bytes) -> Iterator[bytes]:
    if isinstance(value, str):
        encoded = value.encode("ascii")
        decoded = base64.b64decode(encoded, validate=True)
        if base64.b64encode(decoded) != encoded:
            raise ValueError("noncanonical Base64 payload")
        yield encoded
        yield decoded
        return
    yield base64.b64encode(value)
    yield value


def _replacement(value: str | bytes, kind: str, pattern: bytes) -> bytes:
    unit = "base64_chars" if isinstance(value, str) else "source_bytes"
    marker = f"<{kind} omitted; {unit}={len(value)}>".encode()
    return marker[: len(pattern)].ljust(len(pattern), b"A")


class Attempt:
    def __init__(self, capture: Capture, attempt_id: int) -> None:
        self._capture = capture
        self.id = attempt_id
        self._ended = False
        self._request_chunks: list[tuple[bytes, str]] = []

    def request_bytes(self, payload: bytes, *, representation: str) -> None:
        if self._ended:
            return
        if self._capture.omits_request_payload:
            self._capture._write(
                "request.bytes",
                attempt=self.id,
                representation=representation,
                byte_count=len(payload),
                payload_omitted="inline_file_operation",
            )
            return
        if self._capture.redacts_requests:
            self._request_chunks.append((payload, representation))
            return
        self._capture._write(
            "request.bytes",
            attempt=self.id,
            representation=representation,
            payload=payload,
        )

    def _flush_request(self) -> None:
        if not self._request_chunks:
            return
        chunks, self._request_chunks = self._request_chunks, []
        self._capture._write_redacted_request(self.id, chunks)

    def response_start(self, status: int) -> None:
        if not self._ended:
            self._flush_request()
            self._capture._write("response.start", attempt=self.id, status=status)

    def response_bytes(self, payload: bytes, *, representation: str) -> None:
        if self._ended:
            return
        self._flush_request()
        self._capture._write(
            "response.bytes",
            attempt=self.id,
            representation=representation,
            payload=payload,
        )

    def finish(self, outcome: str = "complete") -> None:
        if self._ended:
            return
        self._flush_request()
        self._ended = True
        self._capture._write("attempt.end", attempt=self.id, outcome=outcome)

    def fail(self, exc: BaseException) -> None:
        if self._ended:
            return
        self._flush_request()
        self._ended = True
        self._capture._write(
            "attempt.end",
            attempt=self.id,
            outcome="error",
            error_type=type(exc).__name__,
        )


class Capture:
    def __init__(self, spool: IO[bytes]) -> None:
        self._spool = spool
        self._request_payloads: list[tuple[str | bytes, str]] = []
        self._omit_request_payload = False
        self._attempts = 0
        self._request_bytes = 0
        self._response_bytes = 0
        self._response_chunks = 0
        self._spool_bytes = 0
        self._errors = 0
        self._transports: set[str] = set()
        self._artifact: Artifact | None = None

    @property
    def omits_request_payload(self) -> bool:
        return self._omit_request_payload

    @property
    def redacts_requests(self) -> bool:
        return bool(self._request_payloads)

    def omit_request_payload(self) -> None:
        self._omit_request_payload = True

    def register_request_payload(self, value: str | bytes, kind: str) -> None:
        if any(payload is value for payload, _kind in self._request_payloads):
            return
        self._request_payloads.append((value, kind))

    def record_error(self) -> None:
        self._errors += 1

    def attempt(self, transport: str) -> Attempt:
        self._attempts += 1
        self._transports.add(transport)
        attempt = Attempt(self, self._attempts)
        self._write("attempt.start", attempt=attempt.id, transport=transport)
        return attempt

    def _redact_request(self, payload: bytes) -> bytes | None:
        spans: list[tuple[int, int, bytes]] = []
        for value, kind in self._request_payloads:
            value_spans: list[tuple[int, int, bytes]] = []
            for pattern in _request_patterns(value):
                if not pattern:
                    continue
                start = payload.find(pattern)
                while start != -1:
                    value_spans.append(
                        (
                            start,
                            start + len(pattern),
                            _replacement(value, kind, pattern),
                        )
                    )
                    start = payload.find(pattern, start + 1)
            if not value_spans:
                return None
            spans.extend(value_spans)

        selected: list[tuple[int, int, bytes]] = []
        for start, end, replacement in sorted(
            spans, key=lambda span: span[1] - span[0], reverse=True
        ):
            overlaps = [
                existing
                for existing in selected
                if start < existing[1] and existing[0] < end
            ]
            if any(start < existing[0] or end > existing[1] for existing in overlaps):
                return None
            if not overlaps:
                selected.append((start, end, replacement))

        sanitized = bytearray(payload)
        for start, end, replacement in sorted(selected, reverse=True):
            sanitized[start:end] = replacement
        return bytes(sanitized)

    def _write_redacted_request(
        self,
        attempt: int,
        chunks: Sequence[tuple[bytes, str]],
    ) -> None:
        byte_count = sum(len(chunk) for chunk, _representation in chunks)
        representation = chunks[0][1]
        try:
            payload = (
                chunks[0][0]
                if len(chunks) == 1
                else b"".join(chunk for chunk, _representation in chunks)
            )
            sanitized = self._redact_request(payload)
        except Exception:
            sanitized = None

        if sanitized is None:
            self._errors += 1
            self._write(
                "request.bytes",
                attempt=attempt,
                representation=representation,
                byte_count=byte_count,
                payload_omitted="file_redaction_unverified",
            )
            return

        self._write(
            "request.bytes",
            attempt=attempt,
            representation=representation,
            payload=sanitized,
            byte_count=byte_count,
            redacted_files=len(self._request_payloads),
        )

    def finalize(self) -> Artifact | None:
        started = time.perf_counter()
        path: Path | None = None
        try:
            self._spool.seek(0)
            with tempfile.NamedTemporaryFile(
                prefix="provider-failed-exchange-",
                suffix=".ndjson.gz",
                delete=False,
            ) as output:
                path = Path(output.name)
                with gzip.GzipFile(
                    fileobj=output, mode="wb", compresslevel=1, mtime=0
                ) as compressed:
                    shutil.copyfileobj(self._spool, compressed)
            compressed_bytes = path.stat().st_size
        except Exception:
            self._errors += 1
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    self._errors += 1
            return None
        assert path is not None
        artifact = Artifact(
            path=path,
            compressed_bytes=compressed_bytes,
            compression_ms=(time.perf_counter() - started) * 1000,
        )
        self._artifact = artifact
        return artifact

    def stats(self) -> CaptureStats:
        artifact = self._artifact
        return CaptureStats(
            attempts=self._attempts,
            request_bytes=self._request_bytes,
            response_bytes=self._response_bytes,
            response_chunks=self._response_chunks,
            raw_bytes=self._spool_bytes,
            compressed_bytes=artifact.compressed_bytes if artifact else 0,
            compression_ms=artifact.compression_ms if artifact else 0.0,
            errors=self._errors,
            transports=tuple(sorted(self._transports)),
        )

    def _write(
        self,
        kind: str,
        *,
        attempt: int,
        representation: str | None = None,
        payload: bytes | None = None,
        byte_count: int | None = None,
        **fields: str | int,
    ) -> None:
        try:
            event: dict[str, str | int] = {"event": kind, "attempt": attempt, **fields}
            if representation is not None:
                event["representation"] = representation
            if payload is not None:
                event["bytes_base64"] = base64.b64encode(payload).decode("ascii")
            if byte_count is None and payload is not None:
                byte_count = len(payload)
            if byte_count is not None:
                event["byte_count"] = byte_count
                if kind == "request.bytes":
                    self._request_bytes += byte_count
                elif kind == "response.bytes":
                    self._response_bytes += byte_count
                    self._response_chunks += 1
            encoded = json.dumps(event, separators=(",", ":")).encode() + b"\n"
            self._spool.write(encoded)
            self._spool_bytes += len(encoded)
        except Exception:
            self._errors += 1

    def _cleanup(self) -> None:
        if self._artifact is not None:
            try:
                self._artifact.path.unlink(missing_ok=True)
            except OSError:
                self._errors += 1
        try:
            self._spool.close()
        except Exception:
            self._errors += 1


_current_capture: ContextVar[Capture | None] = ContextVar(
    "provider_exchange_capture", default=None
)


def current_capture() -> Capture | None:
    return _current_capture.get()


@contextmanager
def capture(*, enabled: bool) -> Generator[Capture | None, None, None]:
    if not enabled:
        yield None
        return
    try:
        spool = tempfile.SpooledTemporaryFile(
            max_size=SPOOL_MEMORY_LIMIT_BYTES, mode="w+b"
        )
    except OSError:
        yield None
        return

    active = Capture(spool)
    token = _current_capture.set(active)
    try:
        yield active
    finally:
        _current_capture.reset(token)
        active._cleanup()

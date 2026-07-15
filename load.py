import asyncio
import json
import logging
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeGuard, cast

import aiohttp
import httpx
from dotenv import load_dotenv
from model_library.base import LLMConfig, TextInput
from model_library.registry_utils import get_registry_model

load_dotenv(override=False)

SERVICE_URL = (
    os.environ.get("GATEWAY_LOAD_URL")
    or os.environ.get("MODEL_GATEWAY_URL")
    or "http://127.0.0.1:8000"
).rstrip("/")
MODEL = os.environ.get("LOAD_MODEL", "openai/gpt-4o-mini")
PROMPT = os.environ.get(
    "LOAD_PROMPT",
    "Ignore the preceding filler text. Write a two-page story about cats and dogs "
    "with an ending that teaches a life lesson.",
)
FILLER_TOKENS = int(os.environ.get("LOAD_FILLER_TOKENS", "1000"))
MAX_TOKENS = int(os.environ.get("LOAD_MAX_TOKENS", "1500"))
TEMPERATURE = float(os.environ.get("LOAD_TEMPERATURE", "0"))

NUM_WORKERS = int(os.environ.get("LOAD_WORKERS", "2"))
REQUESTS_PER_WORKER_PER_SEC = int(os.environ.get("LOAD_RPS_PER_WORKER", "3"))
DURATION = int(os.environ.get("LOAD_DURATION", "250"))
PREWARM_CONNECTIONS_PER_WORKER = int(
    os.environ.get("LOAD_PREWARM_CONNECTIONS_PER_WORKER", "0")
)
CONNECT_TIMEOUT = float(os.environ.get("LOAD_CONNECT_TIMEOUT", "30"))
READ_TIMEOUT = float(os.environ.get("LOAD_READ_TIMEOUT", "3660"))
WRITE_TIMEOUT = float(os.environ.get("LOAD_WRITE_TIMEOUT", "60"))
POOL_TIMEOUT = float(os.environ.get("LOAD_POOL_TIMEOUT", "60"))
REQUEST_TIMEOUT = float(os.environ.get("LOAD_REQUEST_TIMEOUT", str(READ_TIMEOUT)))
PREWARM_TIMEOUT = float(os.environ.get("LOAD_PREWARM_TIMEOUT", str(CONNECT_TIMEOUT)))
MAX_CONNECTIONS_PER_WORKER = int(
    os.environ.get("LOAD_MAX_CONNECTIONS_PER_WORKER", "5000")
)
MAX_KEEPALIVE_CONNECTIONS_PER_WORKER = int(
    os.environ.get("LOAD_MAX_KEEPALIVE_CONNECTIONS_PER_WORKER", "1000")
)
LOAD_RUN_ID = os.environ.get("LOAD_RUN_ID") or f"load-{int(time.time())}"
_load_identity_email = "load-test@example.com"
LOAD_IDENTITY = {
    "email": _load_identity_email,
    "benchmark_name": "load-test",
    "agent_name": "load.py",
}
QUERY_CLIENT = os.environ.get("LOAD_QUERY_CLIENT", "model_library").strip().lower()
HTTP_CLIENT = os.environ.get("LOAD_HTTP_CLIENT", "httpx").strip().lower()
MODEL_LIBRARY_LOG_LEVEL = os.environ.get("LOAD_MODEL_LIBRARY_LOG_LEVEL", "WARNING")
RAW_HTTP_DEBUG_FLAG = "LOAD_ALLOW_RAW_HTTP"
REQUEST_LOG_JSONL = os.environ.get("LOAD_REQUEST_LOG_JSONL") or os.path.join(
    ".scratch", "runs", f"{LOAD_RUN_ID}-requests.jsonl"
)
STALL_TIMEOUT_SECONDS = float(os.environ.get("LOAD_STALL_TIMEOUT_SECONDS", "300"))
PENDING_SNAPSHOT_LIMIT = int(os.environ.get("LOAD_PENDING_SNAPSHOT_LIMIT", "50"))

# One /query request normally maps to one provider model request.
MODEL_REQUESTS_PER_ENDPOINT_REQUEST = 1

Operation = Literal["query", "upload", "embeddings", "moderation"]


class GatewayQueryClient(Protocol):
    async def query(self, input: object, **kwargs: object) -> object: ...


LoadClient = httpx.AsyncClient | aiohttp.ClientSession | GatewayQueryClient


def gateway_api_key() -> str:
    return os.environ.get("MODEL_GATEWAY_" + "API_KEY", "")


def validate_query_client_config(query_client: str, *, warn: bool = True) -> None:
    if query_client == "model_library":
        return
    if query_client != "raw":
        raise ValueError("LOAD_QUERY_CLIENT must be 'model_library' or 'raw'")
    if os.environ.get(RAW_HTTP_DEBUG_FLAG) != "1":
        raise ValueError(
            "LOAD_QUERY_CLIENT=raw is disabled by default. "
            f"Use {RAW_HTTP_DEBUG_FLAG}=1 only for hard debugging."
        )
    if warn:
        print(
            "WARNING: LOAD_QUERY_CLIENT=raw bypasses the model_library gateway "
            "client and is for hard debugging only.",
            file=sys.stderr,
        )


@dataclass(frozen=True)
class LoadConfig:
    url: str
    api_key: str
    model: str
    embedding_model: str
    operations: tuple[Operation, ...]
    duration: int
    total: int | None
    workers: int
    rps_per_worker: int
    concurrency: int
    timeout: int
    prompt: str
    max_tokens: int
    temperature: float
    include_token_retry: bool


_ENDPOINTS: dict[Operation, str] = {
    "query": "/query",
    "upload": "/files/upload",
    "embeddings": "/embeddings",
    "moderation": "/moderation",
}


def operation_url(base_url: str, operation: Operation) -> str:
    return f"{base_url.rstrip('/')}{_ENDPOINTS[operation]}"


def build_body(
    config: LoadConfig, operation: Operation, request_id: int
) -> dict[str, Any]:
    if operation == "query":
        body: dict[str, Any] = {
            "model": config.model,
            "inputs": [
                {
                    "kind": "text",
                    "text": f"Request id: {request_id}\n\n{config.prompt}",
                }
            ],
            "config": {
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
            },
        }
        if config.include_token_retry:
            body["token_retry_params"] = {}
        return body
    if operation == "embeddings":
        return {
            "model": config.model,
            "embedding_model": config.embedding_model,
            "text": config.prompt,
            "config": {},
        }
    if operation == "moderation":
        return {"model": config.model, "text": config.prompt, "config": {}}
    raise ValueError(
        "upload load bodies require file content and are not auto-generated"
    )


def build_prompt(request_id: int) -> str:
    rng = random.Random(f"{request_id}-{time.time_ns()}")
    filler_words = [
        "the",
        "river",
        "lantern",
        "copper",
        "meadow",
        "orbit",
        "velvet",
        "window",
        "maple",
        "harbor",
        "silver",
        "cloud",
    ]
    filler = " ".join(
        f"the-{rng.choice(filler_words)}-{rng.randrange(1_000_000):06d}"
        for _ in range(FILLER_TOKENS)
    )
    return f"{filler}\n\nRequest id: {request_id}\n\n{PROMPT}"


def build_query_body(request_id: int, *, worker_id: int) -> dict[str, Any]:
    return {
        "model": MODEL,
        "inputs": [
            {
                "kind": "text",
                "text": build_prompt(request_id),
            }
        ],
        "config": {
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
        "run_id": LOAD_RUN_ID,
        "question_id": f"worker-{worker_id}",
        "query_id": f"load-{request_id}",
        "identity": LOAD_IDENTITY,
    }


def extract_tokens(body: dict[str, Any]) -> tuple[int, int]:
    metadata_raw: object = body.get("metadata") or {}
    if not isinstance(metadata_raw, dict):
        return 0, 0
    metadata = cast(dict[str, Any], metadata_raw)
    in_tokens = metadata.get("in_tokens", 0)
    out_tokens = metadata.get("out_tokens", 0)
    return (
        in_tokens if isinstance(in_tokens, int) else 0,
        out_tokens if isinstance(out_tokens, int) else 0,
    )


def _query_result_tokens(result: object) -> tuple[int, int]:
    metadata = getattr(result, "metadata", None)
    in_tokens = getattr(metadata, "in_tokens", 0)
    out_tokens = getattr(metadata, "out_tokens", 0)
    return (
        in_tokens if isinstance(in_tokens, int) else 0,
        out_tokens if isinstance(out_tokens, int) else 0,
    )


def _is_gateway_query_client(client: object) -> TypeGuard[GatewayQueryClient]:
    return callable(getattr(client, "query", None))


def _configure_model_library_logging() -> None:
    level = getattr(logging, MODEL_LIBRARY_LOG_LEVEL.upper(), logging.WARNING)
    logging.getLogger("llm").setLevel(level)
    for handler in logging.getLogger("llm").handlers:
        handler.setLevel(level)


def _build_gateway_query_client() -> GatewayQueryClient:
    from model_library import model_library_settings

    _configure_model_library_logging()
    settings = {"MODEL_GATEWAY_URL": SERVICE_URL}
    api_key = gateway_api_key()
    if api_key:
        settings["MODEL_GATEWAY_API_KEY"] = api_key
    model_library_settings.set(**settings)
    return cast(
        GatewayQueryClient,
        get_registry_model(
            MODEL,
            LLMConfig(max_tokens=MAX_TOKENS, temperature=TEMPERATURE),
        ),
    )


def _load_query_id(request_id: int) -> str:
    return f"load-{request_id}"


def _counter_lock(counters: dict[str, Any]) -> AbstractContextManager[object]:
    lock = counters.get("lock")
    if lock is None:
        return nullcontext()
    return cast(AbstractContextManager[object], lock)


def _write_request_log_line(counters: dict[str, Any], record: dict[str, Any]) -> None:
    path = str(counters.get("request_log_jsonl") or REQUEST_LOG_JSONL)
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except OSError as exc:
        print(
            f"\n  [request-log] failed path={path} error_type={type(exc).__name__}",
            file=sys.stderr,
            flush=True,
        )


def _safe_request_log_error(exc: Exception) -> str:
    message = str(exc)
    if message.startswith("HTTP "):
        return message.split(":", 1)[0]
    return type(exc).__name__


def _record_request_event(
    counters: dict[str, Any],
    event: str,
    request_id: int,
    worker_id: int,
    **fields: object,
) -> None:
    load_query_id = _load_query_id(request_id)
    record: dict[str, Any] = {
        "event": event,
        "time": time.time(),
        "load_run_id": LOAD_RUN_ID,
        "worker": worker_id,
        "request_id": request_id,
        "load_query_id": load_query_id,
        **fields,
    }
    with _counter_lock(counters):
        pending = counters.get("pending")
        if pending is not None:
            if event in {"scheduled", "started"}:
                pending[load_query_id] = {
                    "event": event,
                    "time": record["time"],
                    "load_run_id": LOAD_RUN_ID,
                    "worker": worker_id,
                    "request_id": request_id,
                    "load_query_id": load_query_id,
                }
            elif event in {"succeeded", "failed", "cancelled"}:
                pending.pop(load_query_id, None)
        _write_request_log_line(counters, record)


def _pending_request_records(counters: dict[str, Any]) -> list[dict[str, Any]]:
    pending = counters.get("pending")
    if pending is None:
        return []
    return [dict(record) for _, record in sorted(pending.items())]


def _pending_request_count(counters: dict[str, Any]) -> int:
    pending = counters.get("pending")
    return len(pending) if pending is not None else 0


def _should_dump_pending_requests(
    counters: dict[str, Any],
    *,
    now: float,
    last_completion_time: float,
    last_stall_dump_time: float,
) -> bool:
    return (
        STALL_TIMEOUT_SECONDS > 0
        and _pending_request_count(counters) > 0
        and now - last_completion_time >= STALL_TIMEOUT_SECONDS
        and now - last_stall_dump_time >= STALL_TIMEOUT_SECONDS
    )


def _dump_pending_requests(
    counters: dict[str, Any],
    *,
    reason: str,
    stalled_seconds: float | None = None,
    limit: int = PENDING_SNAPSHOT_LIMIT,
) -> int:
    pending = _pending_request_records(counters)
    if not pending:
        return 0
    sample = pending[:limit]
    record: dict[str, Any] = {
        "event": "pending_snapshot",
        "time": time.time(),
        "load_run_id": LOAD_RUN_ID,
        "reason": reason,
        "pending_count": len(pending),
        "pending_sample": sample,
        "pending_sample_limit": limit,
    }
    if stalled_seconds is not None:
        record["stalled_seconds"] = stalled_seconds
    with _counter_lock(counters):
        _write_request_log_line(counters, record)
    query_ids = ", ".join(str(item["load_query_id"]) for item in sample)
    suffix = (
        "" if len(sample) == len(pending) else f" (+{len(pending) - len(sample)} more)"
    )
    print(
        f"\n  [pending] reason={reason} count={len(pending)} sample={query_ids}{suffix}",
        file=sys.stderr,
        flush=True,
    )
    return len(pending)


async def _close_gateway_query_client(client: GatewayQueryClient) -> None:
    get_client = getattr(client, "get_client", None)
    if not callable(get_client):
        return
    raw_client = cast(httpx.AsyncClient, get_client())
    await raw_client.aclose()


def _record_successful_request(
    counters: dict[str, Any],
    *,
    request_id: int,
    worker_id: int,
    load_query_id: str,
    t_created: float,
    t_scheduled: float,
    t_done: float,
    in_tokens: int,
    out_tokens: int,
) -> dict[str, Any]:
    with counters["lock"]:
        counters["succeeded"].value += 1
        counters["in_tokens"].value += in_tokens
        counters["out_tokens"].value += out_tokens
        counters["completed"].value += 1
    _record_request_event(
        counters,
        "succeeded",
        request_id,
        worker_id,
        elapsed=t_done - t_created,
        schedule_wait=t_scheduled - t_created,
        query_time=t_done - t_scheduled,
        in_tokens=in_tokens,
        out_tokens=out_tokens,
    )
    return {
        "ok": True,
        "elapsed": t_done - t_created,
        "schedule_wait": t_scheduled - t_created,
        "query_time": t_done - t_scheduled,
        "worker": worker_id,
        "request_id": request_id,
        "load_run_id": LOAD_RUN_ID,
        "load_query_id": load_query_id,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
    }


async def _fire_one(
    client: LoadClient,
    request_id: int,
    worker_id: int,
    counters: dict[str, Any],
) -> dict[str, Any]:
    t_created = time.monotonic()
    load_query_id = _load_query_id(request_id)
    await asyncio.sleep(0)
    t_scheduled = time.monotonic()
    with counters["lock"]:
        counters["dispatched"].value += 1
    _record_request_event(
        counters,
        "started",
        request_id,
        worker_id,
        schedule_wait=t_scheduled - t_created,
    )
    try:
        if _is_gateway_query_client(client):
            result = await asyncio.wait_for(
                client.query(
                    [TextInput(text=build_prompt(request_id))],
                    run_id=LOAD_RUN_ID,
                    question_id=f"worker-{worker_id}",
                    query_id=load_query_id,
                    identity=LOAD_IDENTITY,
                ),
                timeout=REQUEST_TIMEOUT,
            )
            t_done = time.monotonic()
            in_tokens, out_tokens = _query_result_tokens(result)
            return _record_successful_request(
                counters,
                request_id=request_id,
                worker_id=worker_id,
                load_query_id=load_query_id,
                t_created=t_created,
                t_scheduled=t_scheduled,
                t_done=t_done,
                in_tokens=in_tokens,
                out_tokens=out_tokens,
            )
        if isinstance(client, httpx.AsyncClient):
            response = await client.post(
                f"{SERVICE_URL}/query",
                json=build_query_body(request_id, worker_id=worker_id),
            )
            status_code = response.status_code
            response_text = response.text
            body = response.json() if status_code == 200 else None
        elif isinstance(client, aiohttp.ClientSession):
            async with client.post(
                f"{SERVICE_URL}/query",
                json=build_query_body(request_id, worker_id=worker_id),
            ) as response:
                status_code = response.status
                response_text = await response.text()
                body = await response.json() if status_code == 200 else None
        else:
            raise TypeError(f"Unsupported load client: {type(client).__name__}")
        t_done = time.monotonic()
        if status_code == 200 and body is not None:
            in_tokens, out_tokens = extract_tokens(body)
            return _record_successful_request(
                counters,
                request_id=request_id,
                worker_id=worker_id,
                load_query_id=load_query_id,
                t_created=t_created,
                t_scheduled=t_scheduled,
                t_done=t_done,
                in_tokens=in_tokens,
                out_tokens=out_tokens,
            )
        raise Exception(f"HTTP {status_code}: {response_text[:500]}")
    except asyncio.CancelledError:
        t_done = time.monotonic()
        with counters["lock"]:
            counters["failed"].value += 1
            counters["completed"].value += 1
        _record_request_event(
            counters,
            "cancelled",
            request_id,
            worker_id,
            elapsed=t_done - t_created,
            schedule_wait=t_scheduled - t_created,
            query_time=t_done - t_scheduled,
        )
        raise
    except Exception as exc:
        t_done = time.monotonic()
        err_msg = f"{type(exc).__name__}: {str(exc)[:500]}"
        log_err_msg = f"{type(exc).__name__}: {_safe_request_log_error(exc)}"
        print(
            f"\n  [worker {worker_id}] request_id={request_id} "
            f"load_query_id={load_query_id} {log_err_msg}",
            file=sys.stderr,
            flush=True,
        )
        with counters["lock"]:
            counters["failed"].value += 1
            counters["completed"].value += 1
        _record_request_event(
            counters,
            "failed",
            request_id,
            worker_id,
            error=_safe_request_log_error(exc),
            error_type=type(exc).__name__,
            elapsed=t_done - t_created,
            schedule_wait=t_scheduled - t_created,
            query_time=t_done - t_scheduled,
        )
        return {
            "ok": False,
            "error": err_msg,
            "log_error": log_err_msg,
            "traceback": traceback.format_exc(),
            "log_traceback": log_err_msg,
            "elapsed": t_done - t_created,
            "schedule_wait": t_scheduled - t_created,
            "query_time": t_done - t_scheduled,
            "worker": worker_id,
            "request_id": request_id,
            "load_run_id": LOAD_RUN_ID,
            "load_query_id": load_query_id,
        }


async def _prewarm_connections(client: LoadClient, count: int) -> tuple[int, int]:
    async def hit_health() -> None:
        if _is_gateway_query_client(client):
            get_client = getattr(client, "get_client", None)
            if not callable(get_client):
                return
            raw_client = get_client()
            if not isinstance(raw_client, httpx.AsyncClient):
                return
            response = await raw_client.get(f"{SERVICE_URL}/health/live")
            response.raise_for_status()
            return
        if isinstance(client, httpx.AsyncClient):
            response = await client.get(f"{SERVICE_URL}/health/live")
            response.raise_for_status()
            return
        if isinstance(client, aiohttp.ClientSession):
            async with client.get(f"{SERVICE_URL}/health/live") as response:
                response.raise_for_status()
            return
        raise TypeError(f"Unsupported load client: {type(client).__name__}")

    results = await asyncio.gather(
        *(
            asyncio.wait_for(hit_health(), timeout=PREWARM_TIMEOUT)
            for _ in range(count)
        ),
        return_exceptions=True,
    )
    failures: Counter[str] = Counter()
    for result in results:
        if isinstance(result, Exception):
            failures[f"{type(result).__name__}: {str(result)[:200]}"] += 1
    if failures:
        print(
            f"\n  [prewarm] {sum(failures.values())}/{count} health requests failed",
            file=sys.stderr,
            flush=True,
        )
        for error, error_count in failures.most_common(5):
            print(
                f"  [prewarm] {error_count}x {error}",
                file=sys.stderr,
                flush=True,
            )
    return count - sum(failures.values()), sum(failures.values())


async def worker_loop(
    worker_id: int,
    start_time: float,
    counters: dict[str, Any],
) -> list[dict[str, Any]]:
    validate_query_client_config(QUERY_CLIENT, warn=False)
    headers = {"Content-Type": "application/json"}
    api_key = gateway_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request_id = worker_id * 1_000_000
    limits = httpx.Limits(
        max_connections=MAX_CONNECTIONS_PER_WORKER,
        max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS_PER_WORKER,
    )
    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT,
        read=READ_TIMEOUT,
        write=WRITE_TIMEOUT,
        pool=POOL_TIMEOUT,
    )

    if QUERY_CLIENT == "model_library":
        client = _build_gateway_query_client()
        try:
            return await _run_worker_requests(
                client, request_id, worker_id, start_time, counters
            )
        finally:
            await _close_gateway_query_client(client)
    if QUERY_CLIENT == "raw":
        if HTTP_CLIENT == "httpx":
            async with httpx.AsyncClient(
                headers=headers, timeout=timeout, limits=limits
            ) as client:
                return await _run_worker_requests(
                    client, request_id, worker_id, start_time, counters
                )
        if HTTP_CLIENT == "aiohttp":
            aiohttp_timeout = aiohttp.ClientTimeout(
                total=None,
                sock_connect=CONNECT_TIMEOUT,
                sock_read=READ_TIMEOUT,
            )
            connector = aiohttp.TCPConnector(
                limit=MAX_CONNECTIONS_PER_WORKER,
                limit_per_host=MAX_CONNECTIONS_PER_WORKER,
                keepalive_timeout=60,
            )
            async with aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp_timeout,
                connector=connector,
            ) as client:
                return await _run_worker_requests(
                    client, request_id, worker_id, start_time, counters
                )
        raise ValueError("LOAD_HTTP_CLIENT must be 'httpx' or 'aiohttp'")
    raise AssertionError("query client config should have been validated")


async def _run_worker_requests(
    client: LoadClient,
    request_id: int,
    worker_id: int,
    start_time: float,
    counters: dict[str, Any],
) -> list[dict[str, Any]]:
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    dispatch_start_time = time.monotonic()
    if PREWARM_CONNECTIONS_PER_WORKER:
        await _prewarm_connections(client, PREWARM_CONNECTIONS_PER_WORKER)
        dispatch_start_time = time.monotonic()
    while time.monotonic() - dispatch_start_time < DURATION:
        for _ in range(REQUESTS_PER_WORKER_PER_SEC):
            _record_request_event(counters, "scheduled", request_id, worker_id)
            tasks.append(
                asyncio.create_task(_fire_one(client, request_id, worker_id, counters))
            )
            request_id += 1
        await asyncio.sleep(1.0)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, dict)]


def run_worker(args: tuple[int, float, dict[str, Any]]) -> list[dict[str, Any]]:
    worker_id, start_time, counters = args
    return asyncio.run(worker_loop(worker_id, start_time, counters))


def main() -> None:
    validate_query_client_config(QUERY_CLIENT)
    dispatch_rate = NUM_WORKERS * REQUESTS_PER_WORKER_PER_SEC
    total_expected = dispatch_rate * DURATION

    print(f"Gateway: {SERVICE_URL}")
    print(f"Model: {MODEL}")
    print(
        f"Workers: {NUM_WORKERS} | Dispatch rate: {dispatch_rate} req/s | Duration: {DURATION}s"
    )
    if PREWARM_CONNECTIONS_PER_WORKER:
        print(f"Prewarm connections per worker: {PREWARM_CONNECTIONS_PER_WORKER}")
    print(f"Query client: {QUERY_CLIENT}")
    if QUERY_CLIENT == "model_library":
        print(
            "Gateway client: model_library GatewayLLM.query with gateway HTTP retries "
            f"(log_level={MODEL_LIBRARY_LOG_LEVEL.upper()})"
        )
    else:
        print(
            "Client limits: "
            f"http_client={HTTP_CLIENT} "
            f"max_connections_per_worker={MAX_CONNECTIONS_PER_WORKER} "
            f"max_keepalive_per_worker={MAX_KEEPALIVE_CONNECTIONS_PER_WORKER}"
        )
    print(
        "Timeouts: "
        f"connect={CONNECT_TIMEOUT}s read={READ_TIMEOUT}s "
        f"write={WRITE_TIMEOUT}s pool={POOL_TIMEOUT}s "
        f"request={REQUEST_TIMEOUT}s prewarm={PREWARM_TIMEOUT}s"
    )
    print(f"Expected total: ~{total_expected}")
    print(f"Request log JSONL: {REQUEST_LOG_JSONL}")
    if STALL_TIMEOUT_SECONDS > 0:
        print(f"Pending request stall dump: {STALL_TIMEOUT_SECONDS:.0f}s")
    if not gateway_api_key():
        print("WARNING: no gateway API key configured", file=sys.stderr)

    manager = mp.Manager()
    counters: dict[str, Any] = {
        "dispatched": manager.Value("i", 0),
        "completed": manager.Value("i", 0),
        "succeeded": manager.Value("i", 0),
        "failed": manager.Value("i", 0),
        "in_tokens": manager.Value("i", 0),
        "out_tokens": manager.Value("i", 0),
        "pending": manager.dict(),
        "request_log_jsonl": REQUEST_LOG_JSONL,
        "lock": manager.Lock(),
    }

    start_time = time.monotonic()

    from tqdm import tqdm

    dispatched_bar = tqdm(
        total=total_expected, unit="req", desc="Dispatched", position=0
    )
    completed_bar = tqdm(total=total_expected, unit="req", desc="Completed", position=1)

    pool = mp.Pool(NUM_WORKERS)
    worker_args = [
        (worker_id, start_time, counters) for worker_id in range(NUM_WORKERS)
    ]
    async_result = pool.map_async(run_worker, worker_args)

    prev_dispatched = 0
    prev_completed = 0
    last_completion_time = start_time
    last_stall_dump_time = 0.0
    last_minute_time = start_time
    last_minute_ok = 0
    minute_rpms: list[float] = []
    while not async_result.ready():
        dispatched = counters["dispatched"].value
        completed = counters["completed"].value
        ok = counters["succeeded"].value
        err = counters["failed"].value
        out_tokens = counters["out_tokens"].value
        in_flight = dispatched - completed

        if dispatched > prev_dispatched:
            dispatched_bar.update(dispatched - prev_dispatched)
            prev_dispatched = dispatched
        now = time.monotonic()
        if completed > prev_completed:
            wall = now - start_time
            rpm = ok / (wall / 60) if wall > 0 else 0
            model_rpm = rpm * MODEL_REQUESTS_PER_ENDPOINT_REQUEST
            completed_bar.set_postfix_str(
                f"ok={ok} err={err} in_flight={in_flight} rpm={rpm:.0f} "
                f"model_rpm={model_rpm:.0f} out_tok={out_tokens:,}"
            )
            completed_bar.update(completed - prev_completed)
            prev_completed = completed
            last_completion_time = now
            last_stall_dump_time = 0.0

        if _should_dump_pending_requests(
            counters,
            now=now,
            last_completion_time=last_completion_time,
            last_stall_dump_time=last_stall_dump_time,
        ):
            _dump_pending_requests(
                counters,
                reason="stall",
                stalled_seconds=now - last_completion_time,
            )
            last_stall_dump_time = now

        if now - last_minute_time >= 60:
            minute_ok = ok - last_minute_ok
            minute_model = minute_ok * MODEL_REQUESTS_PER_ENDPOINT_REQUEST
            minute_rpms.append(minute_ok)
            sys.stderr.write(
                f"[minute {len(minute_rpms)}] endpoint_rpm={minute_ok} "
                f"model_rpm={minute_model} in_flight={in_flight}\n"
            )
            sys.stderr.flush()
            last_minute_ok = ok
            last_minute_time = now

        time.sleep(0.2)

    all_results_nested = async_result.get()
    pool.close()
    pool.join()

    dispatched = counters["dispatched"].value
    completed = counters["completed"].value
    if dispatched > prev_dispatched:
        dispatched_bar.update(dispatched - prev_dispatched)
    if completed > prev_completed:
        completed_bar.update(completed - prev_completed)
    dispatched_bar.close()
    completed_bar.close()

    wall_elapsed = time.monotonic() - start_time

    all_results: list[dict[str, Any]] = []
    for worker_results in all_results_nested:
        all_results.extend(worker_results)

    ok_results = [result for result in all_results if result.get("ok")]
    err_results = [result for result in all_results if not result.get("ok")]
    total_in = sum(result.get("in_tokens", 0) for result in ok_results)
    total_out = sum(result.get("out_tokens", 0) for result in ok_results)

    actual_rpm = len(ok_results) / (wall_elapsed / 60) if wall_elapsed > 0 else 0
    actual_model_rpm = actual_rpm * MODEL_REQUESTS_PER_ENDPOINT_REQUEST

    print(f"\n{'=' * 70}")
    print(
        f"Dispatched: {len(all_results)} | Succeeded: {len(ok_results)} | Failed: {len(err_results)}"
    )
    print(
        f"Wall time: {wall_elapsed:.1f}s | Endpoint RPM: {actual_rpm:.0f} | Model RPM: {actual_model_rpm:.0f}"
    )

    print("\nTokens:")
    print(f"  Input:  {total_in:,}")
    print(f"  Output: {total_out:,}")
    print(f"  Output tok/s: {total_out / max(wall_elapsed, 0.1):,.0f}")

    if minute_rpms:
        print("\nPer-minute breakdown:")
        for index, minute_rpm in enumerate(minute_rpms, 1):
            model_rpm = minute_rpm * MODEL_REQUESTS_PER_ENDPOINT_REQUEST
            print(
                f"  Minute {index}: endpoint_rpm={minute_rpm:.0f}  model_rpm={model_rpm:.0f}"
            )

    if err_results:
        error_counts: Counter[str] = Counter()
        for result in err_results:
            error_counts[result.get("log_error", result.get("error", "unknown"))] += 1
        print(f"\nError summary ({len(err_results)} total):")
        for error, count in error_counts.most_common(10):
            print(f"  {count:>5}x  {error}")
        print("\nFirst 5 tracebacks:")
        for result in err_results[:5]:
            print(
                f"\n[worker {result.get('worker')}] "
                f"request_id={result.get('request_id')} "
                f"load_query_id={result.get('load_query_id')} "
                f"{result.get('log_error', result.get('error'))}\n"
                f"{result.get('log_traceback', result.get('traceback', ''))}"
            )

    if ok_results:
        schedule_waits = sorted(result["schedule_wait"] for result in ok_results)
        query_times = sorted(result["query_time"] for result in ok_results)
        sample_count = len(ok_results)
        print(f"\nTiming breakdown ({sample_count} samples):")
        print("  Schedule wait:")
        print(
            f"    min={schedule_waits[0]:.4f}s  p50={schedule_waits[sample_count // 2]:.4f}s  "
            f"p95={schedule_waits[int(sample_count * 0.95)]:.4f}s  max={schedule_waits[-1]:.4f}s"
        )
        print("  Query time:")
        print(
            f"    min={query_times[0]:.1f}s  p50={query_times[sample_count // 2]:.1f}s  "
            f"p95={query_times[int(sample_count * 0.95)]:.1f}s  max={query_times[-1]:.1f}s"
        )

        by_worker: dict[int, list[float]] = defaultdict(list)
        for result in ok_results:
            by_worker[result["worker"]].append(result["query_time"])
        print("\n  Per-worker avg query time:")
        for worker_id in sorted(by_worker):
            times = by_worker[worker_id]
            avg = sum(times) / len(times)
            print(f"    worker {worker_id}: {len(times)} samples, avg={avg:.1f}s")

    if err_results:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

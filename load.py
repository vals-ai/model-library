import asyncio
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

SERVICE_URL = (
    os.environ.get("GATEWAY_LOAD_URL")
    or os.environ.get("MODEL_GATEWAY_URL")
    or "http://127.0.0.1:8000"
).rstrip("/")
API_KEY = os.environ.get("MODEL_GATEWAY_API_KEY", "")
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

# One /query request normally maps to one provider model request.
MODEL_REQUESTS_PER_ENDPOINT_REQUEST = 1

Operation = Literal["query", "upload", "embeddings", "moderation"]


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


def build_query_body(request_id: int) -> dict[str, Any]:
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


async def _fire_one(
    client: httpx.AsyncClient,
    request_id: int,
    worker_id: int,
    counters: dict[str, Any],
) -> dict[str, Any]:
    t_created = time.monotonic()
    await asyncio.sleep(0)
    t_scheduled = time.monotonic()
    with counters["lock"]:
        counters["dispatched"].value += 1
    try:
        response = await client.post(
            f"{SERVICE_URL}/query",
            json=build_query_body(request_id),
        )
        t_done = time.monotonic()
        if response.status_code == 200:
            body = response.json()
            in_tokens, out_tokens = extract_tokens(body)
            with counters["lock"]:
                counters["succeeded"].value += 1
                counters["in_tokens"].value += in_tokens
                counters["out_tokens"].value += out_tokens
                counters["completed"].value += 1
            return {
                "ok": True,
                "elapsed": t_done - t_created,
                "schedule_wait": t_scheduled - t_created,
                "query_time": t_done - t_scheduled,
                "worker": worker_id,
                "in_tokens": in_tokens,
                "out_tokens": out_tokens,
            }
        raise Exception(f"HTTP {response.status_code}: {response.text[:500]}")
    except Exception as exc:
        t_done = time.monotonic()
        err_msg = f"{type(exc).__name__}: {str(exc)[:500]}"
        print(f"\n  [worker {worker_id}] {err_msg}", file=sys.stderr, flush=True)
        with counters["lock"]:
            counters["failed"].value += 1
            counters["completed"].value += 1
        return {
            "ok": False,
            "error": err_msg,
            "traceback": traceback.format_exc(),
            "elapsed": t_done - t_created,
            "schedule_wait": t_scheduled - t_created,
            "query_time": t_done - t_scheduled,
            "worker": worker_id,
        }


async def worker_loop(
    worker_id: int,
    start_time: float,
    counters: dict[str, Any],
) -> list[dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    request_id = worker_id * 1_000_000
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    limits = httpx.Limits(max_connections=5000, max_keepalive_connections=1000)

    async with httpx.AsyncClient(headers=headers, timeout=600, limits=limits) as client:
        while time.monotonic() - start_time < DURATION:
            for _ in range(REQUESTS_PER_WORKER_PER_SEC):
                tasks.append(
                    asyncio.create_task(
                        _fire_one(client, request_id, worker_id, counters)
                    )
                )
                request_id += 1
            await asyncio.sleep(1.0)

        results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, dict)]


def run_worker(args: tuple[int, float, dict[str, Any]]) -> list[dict[str, Any]]:
    worker_id, start_time, counters = args
    return asyncio.run(worker_loop(worker_id, start_time, counters))


def main() -> None:
    dispatch_rate = NUM_WORKERS * REQUESTS_PER_WORKER_PER_SEC
    total_expected = dispatch_rate * DURATION

    print(f"Gateway: {SERVICE_URL}")
    print(f"Model: {MODEL}")
    print(
        f"Workers: {NUM_WORKERS} | Dispatch rate: {dispatch_rate} req/s | Duration: {DURATION}s"
    )
    print(f"Expected total: ~{total_expected}")
    if not API_KEY:
        print("WARNING: no gateway API key configured", file=sys.stderr)

    manager = mp.Manager()
    counters: dict[str, Any] = {
        "dispatched": manager.Value("i", 0),
        "completed": manager.Value("i", 0),
        "succeeded": manager.Value("i", 0),
        "failed": manager.Value("i", 0),
        "in_tokens": manager.Value("i", 0),
        "out_tokens": manager.Value("i", 0),
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
        if completed > prev_completed:
            wall = time.monotonic() - start_time
            rpm = ok / (wall / 60) if wall > 0 else 0
            model_rpm = rpm * MODEL_REQUESTS_PER_ENDPOINT_REQUEST
            completed_bar.set_postfix_str(
                f"ok={ok} err={err} in_flight={in_flight} rpm={rpm:.0f} "
                f"model_rpm={model_rpm:.0f} out_tok={out_tokens:,}"
            )
            completed_bar.update(completed - prev_completed)
            prev_completed = completed

        now = time.monotonic()
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
            error_counts[result.get("error", "unknown")] += 1
        print(f"\nError summary ({len(err_results)} total):")
        for error, count in error_counts.most_common(10):
            print(f"  {count:>5}x  {error}")
        print("\nFirst 5 tracebacks:")
        for result in err_results[:5]:
            print(
                f"\n[worker {result.get('worker')}] {result.get('error')}\n{result.get('traceback', '')}"
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


if __name__ == "__main__":
    main()

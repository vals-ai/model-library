"""Startup canary helpers for the gateway app."""

import asyncio
import logging
import os
import time
import uuid
from math import ceil
from typing import cast

import httpx
from fastapi import FastAPI

import model_library.telemetry as telemetry
from model_library.retriers.token import utils as token_utils

logger = logging.getLogger("model_proxy_server")

STARTUP_CANARY_MODEL = "openai/gpt-5.4-nano-2026-03-17"
STARTUP_CANARY_MAX_TOKENS = 32
STARTUP_CANARY_REASONING_EFFORT = "low"
STARTUP_CANARY_TIMEOUT_SECONDS = 30
STARTUP_CANARY_WAIT_TIMEOUT_SECONDS = 30
STARTUP_CANARY_RUN_PREFIX = "gateway-startup-canary-"


def startup_canary_key(run_id: str) -> str:
    return f"{token_utils.KEY_PREFIX}:gateway:startup_canary:{run_id}"


def startup_canary_state(enabled: bool) -> dict[str, str | bool]:
    return {
        "enabled": enabled,
        "status": "pending" if enabled else "disabled",
        "error": "",
    }


async def wait_for_local_live(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            response = await client.get(f"{base_url}/health/live")
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(0.25)
    raise RuntimeError(
        f"startup canary local /health/live never became ready: {last_error}"
    )


async def execute_startup_canary(
    *,
    authority_ready: asyncio.Event,
    managed_root: str,
    control_url: str,
    base_url: str,
    run_id: str,
    model: str,
    max_tokens: int,
    timeout_seconds: float,
    wait_timeout_seconds: float,
    reasoning_effort: str = "low",
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    timeout = httpx.Timeout(timeout_seconds)
    try:
        await _execute_startup_canary(
            authority_ready=authority_ready,
            managed_root=managed_root,
            control_url=control_url,
            base_url=base_url,
            run_id=run_id,
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            timeout_seconds=timeout_seconds,
            wait_timeout_seconds=wait_timeout_seconds,
            transport=transport,
        )
    finally:
        await token_utils.redis_client.delete(startup_canary_key(run_id))


async def _execute_startup_canary(
    *,
    authority_ready: asyncio.Event,
    managed_root: str,
    control_url: str,
    base_url: str,
    run_id: str,
    model: str,
    max_tokens: int,
    reasoning_effort: str,
    timeout: httpx.Timeout,
    timeout_seconds: float,
    wait_timeout_seconds: float,
    transport: httpx.AsyncBaseTransport | None,
) -> None:
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        await wait_for_local_live(
            client,
            base_url=base_url,
            timeout_seconds=wait_timeout_seconds,
        )
        await authority_ready.wait()
        await token_utils.redis_client.set(
            startup_canary_key(run_id),
            "1",
            ex=ceil(wait_timeout_seconds + 3 * timeout_seconds + 1),
        )
        root_headers = {"Authorization": f"Bearer {managed_root}"}
        mint_response = await client.post(
            f"{control_url}/service-auth",
            headers=root_headers,
            json={
                "run_id": run_id,
                "task_id": f"startup-query-{uuid.uuid4().hex}",
                "attempt_id": f"startup-attempt-{uuid.uuid4().hex}",
                "allowed_models": [model],
                "allowed_operations": ["query"],
                "ttl_seconds": 120,
            },
        )
        mint_response.raise_for_status()
        child = mint_response.json()
        try:
            response = await client.post(
                f"{base_url}/query",
                headers={"Authorization": f"Bearer {child['token']}"},
                json={
                    "run_id": run_id,
                    "model": model,
                    "inputs": [{"kind": "text", "text": "Reply with exactly: ok"}],
                    "config": {
                        "max_tokens": max_tokens,
                        "temperature": 0,
                        "reasoning_effort": reasoning_effort,
                    },
                },
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"startup canary /query failed: HTTP {response.status_code}: {response.text[:500]}"
                )
            if not response.json().get("signed_history"):
                raise RuntimeError(
                    "startup canary /query did not return signed_history"
                )
        finally:
            revoke_response = await client.post(
                f"{control_url}/service-auth/revoke",
                headers=root_headers,
                json={"lease_id": child["lease_id"]},
            )
            revoke_response.raise_for_status()


async def run_startup_canary(app: FastAPI) -> None:
    state = cast(dict[str, str | bool], app.state.startup_canary)
    try:
        run_id = f"{STARTUP_CANARY_RUN_PREFIX}{os.environ['GATEWAY_RELEASE_ID']}-{uuid.uuid4().hex}"
        state["run_id"] = run_id
        port = os.environ.get("GATEWAY_PORT", "8000")
        await execute_startup_canary(
            authority_ready=app.state.managed_authority_ready,
            managed_root=os.environ["GATEWAY_STARTUP_CANARY_MANAGED_ROOT"],
            control_url=os.environ["GATEWAY_STARTUP_CANARY_CONTROL_URL"].rstrip("/"),
            run_id=run_id,
            base_url=f"http://127.0.0.1:{port}",
            model=STARTUP_CANARY_MODEL,
            max_tokens=STARTUP_CANARY_MAX_TOKENS,
            timeout_seconds=STARTUP_CANARY_TIMEOUT_SECONDS,
            reasoning_effort=STARTUP_CANARY_REASONING_EFFORT,
            wait_timeout_seconds=STARTUP_CANARY_WAIT_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Gateway startup canary failed")
        telemetry.record_exception(
            exc,
            {
                "gateway.operation": "startup_canary",
                "gateway.error.phase": "startup_canary",
                "gateway.error.code": "startup_canary_failed",
            },
        )
        return
    finally:
        state.pop("run_id", None)
    state["status"] = "passed"
    state["error"] = ""
    logger.info("Gateway startup canary passed")

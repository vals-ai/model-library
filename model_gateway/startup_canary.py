"""Startup canary helpers for the gateway app."""

import asyncio
import logging
import os
import time
import uuid
from typing import cast

import httpx
from fastapi import FastAPI

import model_library.telemetry as telemetry

logger = logging.getLogger("model_proxy_server")

STARTUP_CANARY_MODEL = "openai/gpt-5.4-nano-2026-03-17"
STARTUP_CANARY_MAX_TOKENS = 32
STARTUP_CANARY_REASONING_EFFORT = "low"
STARTUP_CANARY_TIMEOUT_SECONDS = 30
STARTUP_CANARY_WAIT_TIMEOUT_SECONDS = 30


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
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout_seconds: float,
    wait_timeout_seconds: float,
    reasoning_effort: str = "low",
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        await wait_for_local_live(
            client,
            base_url=base_url,
            timeout_seconds=wait_timeout_seconds,
        )
        response = await client.post(
            f"{base_url}/query",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "inputs": [{"kind": "text", "text": "Reply with exactly: ok"}],
                "config": {
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "reasoning_effort": reasoning_effort,
                },
                "run_id": "gateway-startup-canary",
                "question_id": f"startup-{uuid.uuid4().hex[:12]}",
            },
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"startup canary /query failed: HTTP {response.status_code}: {response.text[:500]}"
        )
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("startup canary /query did not return signed_history")
    data_dict = cast(dict[str, object], data)
    if not data_dict.get("signed_history"):
        raise RuntimeError("startup canary /query did not return signed_history")


async def run_startup_canary(app: FastAPI, api_key: str) -> None:
    state = cast(dict[str, str | bool], app.state.startup_canary)
    try:
        port = os.environ.get("GATEWAY_PORT", "8000")
        await execute_startup_canary(
            api_key=api_key,
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
    state["status"] = "passed"
    state["error"] = ""
    logger.info("Gateway startup canary passed")

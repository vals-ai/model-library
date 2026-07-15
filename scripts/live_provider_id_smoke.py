"""Live smoke-check provider response/request IDs.

Examples:
    uv run python scripts/live_provider_id_smoke.py --models openai/gpt-4o-mini
    uv run python scripts/live_provider_id_smoke.py --models openai/gpt-4o-mini,anthropic/claude-3-5-haiku-latest

The script uses configured provider keys or MODEL_GATEWAY_URL like normal
model-library calls. It prints JSON and never prints API keys.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections.abc import Sequence

from dotenv import load_dotenv

from model_library import model
from model_library.base import TextInput
from model_library.base.base import LLMConfig

DEFAULT_MODELS = "openai/gpt-4o-mini"


async def probe_model(model_key: str, *, query_prefix: str) -> dict[str, object]:
    query_id = f"{query_prefix}-{model_key.replace('/', '-')}-{int(time.time() * 1000)}"
    try:
        llm = model(model_key, override_config=LLMConfig(max_tokens=16))
        result = await llm.query(
            [TextInput(text="Reply with exactly: ok")],
            query_id=query_id,
            run_id=query_prefix,
        )
    except Exception as exc:
        return {
            "model": model_key,
            "query_id": query_id,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    extras = result.extras
    return {
        "model": model_key,
        "query_id": query_id,
        "ok": True,
        "gateway_mode": llm.gateway_mode,
        "response_id": extras.response_id,
        "provider_response_id": extras.provider_response_id,
        "provider_request_id": extras.provider_request_id,
        "has_provider_response_id": extras.provider_response_id is not None,
        "has_provider_request_id": extras.provider_request_id is not None,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=os.environ.get("PROVIDER_ID_LIVE_MODELS", DEFAULT_MODELS),
        help="Comma-separated model keys to probe.",
    )
    parser.add_argument(
        "--query-prefix",
        default=os.environ.get("PROVIDER_ID_LIVE_RUN_ID", "provider-id-live-smoke"),
    )
    return parser.parse_args(argv)


async def async_main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(".env")
    args = parse_args(argv)
    model_keys = [key.strip() for key in args.models.split(",") if key.strip()]
    results: list[dict[str, object]] = []
    for model_key in model_keys:
        results.append(await probe_model(model_key, query_prefix=args.query_prefix))
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    return 0 if all(result.get("ok") for result in results) else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())

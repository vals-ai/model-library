"""Exercise the gateway and OpenAI wire protocol with a local provider fixture."""

import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import httpx
import pytest
from fakeredis import aioredis

import model_gateway.app as gateway_app
from model_gateway.key_pools import PoolConfig
from model_library.base import GatewayLLM
from model_library.base.input import (
    ComputerCallOutput,
    ToolDefinition,
)
from model_library.registry_utils import get_registry_model
from tests.unit.model_gateway._support import (
    HEADERS,
    InMemoryAssignmentStore,
    _make_client,
)
from tests.unit.model_gateway.test_app_roles import _quiet_lifecycle

SCREENSHOT = "data:image/png;base64," + base64.b64encode(b"fixture-png").decode()
ACTIONS = [{"type": "screenshot"}, {"type": "type", "text": "private fixture text"}]
CHECKS = [{"id": "check-1", "code": "fixture", "message": "Check this screen"}]
TOOLS = [{"name": "computer", "body": {"type": "computer"}}]


@pytest.fixture(autouse=True)
def mock_all_get_client():
    """Use real provider clients; the fixture provider below answers them."""
    yield


@pytest.fixture
def gateway():
    requests = []
    ledger_events = []

    class Ledger:
        enabled = True

        async def write_event(self, event):
            ledger_events.append(event)

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            assert body["model"] == "gpt-6.1-sol"
            assert self.path == "/v1/responses"
            index = len(requests)
            replies = [
                i for i in body["input"] if i.get("type") == "computer_call_output"
            ]
            if not replies:
                assert body["tools"] == [{"type": "computer"}]
                output = [
                    {
                        "type": "reasoning",
                        "id": "rs-1",
                        "summary": [],
                        "encrypted_content": "fixture-encrypted-reasoning",
                    },
                    {
                        "type": "computer_call",
                        "id": "cu-1",
                        "call_id": "call-1",
                        "actions": ACTIONS,
                        "pending_safety_checks": CHECKS,
                        "status": "completed",
                    },
                ]
            else:
                assert replies == [
                    {
                        "type": "computer_call_output",
                        "call_id": "call-1",
                        "output": {
                            "type": "computer_screenshot",
                            "image_url": SCREENSHOT,
                        },
                        "acknowledged_safety_checks": CHECKS,
                    }
                ]
                output = [
                    {
                        "type": "message",
                        "id": f"msg-{index}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "ORCHID", "annotations": []}
                        ],
                    }
                ]
            response = {
                "id": f"resp-{index}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-6.1-sol",
                "output": output,
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 3,
                    "total_tokens": 14,
                    "input_tokens_details": {
                        "cached_tokens": 0,
                        "cache_write_tokens": 0,
                    },
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
            payload = (
                "data: "
                + json.dumps(
                    {
                        "type": "response.completed",
                        "response": response,
                        "sequence_number": 1,
                    }
                )
                + "\n\ndata: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("x-request-id", f"req-{index}")
            self.end_headers()
            self.wfile.write(payload.encode())

        def log_message(self, *_args):
            pass

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    try:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "fixture-only",
                "OPENAI_API_KEY_2": "fixture-secondary",
                "MODEL_GATEWAY_URL": "",
                "GATEWAY_OTEL_ENABLED": "false",
                "REDIS_URL": "redis://computer-gateway-test",
                "GATEWAY_STARTUP_CANARY_ENABLED": "false",
            },
        ):

            def provider_model(name, config):
                with patch(
                    "model_library.registry_utils._gateway_url", return_value=None
                ):
                    return get_registry_model(
                        name,
                        override_config=config.model_copy(
                            update={
                                "custom_endpoint": f"http://127.0.0.1:{provider.server_port}/v1",
                                "max_tokens": 8192,
                            }
                        ),
                    )

            pool = PoolConfig(
                pools={"openai/default": ("OPENAI_API_KEY",)}
            )
            with (
                patch(
                    "model_gateway.model_helpers.get_registry_model",
                    side_effect=provider_model,
                ),
                patch(
                    "model_gateway.model_helpers.current_pool_config", return_value=pool
                ),
                patch.object(
                    gateway_app.async_redis,
                    "from_url",
                    return_value=aioredis.FakeRedis(decode_responses=True),
                ),
                _quiet_lifecycle(),
            ):
                with _make_client(
                    runtime_role="query",
                    api_keys={"test": "sk-test"},
                ) as client:
                    client.app.state.key_assignments = InMemoryAssignmentStore()
                    client.app.state.usage_ledger = Ledger()
                    yield client, requests, ledger_events
    finally:
        provider.shutdown()
        provider.server_close()
        thread.join()


@pytest.mark.asyncio
async def test_native_computer_gateway_round_trip(gateway):
    client, requests, ledger_events = gateway

    class Settings:
        MODEL_GATEWAY_URL = "http://gateway.test"
        MODEL_GATEWAY_API_KEY = "sk-test"

        def get(self, name, default=None):
            return getattr(self, name, default)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=client.app),
        base_url="http://gateway.test",
        headers={**HEADERS, "Content-Type": "application/json"},
    ) as transport:
        with (
            patch("model_library.model_library_settings", Settings()),
            patch.dict("model_library.base.base.client_registry", {}, clear=True),
            patch(
                "model_library.base.gateway.gateway_httpx_client",
                return_value=transport,
            ),
        ):
            llm = GatewayLLM("gpt-6.1-sol", "openai")
            tools = [ToolDefinition(**TOOLS[0])]
            first = await llm.query("Request the screenshot.", tools=tools)
            assert first.computer_calls[0].actions == ACTIONS
            assert first.computer_calls[0].pending_safety_checks == CHECKS
            second = await llm.query(
                [
                    ComputerCallOutput(
                        call_id="call-1",
                        screenshot_url=SCREENSHOT,
                        acknowledged_safety_checks=CHECKS,
                    )
                ],
                tools=tools,
                history=first.history,
            )
            assert second.output_text == "ORCHID"
            # Signed history keeps the screenshot, so a later turn can still read it.
            [kept] = [i for i in second.history if isinstance(i, ComputerCallOutput)]
            assert kept.screenshot_url == SCREENSHOT
    # The replay carries the encrypted reasoning and the call; the provider stores nothing.
    replayed = {item.get("type"): item for item in requests[1]["input"]}
    assert replayed["reasoning"]["encrypted_content"] == "fixture-encrypted-reasoning"
    assert replayed["computer_call"]["call_id"] == "call-1"
    assert all(request["store"] is False for request in requests)
    assert "private fixture text" not in json.dumps(ledger_events, default=str)
    assert "Check this screen" not in json.dumps(ledger_events, default=str)

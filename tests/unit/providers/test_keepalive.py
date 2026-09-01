import json
from unittest.mock import MagicMock, patch

import pytest

from model_library.base.base import client_registry
from model_library.providers.amazon import AmazonModel
from model_library.providers.xai import XAIModel

# The unit-test fixture replaces provider get_client methods at runtime.
_AMAZON_GET_CLIENT = AmazonModel.get_client
_XAI_GET_CLIENT = XAIModel.get_client


@pytest.mark.parametrize(
    ("api_key", "expected_credentials"),
    [
        ("using-environment", {}),
        (
            json.dumps(
                {
                    "AWS_ACCESS_KEY_ID": "access-key",
                    "AWS_SECRET_ACCESS_KEY": "secret-key",
                    "AWS_SESSION_TOKEN": "session-token",
                    "AWS_DEFAULT_REGION": "us-east-1",
                }
            ),
            {
                "aws_access_key_id": "access-key",
                "aws_secret_access_key": "secret-key",
                "aws_session_token": "session-token",
                "region_name": "us-east-1",
            },
        ),
    ],
)
def test_amazon_client_enables_tcp_keepalive_without_changing_credentials(
    api_key: str, expected_credentials: dict[str, str]
):
    sdk_client = MagicMock()

    with (
        patch.dict(client_registry, {}, clear=True),
        patch(
            "model_library.providers.amazon.boto3.client", return_value=sdk_client
        ) as create_client,
    ):
        model = AmazonModel("anthropic.claude-3-5-haiku-2024-10-22-v2:0")
        client = _AMAZON_GET_CLIENT(model, api_key=api_key)

    assert client is sdk_client
    assert create_client.call_args.args == ("bedrock-runtime",)
    client_arguments = create_client.call_args.kwargs
    config = client_arguments["config"]
    assert {
        key: value for key, value in client_arguments.items() if key != "config"
    } == (expected_credentials)
    assert config.max_pool_connections == 1000
    assert config.tcp_keepalive is True


@pytest.mark.parametrize("base_url", [None, "custom.api.x.ai"])
def test_xai_client_passes_explicit_keepalive_options(base_url: str | None):
    sdk_client = MagicMock()

    with (
        patch.dict(client_registry, {}, clear=True),
        patch(
            "model_library.providers.xai.AsyncClient", return_value=sdk_client
        ) as create_client,
    ):
        model = XAIModel("grok-3-mini")
        client = _XAI_GET_CLIENT(model, api_key="xai-key", base_url=base_url)

    expected_arguments = {
        "api_key": "xai-key",
        "channel_options": [
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 10_000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ],
        "timeout": 60 * 60 * 24,
    }
    if base_url:
        expected_arguments["api_host"] = base_url

    assert client is sdk_client
    create_client.assert_called_once_with(**expected_arguments)

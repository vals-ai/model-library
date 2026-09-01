import json
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from model_library.base import LLMConfig
from model_library.base.base import client_registry
from model_library.providers.google.google import GoogleConfig, GoogleModel

_REAL_GET_CLIENT = GoogleModel.get_client
_VERTEX_CREDENTIALS = json.dumps(
    {
        "GCP_REGION": "europe-west1",
        "GCP_PROJECT_ID": "test-project",
        "GCP_CREDS": json.dumps(
            {
                "type": "service_account",
                "project_id": "test-project",
                "private_key_id": "test-key-id",
                "private_key": "test-private-key",
                "client_email": "test@example.com",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        ),
    }
)


@pytest.fixture(autouse=True)
def clear_client_registry() -> Iterator[None]:
    with patch.dict(client_registry, {}, clear=True):
        yield


def test_vertex_client_uses_configured_region() -> None:
    sdk_client = MagicMock()
    config = LLMConfig(
        custom_api_key=SecretStr(_VERTEX_CREDENTIALS),
        provider_config=GoogleConfig(use_vertex=True),
    )

    with (
        patch.object(GoogleModel, "get_client", _REAL_GET_CLIENT),
        patch(
            "model_library.providers.google.google.Client", return_value=sdk_client
        ) as create_client,
        patch(
            "model_library.providers.google.google.service_account.Credentials.from_service_account_info",
            return_value=MagicMock(),
        ),
    ):
        model = GoogleModel("gemini-2.5-flash", config=config)

        assert model.get_client() is sdk_client
        assert create_client.call_args.kwargs["location"] == "europe-west1"

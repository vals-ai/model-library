from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from model_gateway.auth import create_auth_middleware


def _client() -> TestClient:
    app = FastAPI()
    app.middleware("http")(
        create_auth_middleware(
            {
                "default": "key-default",
                "security-testing": "key-security",
            }
        )
    )

    @app.get("/protected")
    async def protected(request: Request) -> dict[str, str]:
        return {"api_key_name": request.state.gateway_api_key_name}

    return TestClient(app)


def test_auth_resolves_configured_api_key_name() -> None:
    client = _client()

    default_response = client.get(
        "/protected", headers={"Authorization": "Bearer key-default"}
    )
    security_response = client.get(
        "/protected", headers={"Authorization": "Bearer key-security"}
    )

    assert default_response.status_code == 200
    assert default_response.json() == {"api_key_name": "default"}
    assert security_response.status_code == 200
    assert security_response.json() == {"api_key_name": "security-testing"}


def test_auth_rejects_unknown_api_key() -> None:
    response = _client().get("/protected", headers={"Authorization": "Bearer unknown"})

    assert response.status_code == 401

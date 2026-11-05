"""Managed run-token lifecycle through the Gateway HTTP app with PostgreSQL and Redis."""

import json
import os
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest
from fastapi.openapi.utils import get_openapi
from redis.asyncio import Redis

import model_gateway.app as gateway_app
from model_gateway.managed_auth.authority_publisher import ManagedAuthorityPublisher
from model_gateway.managed_auth.credentials import create_managed_key_secret
from model_gateway.managed_auth.database import create_engine
from model_gateway.managed_auth.reconciliation import HumanReconciler
from model_gateway.managed_auth.store import ManagedAuthStore
from model_gateway.managed_auth.types import HumanIdentity, PersonalOwner, ProjectReference
from model_gateway.run_tokens import LegacyRunTokenClaims, mint_run_token, sha256_hex


@pytest.mark.asyncio
async def test_managed_root_child_rotation_scope_and_revocation(monkeypatch):
    engine = create_engine(os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"], 4)
    store = ManagedAuthStore(engine)
    owner = HumanIdentity("https://issuer.example", f"owner-{uuid4()}")
    request_id, key_id = uuid4(), uuid4()
    other_request_id, other_key_id = uuid4(), uuid4()
    project = ProjectReference(uuid4(), "organization", "development")
    root, verifier = create_managed_key_secret()
    other_root, other_verifier = create_managed_key_secret()
    next_root, next_verifier = create_managed_key_secret()
    redis = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    namespace = f"managed-tokens-{uuid4()}"
    owner_key = f"gateway:managed-auth:{namespace}:owner"
    active_key = f"gateway:managed-auth:{namespace}:active"
    owner_token = uuid4().hex
    monkeypatch.setenv("MODEL_GATEWAY_AUTH_NAMESPACE", namespace)
    monkeypatch.setenv("MODEL_GATEWAY_AUTH_POOL_SIZE", "4")
    monkeypatch.setenv("GATEWAY_RUNTIME_ROLE", "control")
    monkeypatch.setenv("GATEWAY_USAGE_LEDGER_MODE", "disabled")
    monkeypatch.setenv("DESCOPE_PROJECT_ID", "local")
    monkeypatch.setenv("DESCOPE_EXPECTED_ISSUER", owner.issuer)
    monkeypatch.setenv("DESCOPE_ORGANIZATION_ID", "organization")
    monkeypatch.setenv("DESCOPE_MACHINE_ACCESS_KEY_ID", f"machine-{uuid4()}")
    monkeypatch.setenv("DESCOPE_MACHINE_SERVICE_OWNER_ID", str(uuid4()))
    monkeypatch.setenv("DESCOPE_MANAGEMENT_CREDENTIAL", "local-only")

    class Settings:
        def get(self, name: str, default: str = "") -> str:
            return {
                "MODEL_GATEWAY_API_KEYS": json.dumps({"operations": "sk-local"}),
                "MODEL_GATEWAY_HMAC_SECRET": "local-only-secret",
            }.get(name, default)

        def unset(self, key: str) -> None:
            pass

    try:
        await redis.set(owner_key, owner_token, nx=True, ex=3600)
        await store.record_human(owner, True, owner)
        await store.create_request(
            request_id, "key", "work", owner,
            PersonalOwner(owner), None, project, None, None,
        )
        pending = await store.decide_request(request_id, owner, key_id, ("monthly", Decimal("10")), None)
        collected = await store.collect_key(key_id, pending.revision, verifier, owner)
        await store.create_request(
            other_request_id, "other key", "work", owner,
            PersonalOwner(owner), None, project, None, None,
        )
        other_pending = await store.decide_request(
            other_request_id, owner, other_key_id, None, None,
        )
        await store.collect_key(other_key_id, other_pending.revision, other_verifier, owner)
        with patch.object(gateway_app, "model_library_settings", Settings()):
            app = gateway_app.create_app()
            # Generate only this registered route: the full app's FileInput schema is broken.
            mint_route = next(
                route for route in app.routes
                if route.path == "/service-auth" and "POST" in route.methods
            )
            mint_schema = get_openapi(
                title=app.title, version=app.version, routes=[mint_route],
            )
            request_ref = mint_schema["paths"]["/service-auth"]["post"]["requestBody"][
                "content"
            ]["application/json"]["schema"]["$ref"]
            request_schema = mint_schema["components"]["schemas"][
                request_ref.rsplit("/", 1)[-1]
            ]
            assert "allowed_models" in request_schema["required"]
            models_schema = request_schema["properties"]["allowed_models"]
            assert models_schema["type"] == "array"
            async with app.router.lifespan_context(app):
                publisher = ManagedAuthorityPublisher(
                    engine, redis,
                    HumanReconciler(store, app.state.managed_auth_verifier, owner.issuer),
                    namespace=namespace,
                )
                assert await publisher.publish_once(owner_token)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://testserver"
                ) as client:
                    root_headers = {"Authorization": f"Bearer {root}"}
                    operational_headers = {"Authorization": "Bearer sk-local"}
                    body = {
                        "run_id": f"run-{uuid4()}", "task_id": "task", "attempt_id": "attempt",
                        "allowed_models": ["openai/gpt-4o"],
                        "allowed_operations": ["query"], "ttl_seconds": 300,
                    }
                    operational_mint = await client.post(
                        "/service-auth", headers=operational_headers, json=body,
                    )
                    assert operational_mint.status_code == 200, operational_mint.text
                    assert (await client.get(
                        "/registry",
                        headers={"Authorization": f"Bearer {operational_mint.json()['token']}"},
                    )).status_code == 200
                    assert (await client.get("/registry", headers=operational_headers)).status_code == 200
                    assert (await client.post("/query", headers=operational_headers)).status_code == 404
                    assert (await client.get("/registry", headers=root_headers)).status_code == 200
                    minted = await client.post("/service-auth", headers=root_headers, json=body)
                    assert minted.status_code == 200, minted.text
                    other_headers = {"Authorization": f"Bearer {other_root}"}
                    other_minted = await client.post("/service-auth", headers=other_headers, json=body)
                    assert other_minted.status_code == 200, other_minted.text
                    other_child_headers = {"Authorization": f"Bearer {other_minted.json()['token']}"}
                    assert (await client.get("/registry", headers=other_child_headers)).status_code == 200
                    assert (await client.post(
                        "/service-auth/revoke", headers=root_headers,
                        json={"lease_id": other_minted.json()["lease_id"]},
                    )).status_code == 403
                    assert (await client.get("/registry", headers=other_child_headers)).status_code == 200
                    without_models = body.copy()
                    del without_models["allowed_models"]
                    assert (await client.post(
                        "/service-auth", headers=root_headers, json=without_models,
                    )).status_code == 400
                    assert (await client.post(
                        "/service-auth", headers=root_headers,
                        json={**body, "allowed_models": None},
                    )).status_code == 400
                    empty_models = await client.post(
                        "/service-auth", headers=root_headers,
                        json={**body, "run_id": f"run-{uuid4()}", "allowed_models": []},
                    )
                    assert empty_models.status_code == 200, empty_models.text
                    assert (await client.post(
                        "/service-auth/revoke", headers=root_headers,
                        json={"lease_id": empty_models.json()["lease_id"]},
                    )).json() == {"revoked": 1}
                    omitted_attempt = {key: value for key, value in body.items() if key != "attempt_id"}
                    omitted_attempt["run_id"] = f"run-{uuid4()}"
                    legacy_mint = await client.post(
                        "/service-auth", headers=root_headers, json=omitted_attempt,
                    )
                    assert legacy_mint.status_code == 200, legacy_mint.text
                    assert (await client.post(
                        "/service-auth/revoke", headers=root_headers,
                        json={"lease_id": legacy_mint.json()["lease_id"]},
                    )).json() == {"revoked": 1}
                    child_headers = {"Authorization": f"Bearer {minted.json()['token']}"}
                    visible = await client.get("/registry", headers=child_headers)
                    assert visible.status_code == 200, visible.text
                    assert "openai/gpt-4o" in visible.json()["models"]
                    assert (await client.post("/service-auth", headers=child_headers, json=body)).status_code == 403
                    # This child's root digest is not in the configured allowlist.
                    old_token, _ = await mint_run_token(
                        LegacyRunTokenClaims(
                            run_id=body["run_id"],
                            task_id=body["task_id"],
                            attempt_id=body["attempt_id"],
                            allowed_models=body["allowed_models"],
                            allowed_operations=body["allowed_operations"],
                            minted_by="operations",
                            minted_with=sha256_hex(f"sk-retired-{uuid4()}"),
                        ),
                        300,
                    )
                    assert (await client.get(
                        "/registry", headers={"Authorization": f"Bearer {old_token}"},
                    )).status_code == 401

                    rotated = await store.rotate_key(key_id, collected.revision, next_verifier, owner)
                    assert await publisher.publish_once(owner_token)
                    assert (await client.post("/service-auth", headers=root_headers, json=body)).status_code == 401
                    assert (await client.get("/registry", headers=child_headers)).status_code == 200
                    new_root_headers = {"Authorization": f"Bearer {next_root}"}
                    fresh = await client.post(
                        "/service-auth", headers=new_root_headers,
                        json={**body, "run_id": f"run-{uuid4()}"},
                    )
                    assert fresh.status_code == 200, fresh.text
                    fresh_headers = {"Authorization": f"Bearer {fresh.json()['token']}"}
                    fresh_registry = await client.get("/registry", headers=fresh_headers)
                    assert fresh_registry.status_code == 200, fresh_registry.text
                    assert "openai/gpt-4o" in fresh_registry.json()["models"]
                    revoked_run = await client.post(
                        "/service-auth/revoke", headers=new_root_headers,
                        json={"run_id": body["run_id"]},
                    )
                    assert revoked_run.status_code == 200
                    assert revoked_run.json()["revoked"] == 1
                    assert (await client.get("/registry", headers=child_headers)).status_code == 401
                    assert (await client.get("/registry", headers=other_child_headers)).status_code == 200
                    await store.record_human(owner, False, owner)
                    assert await publisher.publish_once(owner_token)
                    assert (await client.post("/service-auth", headers=new_root_headers, json=body)).status_code == 401
                    await store.record_human(owner, True, owner)
                    assert await publisher.publish_once(owner_token)
                    await store.revoke_key(key_id, rotated.revision, owner)
                    assert await publisher.publish_once(owner_token)
                    assert (await client.post("/service-auth", headers=new_root_headers, json=body)).status_code == 401
    finally:
        await redis.delete(owner_key, active_key)
        await redis.aclose()
        await engine.dispose()

import base64
import time
from types import SimpleNamespace

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from descope import AuthException
from fastapi import FastAPI, HTTPException

from model_gateway.managed_auth.api import router
from model_gateway.managed_auth.identity import HumanVerifier, VerifiedHuman
from model_gateway.managed_auth.types import HumanIdentity


PROJECT = "project-test"
ISSUER = "https://api.descope.com/project-test"
AUDIENCE = PROJECT
ORGANIZATION = "organization-test"


def _b64(value: int) -> str:
    return base64.urlsafe_b64encode(value.to_bytes((value.bit_length() + 7) // 8, "big")).rstrip(b"=").decode()


@pytest.fixture
def signed_session():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key().public_numbers()
    jwks = {"keys": [{"kty": "RSA", "kid": "local-key", "alg": "RS256", "use": "sig", "n": _b64(public_key.n), "e": _b64(public_key.e)}]}

    def sign(subject="human-1", issuer=ISSUER, audience=AUDIENCE, permissions=None, **claims):
        payload = {
            "sub": subject,
            "iss": issuer,
            "exp": int(time.time()) + 3600,
            "permissions": ["platform-admin"] if permissions is None else permissions,
            **claims,
        }
        if audience is not None:
            payload["aud"] = audience
        return jwt.encode(payload, private_key, algorithm="RS256", headers={"kid": "local-key"})

    return sign, jwks


@pytest.fixture
async def verifier(signed_session):
    _, jwks = signed_session
    state = {
        "status": "enabled", "user_id": "human-1", "tenants": [{"tenantId": ORGANIZATION}],
        "permissions": ["platform-admin", "other-global"], "user_status": 200,
        "key_id": "bound-key-1", "key_status": "active", "bound_user_id": "human-1",
    }
    calls = []

    def respond(request):
        if request.url.path == "/v2/keys/project-test":
            return httpx.Response(200, json=jwks)
        if request.url.path == "/v1/mgmt/accesskey":
            calls.append(request)
            key = {"id": state["key_id"], "status": state["key_status"]}
            if state["bound_user_id"] is not None:
                key["boundUserId"] = state["bound_user_id"]
            return httpx.Response(200, json={"key": key})
        if request.url.path == "/v1/mgmt/user":
            calls.append(request)
            if state["user_status"] != 200:
                return httpx.Response(state["user_status"], json={"errorMessage": "unavailable"})
            return httpx.Response(200, json={"user": {
                "userId": state["user_id"],
                "status": state["status"],
                "userTenants": state["tenants"],
                "permissions": state["permissions"],
            }})
        raise AssertionError(f"Unexpected SDK request: {request.url}")

    verifier = HumanVerifier(
        project_id=PROJECT,
        expected_issuer=ISSUER,
        required_organization=ORGANIZATION,
        management_credential="management-test",
        registered_machine_key_id="registered-machine-key",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    verifier._client._auth_http._async_client = client
    verifier._client._mgmt_http._async_client = client
    verifier._client._license_attempted = True  # The SDK's unrelated license handshake has its own transport.
    yield verifier, state, calls
    await verifier.aclose()


@pytest.mark.integration
async def test_signed_human_resolves_current_user_and_global_permissions(verifier, signed_session):
    verifier, state, calls = verifier
    sign, _ = signed_session
    result = await verifier.verify(
        sign(permissions=["stale-admin"], tenants={ORGANIZATION: {"permissions": ["tenant-only"]}}),
        allow_bound_key=False,
    )
    assert result == VerifiedHuman(HumanIdentity(ISSUER, "human-1"), frozenset(state["permissions"]))
    assert len(calls) == 1
    assert calls[0].url.params["userId"] == "human-1"
    assert calls[0].headers["authorization"] == f"Bearer {PROJECT}:management-test"


@pytest.mark.integration
async def test_signed_session_without_audience_resolves_current_user(verifier, signed_session):
    verifier, state, calls = verifier
    result = await verifier.verify(signed_session[0](audience=None), allow_bound_key=False)
    assert result == VerifiedHuman(HumanIdentity(ISSUER, "human-1"), frozenset(state["permissions"]))
    assert len(calls) == 1


@pytest.mark.integration
async def test_bound_key_resolves_current_user_and_global_permissions(verifier, signed_session):
    verifier, state, calls = verifier
    result = await verifier.verify(
        signed_session[0](
            subject="bound-key-1", user_id="human-1", audience=None,
            permissions=["stale-admin"], tenants={ORGANIZATION: {"permissions": ["tenant-only"]}},
        ),
        allow_bound_key=True,
    )
    assert result == VerifiedHuman(HumanIdentity(ISSUER, "human-1"), frozenset(state["permissions"]))
    assert [request.url.path for request in calls] == ["/v1/mgmt/accesskey", "/v1/mgmt/user"]
    assert calls[0].url.params["id"] == "bound-key-1"
    assert calls[1].url.params["userId"] == "human-1"
    assert all(request.headers["authorization"] == f"Bearer {PROJECT}:management-test" for request in calls)


@pytest.mark.integration
async def test_bound_key_can_read_but_cannot_write_managed_auth_even_as_admin(
    verifier, signed_session,
):
    human_verifier, _, calls = verifier
    registered = []

    async def register(person):
        registered.append(person)

    async def list_requests():
        return []

    app = FastAPI()
    app.include_router(router)
    app.state.managed_auth_verifier = human_verifier
    app.state.managed_auth_service = SimpleNamespace(register=register)
    app.state.managed_auth_store = SimpleNamespace(list_requests=list_requests)
    token = signed_session[0](subject="bound-key-1", user_id="human-1")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway"
    ) as client:
        headers = {"Authorization": f"Bearer {token}"}
        read = await client.get("/admin/managed-auth/requests", headers=headers)
        write = await client.post(
            "/admin/managed-auth/keys/00000000-0000-0000-0000-000000000001/revoke",
            headers=headers,
            json={"expected_revision": 1},
        )

    assert read.status_code == 200 and read.json() == []
    assert write.status_code == 401
    assert len(registered) == 1
    assert registered[0].permissions == frozenset(["platform-admin", "other-global"])
    assert [request.url.path for request in calls] == ["/v1/mgmt/accesskey", "/v1/mgmt/user"]


@pytest.mark.integration
@pytest.mark.parametrize("field,value,claimed_user", [
    ("key_id", "another-key", "human-1"),
    ("key_status", "inactive", "human-1"),
    ("bound_user_id", "", "human-1"),
    ("bound_user_id", None, "human-1"),
    ("bound_user_id", "human-1", "another-human"),
])
async def test_key_binding_must_match_current_active_record(verifier, signed_session, field, value, claimed_user):
    verifier, state, calls = verifier
    state[field] = value
    with pytest.raises(HTTPException) as error:
        await verifier.verify(
            signed_session[0](subject="bound-key-1", user_id=claimed_user),
            allow_bound_key=True,
        )
    assert error.value.status_code == 401
    assert [request.url.path for request in calls] == ["/v1/mgmt/accesskey"]


@pytest.mark.integration
@pytest.mark.parametrize("status,tenants", [
    ("disabled", [{"tenantId": ORGANIZATION}]),
    ("enabled", [{"tenantId": "other-organization"}]),
])
async def test_bound_key_still_requires_enabled_user_and_membership(verifier, signed_session, status, tenants):
    verifier, state, calls = verifier
    state.update(status=status, tenants=tenants)
    with pytest.raises(HTTPException) as error:
        await verifier.verify(
            signed_session[0](subject="bound-key-1", user_id="human-1"),
            allow_bound_key=True,
        )
    assert error.value.status_code == 403
    assert [request.url.path for request in calls] == ["/v1/mgmt/accesskey", "/v1/mgmt/user"]
    assert calls[1].url.params["userId"] == "human-1"


@pytest.mark.integration
@pytest.mark.parametrize("subject,dsub", [
    ("registered-machine-key", "human-1"),
    ("bound-key-1", "registered-machine-key"),
])
async def test_registered_machine_key_cannot_be_a_human_even_with_user_claim(
    verifier, signed_session, subject, dsub,
):
    verifier, _, calls = verifier
    with pytest.raises(HTTPException) as error:
        await verifier.verify(
            signed_session[0](subject=subject, user_id="human-1", dsub=dsub),
            allow_bound_key=True,
        )
    assert error.value.status_code == 401
    assert calls == []


@pytest.mark.integration
async def test_claimed_email_does_not_replace_management_identity(verifier, signed_session):
    verifier, state, calls = verifier
    state["user_id"] = "another-human"
    with pytest.raises(HTTPException) as error:
        await verifier.verify(signed_session[0](email="claim@example.com"), allow_bound_key=False)
    assert error.value.status_code == 401
    assert calls[-1].url.params["userId"] == "human-1"


@pytest.mark.integration
@pytest.mark.parametrize(
    "claims",
    [{"issuer": "https://other-project"}, {"audience": "other-audience"}, {"subject": "access-key-1", "dsub": "human-1"}],
)
async def test_signed_claims_cannot_select_another_human(verifier, signed_session, claims):
    verifier, _, calls = verifier
    sign, _ = signed_session
    with pytest.raises(HTTPException) as error:
        await verifier.verify(sign(**claims), allow_bound_key=False)
    assert error.value.status_code == 401
    if claims.get("subject") == "access-key-1":
        assert calls[0].url.params["userId"] == "access-key-1"
    else:
        assert calls == []


@pytest.mark.integration
@pytest.mark.parametrize("status,tenants", [
    ("disabled", [{"tenantId": ORGANIZATION}]),
    ("enabled", [{"tenantId": "other-organization"}]),
])
async def test_current_status_and_membership_gate_humans(verifier, signed_session, status, tenants):
    verifier, state, _ = verifier
    state.update(status=status, tenants=tenants)
    with pytest.raises(HTTPException) as error:
        await verifier.verify(signed_session[0](), allow_bound_key=False)
    assert error.value.status_code == 403


@pytest.mark.integration
async def test_bad_signature_cannot_reach_management(verifier, signed_session):
    verifier, _, calls = verifier
    token = signed_session[0]()
    header, payload, signature = token.split(".")
    altered = header + "." + payload + "." + ("A" if signature[0] != "A" else "B") + signature[1:]
    with pytest.raises(HTTPException) as error:
        await verifier.verify(altered, allow_bound_key=False)
    assert error.value.status_code == 401
    assert calls == []


@pytest.mark.integration
async def test_expired_session_cannot_reach_management(verifier, signed_session):
    verifier, _, calls = verifier
    with pytest.raises(HTTPException) as error:
        await verifier.verify(
            signed_session[0](exp=int(time.time()) - 60), allow_bound_key=False
        )
    assert error.value.status_code == 401
    assert calls == []


@pytest.mark.integration
async def test_user_missing_is_not_inferred_from_normalized_jwt(verifier, signed_session):
    verifier, state, _ = verifier
    state["user_status"] = 404
    with pytest.raises(HTTPException) as error:
        await verifier.verify(signed_session[0](), allow_bound_key=False)
    assert error.value.status_code == 401


@pytest.mark.integration
async def test_user_without_global_grants_has_empty_permissions(verifier, signed_session):
    verifier, state, _ = verifier
    state["permissions"] = []
    result = await verifier.verify(
        signed_session[0](permissions=["stale-admin"]), allow_bound_key=False
    )
    assert result.permissions == frozenset()


@pytest.mark.integration
async def test_management_outage_is_not_invalid_credentials(verifier, signed_session):
    verifier, state, _ = verifier
    state["user_status"] = 503
    with pytest.raises(AuthException) as error:
        await verifier.verify(signed_session[0](), allow_bound_key=False)
    assert error.value.status_code == 503


@pytest.mark.integration
async def test_current_eligibility_uses_management_status_and_membership(verifier):
    verifier, state, calls = verifier
    identity = HumanIdentity(ISSUER, "human-1")
    assert await verifier.current_eligibility(identity)
    state["status"] = "disabled"
    assert not await verifier.current_eligibility(identity)
    state["status"] = "enabled"
    state["tenants"] = []
    assert not await verifier.current_eligibility(identity)
    state["tenants"] = [{"tenantId": ORGANIZATION}]
    state["user_status"] = 404
    assert not await verifier.current_eligibility(identity)
    state["user_status"] = 503
    with pytest.raises(AuthException):
        await verifier.current_eligibility(identity)
    assert all(request.url.params["userId"] == "human-1" for request in calls)

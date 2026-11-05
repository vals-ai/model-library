import base64
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from descope import AuthException
from fastapi import HTTPException

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
    state = {"status": "enabled", "user_id": "human-1", "tenants": [{"tenantId": ORGANIZATION}],
             "permissions": ["platform-admin", "other-global"], "user_status": 200}
    calls = []

    def respond(request):
        if request.url.path == "/v2/keys/project-test":
            return httpx.Response(200, json=jwks)
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
        sign(permissions=["stale-admin"], tenants={ORGANIZATION: {"permissions": ["tenant-only"]}})
    )
    assert result == VerifiedHuman(HumanIdentity(ISSUER, "human-1"), frozenset(state["permissions"]))
    assert len(calls) == 1
    assert calls[0].url.params["userId"] == "human-1"
    assert calls[0].headers["authorization"] == f"Bearer {PROJECT}:management-test"


@pytest.mark.integration
async def test_signed_session_without_audience_resolves_current_user(verifier, signed_session):
    verifier, state, calls = verifier
    result = await verifier.verify(signed_session[0](audience=None))
    assert result == VerifiedHuman(HumanIdentity(ISSUER, "human-1"), frozenset(state["permissions"]))
    assert len(calls) == 1


@pytest.mark.integration
async def test_claimed_email_does_not_replace_management_identity(verifier, signed_session):
    verifier, state, calls = verifier
    state["user_id"] = "another-human"
    with pytest.raises(HTTPException) as error:
        await verifier.verify(signed_session[0](email="claim@example.com"))
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
        await verifier.verify(sign(**claims))
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
        await verifier.verify(signed_session[0]())
    assert error.value.status_code == 403


@pytest.mark.integration
async def test_bad_signature_cannot_reach_management(verifier, signed_session):
    verifier, _, calls = verifier
    token = signed_session[0]()
    header, payload, signature = token.split(".")
    altered = header + "." + payload + "." + ("A" if signature[0] != "A" else "B") + signature[1:]
    with pytest.raises(HTTPException) as error:
        await verifier.verify(altered)
    assert error.value.status_code == 401
    assert calls == []


@pytest.mark.integration
async def test_expired_session_cannot_reach_management(verifier, signed_session):
    verifier, _, calls = verifier
    with pytest.raises(HTTPException) as error:
        await verifier.verify(signed_session[0](exp=int(time.time()) - 60))
    assert error.value.status_code == 401
    assert calls == []


@pytest.mark.integration
async def test_user_missing_is_not_inferred_from_normalized_jwt(verifier, signed_session):
    verifier, state, _ = verifier
    state["user_status"] = 404
    with pytest.raises(HTTPException) as error:
        await verifier.verify(signed_session[0]())
    assert error.value.status_code == 401


@pytest.mark.integration
async def test_user_without_global_grants_has_empty_permissions(verifier, signed_session):
    verifier, state, _ = verifier
    state["permissions"] = []
    result = await verifier.verify(signed_session[0](permissions=["stale-admin"]))
    assert result.permissions == frozenset()


@pytest.mark.integration
async def test_management_outage_is_not_invalid_credentials(verifier, signed_session):
    verifier, state, _ = verifier
    state["user_status"] = 503
    with pytest.raises(AuthException) as error:
        await verifier.verify(signed_session[0]())
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

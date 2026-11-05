import asyncio
import os
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from jwt.exceptions import PyJWTError
from sqlalchemy import select

from model_gateway.managed_auth.api import router
from model_gateway.managed_auth.credentials import managed_key_verifier
from model_gateway.managed_auth.database import create_engine
from model_gateway.managed_auth.identity import HumanVerifier, MachineVerifier
from model_gateway.managed_auth.service import ManagedAuthService
from model_gateway.managed_auth.store import ManagedAuthStore
from model_gateway.managed_auth.tables import audit
from model_gateway.managed_auth.types import HumanIdentity


@pytest.mark.asyncio
async def test_http_management_project_keys_limits_and_restricted_machine():
    engine = create_engine(os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"], 5)
    db = ManagedAuthStore(engine)
    issuer = "https://identity.example"
    organization = uuid4()
    tenant_id = f"descope-tenant-{uuid4()}"
    unrelated_tenant_id = f"unrelated-tenant-{uuid4()}"
    machine_key_id = f"registered-machine-{uuid4()}"
    service_id = uuid4()
    project_id, machine_project_id, other_project_id = uuid4(), uuid4(), uuid4()
    subjects = {name: f"{name}-{uuid4()}" for name in ("owner", "admin", "teammate", "outsider")}
    load_calls = []

    async def validate_session(token):
        if token == "machine-access-key":
            raise PyJWTError("Access key is not a human session")
        if token == "machine-jwt":
            return {"sessionToken": {"iss": issuer, "sub": machine_key_id,
                                     "dsub": machine_key_id}}
        if token == "machine-sub-jwt":
            return {"sessionToken": {"iss": issuer, "sub": machine_key_id}}
        if token == "machine-dsub-jwt":
            return {"sessionToken": {"iss": issuer, "sub": subjects["admin"],
                                     "dsub": machine_key_id}}
        return {"sessionToken": {"iss": issuer, "sub": subjects[token]}}

    async def load_user(user_id):
        load_calls.append(user_id)
        return {"user": {"userId": user_id, "status": "enabled",
                         "userTenants": [{"tenantId": unrelated_tenant_id},
                                         {"tenantId": tenant_id}],
                         "permissions": ["platform-admin"] if user_id == subjects["admin"] else []}}

    async def exchange_access_key(access_key):
        assert access_key == "machine-access-key"
        return {"sessionToken": {"iss": issuer}, "keyId": machine_key_id,
                "tenants": {tenant_id: {}}}

    human_verifier = HumanVerifier("descope-project", issuer, tenant_id, "management-key", machine_key_id)
    human_verifier._client = SimpleNamespace(
        validate_session=validate_session,
        mgmt=SimpleNamespace(user=SimpleNamespace(load_by_user_id=load_user)),
    )
    machine_verifier = MachineVerifier("descope-project", issuer, tenant_id, machine_key_id, service_id)
    machine_verifier._client = SimpleNamespace(exchange_access_key=exchange_access_key)
    app = FastAPI()
    app.include_router(router)
    app.state.managed_auth_store = db
    app.state.managed_auth_service = ManagedAuthService(db, Decimal("500"))
    app.state.managed_auth_verifier = human_verifier
    app.state.managed_machine_verifier = machine_verifier
    prefix = "/admin/managed-auth"
    project_path = f"/projects/{project_id}"
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
            async def call(token, method, path, **kwargs):
                return await client.request(method, prefix + path,
                                            headers={"Authorization": f"Bearer {token}"}, **kwargs)


            owner_identity = {"issuer": issuer, "subject": subjects["owner"]}
            admin_identity = {"issuer": issuer, "subject": subjects["admin"]}
            machine_identity = {"kind": "machine", "issuer": issuer, "subject": machine_key_id}
            owner = {"kind": "personal", "identity": owner_identity}
            assert (await client.get(prefix + "/catalog")).status_code == 401
            assert (await call("machine-access-key", "GET", "/catalog")).status_code == 401
            assert (await call("machine-jwt", "GET", "/catalog")).status_code == 401
            assert (await call("machine-sub-jwt", "GET", "/catalog")).status_code == 401
            assert (await call("machine-dsub-jwt", "GET", "/catalog")).status_code == 401
            assert load_calls == []
            await asyncio.gather(*(call("owner", "GET", "/catalog") for _ in range(3)))
            capabilities = (await call("owner", "GET", "/capabilities")).json()
            assert capabilities["can_request"] is True
            assert "personal_budget_ids" not in capabilities
            assert (await call("admin", "POST", "/budgets", json={
                "kind": "personal", "period": "monthly", "limit_usd": "25.00",
                "identity": owner_identity,
            })).status_code == 404
            create_body = {"project_id": str(project_id),
                           "display_name": "owner key", "description": "test", "owner": owner,
                           "key_limit": None}
            assert (await call("owner", "POST", "/requests", json={
                key: value for key, value in create_body.items() if key != "project_id"
            })).status_code == 422
            assert (await call("owner", "POST", "/requests", json={
                **create_body, "description": "",
            })).status_code == 422
            assert (await call("owner", "POST", "/requests", json=create_body)).status_code == 409
            await db.create_service(service_id, f"registered-{uuid4()}",
                                    HumanIdentity(issuer, subjects["admin"]))
            first_machine = await call("machine-access-key", "POST", "/machine/requests", json={
                "project_id": str(project_id), "project_organization_id": str(organization),
                "project_environment": "dev", "display_name": "first project key",
                "description": "worker", "requested_budget": "5000", "budget_reason": "full rerun",
            })
            assert first_machine.status_code == 200, first_machine.text
            request_id = first_machine.json()["id"]
            assert first_machine.json()["status"] == "pending" and first_machine.json()["key_id"] is None
            assert first_machine.json()["project"]["project_id"] == str(project_id)
            assert await db.get_project_policy(project_id) is None
            assert (await call("admin", "GET", f"{project_path}/policy")).status_code == 404
            assert (await call("owner", "POST", f"/requests/{request_id}/approve",
                               json={"initial_policy": {
                                   "period": "monthly", "base_limit_usd": "20.00"}})).status_code == 403
            assert (await call("admin", "POST", f"/requests/{request_id}/approve",
                               json={})).status_code == 422
            for invalid_policy in (
                {"period": None, "base_limit_usd": None},
                {"period": "monthly", "base_limit_usd": "0"},
                {"period": "monthly", "base_limit_usd": None},
                {"period": None, "base_limit_usd": "20"},
            ):
                rejected = await call("admin", "POST", f"/requests/{request_id}/approve",
                                      json={"initial_policy": invalid_policy})
                assert rejected.status_code == 422, rejected.text
                pending = (await call("admin", "GET", f"/requests/{request_id}")).json()
                assert pending["status"] == "pending" and pending["key_id"] is None
                assert await db.get_project_policy(project_id) is None
            approved = await call("admin", "POST", f"/requests/{request_id}/approve",
                                  json={"initial_policy": {
                                      "period": "monthly", "base_limit_usd": "20.00"}})
            assert approved.status_code == 200, approved.text
            direct = await call("owner", "POST", "/requests", json=create_body)
            assert direct.status_code == 200 and direct.json()["status"] == "approved"
            key = (await call("owner", "GET", f"/keys/{direct.json()['key_id']}")).json()
            key_id = key["id"]
            assert key["status"] == "pending_collection" and key["revision"] == 1
            assert key["project"]["project_id"] == str(project_id)
            assert "verifier_hash" not in key and "secret" not in key
            assert key["key_limit"] is None
            policy = (await call("admin", "GET", f"{project_path}/policy")).json()
            assert policy["period"] == "monthly" and Decimal(policy["base_limit_usd"]) == Decimal("20")
            assert policy["revision"] == 1
            assert (await call("owner", "GET", f"/keys/{key_id}")).json()["key_limit"] is None
            assert (await call("admin", "POST", f"/keys/{key_id}/collect",
                               json={"expected_revision": 1})).status_code == 403
            assert (await call("outsider", "GET", f"/keys/{key_id}")).json()["id"] == key_id
            collected = await call("owner", "POST", f"/keys/{key_id}/collect",
                                   json={"expected_revision": 1})
            assert collected.status_code == 200, collected.text
            secret = collected.json()["secret"]
            assert secret.startswith("mgwk_")
            assert collected.json()["key"]["status"] == "active"
            assert "secret" not in (await call("owner", "GET", f"/keys/{key_id}")).json()
            assert (await db.get_key_by_verifier(managed_key_verifier(secret))).id == UUID(key_id)
            limit_path = f"/keys/{key_id}/limit"
            key_limit = {"period": "daily", "limit_usd": "9.50"}
            limit_body = {"expected_revision": 2, "key_limit": key_limit}
            assert (await call("owner", "PUT", limit_path, json=limit_body)).status_code == 403
            assert (await call("teammate", "PUT", limit_path, json=limit_body)).status_code == 403
            assert (await call("machine-access-key", "PUT", limit_path, json=limit_body)).status_code == 401
            assert (await call("admin", "PUT", limit_path, json={
                **limit_body, "expected_revision": 1,
            })).status_code == 409
            capped = await call("admin", "PUT", limit_path, json=limit_body)
            assert capped.status_code == 200, capped.text
            assert capped.json()["revision"] == 3
            assert capped.json()["key_limit"] == key_limit
            assert capped.json()["project"] == key["project"]
            assert (await db.get_key_by_verifier(managed_key_verifier(secret))).id == UUID(key_id)
            assert (await call("owner", "GET", f"/keys/{key_id}")).json()["key_limit"] == key_limit
            changed_limit = {"period": "monthly", "limit_usd": "12.00"}
            assert (await call("owner", "PUT", limit_path, json={
                "expected_revision": 3, "key_limit": changed_limit,
            })).status_code == 403
            changed_cap = await call("admin", "PUT", limit_path, json={
                "expected_revision": 3, "key_limit": changed_limit,
            })
            assert changed_cap.status_code == 200, changed_cap.text
            assert changed_cap.json()["revision"] == 4
            assert changed_cap.json()["key_limit"] == changed_limit
            assert (await call("machine-access-key", "PUT", limit_path, json={
                "expected_revision": 4, "key_limit": None,
            })).status_code == 401
            assert (await call("owner", "POST", f"/keys/{key_id}/collect",
                               json={"expected_revision": 1})).status_code == 409
            rotated = await call("owner", "POST", f"/keys/{key_id}/rotate",
                                 json={"expected_revision": 4})
            assert rotated.status_code == 200 and rotated.json()["secret"] != secret
            assert rotated.json()["key"]["key_limit"] == changed_limit
            cleared = await call("admin", "PUT", limit_path, json={
                "expected_revision": 5, "key_limit": None,
            })
            assert cleared.status_code == 200, cleared.text
            assert cleared.json()["revision"] == 6 and cleared.json()["key_limit"] is None
            assert cleared.json()["project"] == key["project"]
            assert (await db.get_key_by_verifier(
                managed_key_verifier(rotated.json()["secret"]))).id == UUID(key_id)
            assert (await call("owner", "GET", f"/keys/{key_id}")).json()["key_limit"] is None
            assert (await call("owner", "POST", f"/keys/{key_id}/revoke",
                               json={"expected_revision": 6})).status_code == 200
            assert (await call("owner", "GET", f"/keys/{key_id}")).json()["status"] == "revoked"

            service = await call("admin", "POST", "/services", json={"name": f"batch-{uuid4()}"})
            assert service.status_code == 200, service.text
            managed_service_id = service.json()["id"]
            assert (await call("owner", "POST", "/requests", json={
                **create_body, "owner": {"kind": "service", "service_id": managed_service_id},
            })).status_code == 403
            additional_body = {**create_body, "display_name": "shared project key",
                               "owner": {"kind": "service", "service_id": managed_service_id}}
            additional = await call("admin", "POST", "/requests", json=additional_body)
            assert additional.status_code == 200, additional.text
            assert additional.json()["status"] == "approved"
            additional_key_id = additional.json()["key_id"]
            assert additional_key_id != key_id
            assert (await call("admin", "GET", f"{project_path}/policy")).json()["revision"] == 1
            second_key = (await call("owner", "GET", f"/keys/{additional_key_id}")).json()
            assert second_key["project"] == key["project"]
            assert second_key["key_limit"] is None
            assert (await call("admin", "POST", f"/requests/{additional.json()['id']}/approve",
                               json={})).status_code == 409
            second_collected = await call("admin", "POST", f"/keys/{additional_key_id}/collect",
                                          json={"expected_revision": 1})
            assert second_collected.status_code == 200, second_collected.text
            service_limit = {"period": "weekly", "limit_usd": "4.25"}
            capped_service = await call("admin", "PUT", f"/keys/{additional_key_id}/limit", json={
                "expected_revision": 2, "key_limit": service_limit,
            })
            assert capped_service.status_code == 200, capped_service.text
            assert capped_service.json()["key_limit"] == service_limit
            assert (await call("admin", "POST", "/requests", json={
                **additional_body, "project_id": str(other_project_id),
                "display_name": "replacement in another project",
            })).status_code == 409
            replacement_request = await call("machine-access-key", "POST", "/machine/requests", json={
                "project_id": str(other_project_id), "project_organization_id": str(organization),
                "project_environment": "dev", "display_name": "replacement in another project",
                "description": "worker", "requested_budget": "5000", "budget_reason": "full rerun",
            })
            assert replacement_request.status_code == 200 and replacement_request.json()["status"] == "pending"
            replacement = await call(
                "admin", "POST", f"/requests/{replacement_request.json()['id']}/approve",
                json={"initial_policy": {
                    "period": "monthly", "base_limit_usd": "11.00"}},
            )
            assert replacement.status_code == 200, replacement.text
            assert replacement.json()["id"] != additional_key_id
            assert replacement.json()["project"]["project_id"] == str(other_project_id)
            assert (await call("machine-access-key", "POST", f"/machine/keys/{replacement.json()['id']}/collect",
                               json={"expected_revision": 1})).status_code == 200
            assert (await call("owner", "GET", f"/keys/{additional_key_id}")).json()["status"] == "active"
            assert (await call("owner", "POST", f"/keys/{additional_key_id}/rotate",
                               json={"expected_revision": 3})).status_code == 403
            assert (await db.get_key_by_verifier(
                managed_key_verifier(second_collected.json()["secret"]))).id == UUID(additional_key_id)

            assert (await call("outsider", "POST", f"{project_path}/limit-requests", json={
                "kind": "temporary", "increase_usd": "3.00", "reason": "exception",
            })).status_code == 403
            temporary = await call("admin", "POST", f"{project_path}/limit-requests", json={
                "kind": "temporary", "increase_usd": "3.00", "reason": "team capacity",
            })
            assert temporary.status_code == 200, temporary.text
            temporary_id = temporary.json()["id"]
            assert temporary.json()["creator"] == admin_identity
            assert temporary.json()["status"] == "pending"
            assert temporary.json()["period_start"] is None
            for invalid_policy in (
                {"period": None, "base_limit_usd": None},
                {"period": "monthly", "base_limit_usd": "0"},
                {"period": "monthly", "base_limit_usd": None},
                {"period": None, "base_limit_usd": "25"},
            ):
                rejected = await call("admin", "PUT", f"{project_path}/policy",
                                      json={"expected_revision": 1, **invalid_policy})
                assert rejected.status_code == 422, rejected.text
                unchanged = await db.get_project_policy(project_id)
                assert unchanged.revision == 1
                assert unchanged.period == "monthly"
                assert unchanged.base_limit_usd == Decimal("20.00")
            assert (await call("admin", "PUT", f"{project_path}/policy", json={
                "expected_revision": 1, "period": "monthly", "base_limit_usd": "25.00",
            })).status_code == 200
            assert (await call("admin", "PUT", f"{project_path}/policy", json={
                "expected_revision": 1, "period": "monthly", "base_limit_usd": "30.00",
            })).status_code == 409
            assert (await call("owner", "PUT", f"{project_path}/policy", json={
                "expected_revision": 2, "period": "monthly", "base_limit_usd": "30.00",
            })).status_code == 403
            pending_policy = await call("admin", "GET", f"{project_path}/policy")
            assert pending_policy.status_code == 200
            assert Decimal(pending_policy.json()["base_limit_usd"]) == Decimal("25.00")
            assert (await call("machine-access-key", "POST", f"/limit-requests/{temporary_id}/approve",
                               json={"expected_current_policy_revision": 2})).status_code == 401
            assert (await call("teammate", "POST", f"/limit-requests/{temporary_id}/approve",
                               json={"expected_current_policy_revision": 2})).status_code == 403
            assert (await call("admin", "POST", f"/limit-requests/{temporary_id}/approve",
                               json={"expected_current_policy_revision": 1})).status_code == 409
            assert (await call("admin", "GET", f"/limit-requests/{temporary_id}")).json()["status"] == "pending"
            approved_temp = await call("admin", "POST", f"/limit-requests/{temporary_id}/approve",
                                       json={"expected_current_policy_revision": 2})
            assert approved_temp.status_code == 200, approved_temp.text
            assert approved_temp.json()["status"] == "approved"
            start = datetime.fromisoformat(approved_temp.json()["period_start"])
            end = datetime.fromisoformat(approved_temp.json()["period_end"])
            assert start <= datetime.now(UTC) < end
            after_temporary = await call("admin", "GET", f"{project_path}/policy")
            assert after_temporary.status_code == 200
            assert Decimal(after_temporary.json()["base_limit_usd"]) == Decimal("25.00")
            assert (await call("admin", "POST", f"/limit-requests/{temporary_id}/approve",
                               json={"expected_current_policy_revision": 3})).status_code == 409
            permanent = await call("admin", "POST", f"{project_path}/limit-requests", json={
                "kind": "permanent", "increase_usd": "7.00", "reason": "recurring usage",
            })
            assert permanent.status_code == 200, permanent.text
            assert permanent.json()["status"] == "pending"
            assert Decimal((await call("admin", "GET", f"{project_path}/policy")).json()[
                "base_limit_usd"
            ]) == Decimal("25.00")
            approved_perm = await call("admin", "POST", f"/limit-requests/{permanent.json()['id']}/approve",
                                       json={"expected_current_policy_revision": 3})
            assert approved_perm.status_code == 200, approved_perm.text
            assert approved_perm.json()["status"] == "approved"
            after_permanent = await call("admin", "GET", f"{project_path}/policy")
            assert after_permanent.status_code == 200
            assert Decimal(after_permanent.json()["base_limit_usd"]) == Decimal("25.00")
            assert Decimal(after_permanent.json()["total_limit_usd"]) == Decimal("35.00")
            switched = await call("admin", "PUT", f"{project_path}/policy", json={
                "expected_revision": 4, "period": "weekly", "base_limit_usd": "32.00",
            })
            assert switched.status_code == 200, switched.text
            assert switched.json()["period"] == "weekly"
            assert Decimal(switched.json()["base_limit_usd"]) == Decimal("32.00")
            assert switched.json()["revision"] == 5
            retained = (await call("admin", "GET", f"/limit-requests/{temporary_id}")).json()
            assert datetime.fromisoformat(retained["period_start"]) == start
            assert datetime.fromisoformat(retained["period_end"]) == end
            denied_request = await call("admin", "POST", f"{project_path}/limit-requests", json={
                "kind": "permanent", "increase_usd": "2.00", "reason": "not approved",
            })
            assert denied_request.status_code == 200 and denied_request.json()["status"] == "pending"
            denied = await call("admin", "POST", f"/limit-requests/{denied_request.json()['id']}/deny")
            assert denied.status_code == 200 and denied.json()["status"] == "denied"
            assert (await call("admin", "GET", f"{project_path}/policy")).json()["revision"] == 5
            assert (await call("admin", "GET", f"/keys/{additional_key_id}")).json()["project"] == key["project"]

            machine_body = {"project_id": str(machine_project_id),
                            "project_organization_id": str(organization),
                            "project_environment": "dev", "display_name": "machine first key",
                            "description": "worker", "requested_budget": "5000",
                            "budget_reason": "full rerun"}
            machine_request = await call("machine-access-key", "POST", "/machine/requests", json=machine_body)
            assert machine_request.status_code == 200, machine_request.text
            machine_request_id = machine_request.json()["id"]
            assert machine_request.json()["status"] == "pending"
            assert machine_request.json()["owner"] == {"kind": "service", "service_id": str(service_id)}
            assert machine_request.json()["creator"] == machine_identity
            assert await db.get_project_policy(machine_project_id) is None
            assert (await call("machine-access-key", "GET", f"/machine/requests/{machine_request_id}")).json()["status"] == "pending"
            assert (await call("machine-access-key", "POST", "/machine/requests", json={
                **machine_body, "description": "",
            })).status_code == 422
            assert (await call("machine-access-key", "POST", f"/requests/{machine_request_id}/approve",
                               json={"initial_policy": {
                                   "period": "monthly", "base_limit_usd": "12"}})).status_code == 401
            assert (await call("machine-access-key", "PUT", f"/projects/{machine_project_id}/policy",
                               json={"expected_revision": 1, "period": "monthly",
                                     "base_limit_usd": "13"})).status_code == 401
            assert (await call("machine-access-key", "POST", f"/projects/{machine_project_id}/limit-requests",
                               json={"kind": "temporary", "increase_usd": "1", "reason": "no"})).status_code == 401
            assert (await call("machine-jwt", "POST", f"/requests/{machine_request_id}/approve",
                               json={})).status_code == 401
            assert (await call("machine-jwt", "POST", f"/projects/{machine_project_id}/limit-requests",
                               json={"kind": "permanent", "increase_usd": "1", "reason": "no"})).status_code == 401
            assert (await call("machine-sub-jwt", "POST", f"/requests/{machine_request_id}/approve",
                               json={})).status_code == 401
            assert machine_key_id not in load_calls
            approved_machine = await call("admin", "POST", f"/requests/{machine_request_id}/approve",
                                          json={"initial_policy": {
                                              "period": "monthly", "base_limit_usd": "12.00"}})
            assert approved_machine.status_code == 200, approved_machine.text
            machine_status = await call("machine-access-key", "GET",
                                        f"/machine/requests/{machine_request_id}")
            assert machine_status.status_code == 200 and machine_status.json()["status"] == "approved"
            issued_machine_key_id = machine_status.json()["key_id"]
            assert issued_machine_key_id
            assert (await call("machine-access-key", "GET",
                               f"/machine/keys/{additional_key_id}")).status_code == 403
            assert (await call("machine-access-key", "POST",
                               f"/machine/keys/{additional_key_id}/collect",
                               json={"expected_revision": 2})).status_code == 403
            assert (await call("machine-access-key", "GET",
                               f"/machine/requests/{additional.json()['id']}")).status_code == 403
            machine_key_response = await call("machine-access-key", "GET",
                                              f"/machine/keys/{issued_machine_key_id}")
            assert machine_key_response.status_code == 200, machine_key_response.text
            machine_key = machine_key_response.json()
            assert machine_key["creator"] == machine_identity
            assert machine_key["status"] == "pending_collection" and machine_key["revision"] == 1
            assert "secret" not in machine_key and "verifier_hash" not in machine_key
            machine_collected = await call("machine-access-key", "POST",
                                           f"/machine/keys/{issued_machine_key_id}/collect",
                                           json={"expected_revision": machine_key["revision"]})
            assert machine_collected.status_code == 200, machine_collected.text
            assert (await db.get_key_by_verifier(managed_key_verifier(
                machine_collected.json()["secret"]))).id == UUID(issued_machine_key_id)
            assert (await call("machine-access-key", "POST",
                               f"/machine/keys/{issued_machine_key_id}/collect",
                               json={"expected_revision": machine_key["revision"]})).status_code == 409
            human_created = await call("admin", "POST", "/requests", json={
                **create_body, "project_id": str(machine_project_id),
                "owner": {"kind": "service", "service_id": str(service_id)},
                "display_name": "same service human-created key",
            })
            assert human_created.status_code == 200 and human_created.json()["status"] == "approved"
            human_created_status = await call(
                "machine-access-key", "GET", f"/machine/requests/{human_created.json()['id']}",
            )
            assert human_created_status.status_code == 200
            human_created_key = human_created_status.json()["key_id"]
            human_created_metadata = await call(
                "machine-access-key", "GET", f"/machine/keys/{human_created_key}",
            )
            assert human_created_metadata.status_code == 200, human_created_metadata.text
            machine_collected_human_key = await call(
                "machine-access-key", "POST", f"/machine/keys/{human_created_key}/collect",
                json={"expected_revision": human_created_metadata.json()["revision"]},
            )
            assert machine_collected_human_key.status_code == 200, machine_collected_human_key.text
            assert machine_collected_human_key.json()["key"]["creator"] == {
                "kind": "human", "issuer": issuer, "subject": subjects["admin"],
            }
            next_machine = await call("machine-access-key", "POST", "/machine/requests", json={
                **machine_body, "display_name": "machine second key",
            })
            assert next_machine.status_code == 200, next_machine.text
            assert next_machine.json()["status"] == "approved"
            next_machine_status = await call(
                "machine-access-key", "GET", f"/machine/requests/{next_machine.json()['id']}",
            )
            assert next_machine_status.status_code == 200
            next_machine_key_id = next_machine_status.json()["key_id"]
            assert next_machine_key_id != issued_machine_key_id
            assert (await db.get_project_policy(machine_project_id)).base_limit_usd == Decimal("12.00")
            next_machine_metadata = await call(
                "machine-access-key", "GET", f"/machine/keys/{next_machine_key_id}",
            )
            assert next_machine_metadata.status_code == 200
            assert (await call("machine-access-key", "POST",
                               f"/machine/keys/{next_machine_key_id}/collect",
                               json={"expected_revision": next_machine_metadata.json()["revision"]})).status_code == 200
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_machine_request_requested_budget_auto_approves_at_threshold_and_waits_above():
    engine = create_engine(os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"], 5)
    db = ManagedAuthStore(engine)
    issuer = "https://identity.example"
    tenant_id = f"descope-tenant-{uuid4()}"
    machine_key_id = f"registered-machine-{uuid4()}"
    admin_subject = f"admin-{uuid4()}"
    service_id, organization = uuid4(), uuid4()
    at_threshold, above, keeps_amount, admin_typed = (uuid4() for _ in range(4))

    async def validate_session(token):
        assert token == "admin"
        return {"sessionToken": {"iss": issuer, "sub": admin_subject}}

    async def load_user(user_id):
        return {"user": {"userId": user_id, "status": "enabled",
                         "userTenants": [{"tenantId": tenant_id}],
                         "permissions": ["platform-admin"]}}

    async def exchange_access_key(access_key):
        assert access_key == "machine-access-key"
        return {"sessionToken": {"iss": issuer}, "keyId": machine_key_id,
                "tenants": {tenant_id: {}}}

    human_verifier = HumanVerifier("descope-project", issuer, tenant_id, "management-key", machine_key_id)
    human_verifier._client = SimpleNamespace(
        validate_session=validate_session,
        mgmt=SimpleNamespace(user=SimpleNamespace(load_by_user_id=load_user)),
    )
    machine_verifier = MachineVerifier("descope-project", issuer, tenant_id, machine_key_id, service_id)
    machine_verifier._client = SimpleNamespace(exchange_access_key=exchange_access_key)
    app = FastAPI()
    app.include_router(router)
    app.state.managed_auth_store = db
    app.state.managed_auth_service = ManagedAuthService(db, Decimal("500"))
    app.state.managed_auth_verifier = human_verifier
    app.state.managed_machine_verifier = machine_verifier
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
            async def call(token, method, path, **kwargs):
                return await client.request(method, "/admin/managed-auth" + path,
                                            headers={"Authorization": f"Bearer {token}"}, **kwargs)

            def body(project_id, **budget):
                return {"project_id": str(project_id), "project_organization_id": str(organization),
                        "project_environment": "dev", "display_name": "project key",
                        "description": "worker", **budget}

            await db.create_service(service_id, f"registered-{uuid4()}", HumanIdentity(issuer, admin_subject))
            for invalid in (
                {},
                {"requested_budget": "300"},
                {"budget_reason": "sweep"},
                {"requested_budget": "0", "budget_reason": "sweep"},
                {"requested_budget": "300", "budget_reason": "   "},
            ):
                assert (await call("machine-access-key", "POST", "/machine/requests",
                                   json=body(uuid4(), **invalid))).status_code == 422

            approved = await call("machine-access-key", "POST", "/machine/requests", json=body(
                at_threshold, requested_budget="500", budget_reason="nightly sweep"))
            assert approved.status_code == 200, approved.text
            assert approved.json()["status"] == "approved" and approved.json()["key_id"] is not None
            assert approved.json()["decided_by"] is None
            assert approved.json()["requested_budget"] == "500"
            assert approved.json()["budget_reason"] == "nightly sweep"
            policy = await db.get_project_policy(at_threshold)
            assert policy is not None and (policy.period, policy.base_limit_usd) == ("monthly", Decimal("500"))
            key = (await call("machine-access-key", "GET", f"/machine/keys/{approved.json()['key_id']}")).json()
            assert key["status"] == "pending_collection" and key["description"] == "worker"
            history = (await call("admin", "GET", f"/projects/{at_threshold}/history")).json()
            assert [(event["event"], event["actor"], event["reason"]) for event in history] == [(
                "approved", {"issuer": "managed-auth-system", "subject": "requested-budget-auto-approval"},
                "nightly sweep",
            )]
            async with engine.connect() as connection:
                detail = (await connection.execute(
                    select(audit.c.detail).where(audit.c.target_kind == "project_policy",
                                                 audit.c.target_id == at_threshold)
                )).scalar_one()
            assert detail["auto_approval_threshold_usd"] == "500"

            pending = {}
            for project_id, budget in ((above, {"requested_budget": "500.01", "budget_reason": "full rerun"}),
                                       (keeps_amount, {"requested_budget": "5000", "budget_reason": "full rerun"}),
                                       (admin_typed, {"requested_budget": "5000", "budget_reason": "full rerun"})):
                response = await call("machine-access-key", "POST", "/machine/requests", json=body(project_id, **budget))
                assert response.status_code == 200, response.text
                assert response.json()["status"] == "pending" and response.json()["key_id"] is None
                assert await db.get_project_policy(project_id) is None
                pending[project_id] = response.json()
            assert (pending[above]["requested_budget"], pending[above]["budget_reason"]) == ("500.01", "full rerun")
            listed = {item["id"]: item for item in (await call("admin", "GET", "/requests")).json()}
            assert listed[pending[keeps_amount]["id"]]["requested_budget"] == "5000"
            assert listed[pending[keeps_amount]["id"]]["budget_reason"] == "full rerun"

            assert (await call("admin", "POST", f"/requests/{pending[keeps_amount]['id']}/approve",
                               json={})).status_code == 422
            for project_id, amount in ((keeps_amount, "5000"), (admin_typed, "750")):
                approval = await call("admin", "POST", f"/requests/{pending[project_id]['id']}/approve",
                                      json={"initial_policy": {"period": "monthly", "base_limit_usd": amount}})
                assert approval.status_code == 200, approval.text
                policy = await db.get_project_policy(project_id)
                assert policy is not None and policy.base_limit_usd == Decimal(amount)
            assert (await call("admin", "POST", f"/requests/{pending[above]['id']}/approve", json={
                "initial_policy": {"period": "monthly", "base_limit_usd": "500.01"}})).status_code == 200
            assert (await call("admin", "GET", f"/projects/{keeps_amount}/history")).json()[0]["reason"] == "full rerun"
    finally:
        await engine.dispose()

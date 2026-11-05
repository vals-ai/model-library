"""The approved, populated managed-auth Alembic upgrade on isolated PostgreSQL."""

import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import pytest
from alembic.config import Config
from alembic.runtime.environment import EnvironmentContext
from alembic.script import ScriptDirectory
from redis.asyncio import Redis
from sqlalchemy import MetaData, Table, event, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

import model_gateway.app as gateway_app
from model_gateway.managed_auth import tables
from model_gateway.managed_auth.authority_publisher import ManagedAuthorityPublisher
from model_gateway.managed_auth.credentials import create_managed_key_secret
from model_gateway.managed_auth.database import create_engine
from model_gateway.managed_auth.reconciliation import HumanReconciler
from model_gateway.managed_auth.store import ManagedAuthStore
from model_gateway.managed_auth.types import HumanIdentity, ProjectReference
from model_gateway.run_token_types import ModelPermissions
from model_gateway.run_tokens import RunTokenClaims, mint_run_token, revoke_lease
from model_library.retriers.token import utils as token_utils


# The verified R52 key/request/project triples are deliberately independent of the migration.
ASSIGNMENTS = (
    ("ca6ae698-e5fd-4859-882b-b54ef10250e3", "483276d1-b136-41ac-ad4b-c68307c79621", "cb5760f5-c76e-44e6-bf42-1bc02c13310a"),
    ("fc078ef7-e36d-4b80-b06e-9de39d074a68", "08d071a1-8fa6-4133-a862-cd024fe2e63e", "372ca605-aee0-4e05-862a-f96594dd24ca"),
    ("b8ef4474-41d9-4359-9d7d-6013909c54c2", "d8076991-03a0-4c43-aa3d-22e4409690ec", "bb9a8186-f2f3-4cb2-874c-a461d42d1bf1"),
    ("f20de50f-2be4-44b4-bc25-306db827fd76", "e664cd37-b6d1-4d73-9f57-e8cfde8616f4", "f749d1d2-9119-4732-9ca7-21df58da6c5e"),
    ("7d9ee5db-a30e-4507-ad6a-7801bf1059c6", "b2b20528-7dfc-46b3-9adb-c77c9a420beb", "c9970734-457b-4bec-806b-212a5431cbbf"),
    ("8e91ab65-dec6-42d5-9214-8dee700d81b6", "e8b2a299-31e6-4db5-85a3-fe94b24b2b72", "89c9cc80-fe36-4a7c-b110-c8a3c1d0680a"),
    ("231d1d24-9104-4496-9ce1-5352af2b51f6", "c3f1791a-a1fd-4ecd-a86b-a9e71b042d00", "f749d1d2-9119-4732-9ca7-21df58da6c5e"),
    ("9008f09f-44d1-4fc1-97ce-f2c026c97bd3", "ea09bb27-f113-44cf-9521-05ffca6e19a7", "e70b0f22-fe64-40fc-84be-12555064545c"),
    ("b49a5484-4c08-4194-92b6-6a8c83a49a8a", "11d6053b-b1b2-4670-9587-d5c09126dbe7", "29011c39-60be-42cd-95fa-49e575ecf926"),
    ("aef93c7c-4b2c-4dda-a1c8-33aaa2e0f120", "6fd85856-1fa8-4d1a-a078-5fb7ea595016", "77bb3ea8-55de-4d9d-8b6d-f7910285cd13"),
    ("76ce5d10-4295-4683-b5b9-98b028ac211f", "c91a3358-3ad3-4fc3-b7ac-ce714b67af0e", "929fc456-80a7-4d49-9cc4-7b1f47785b43"),
    ("73178426-7667-4a52-98ec-41b34198f4fa", "655dc53e-f008-43de-bfab-d694ea9d764c", "12257d67-d453-426d-9298-4387c37741ee"),
    ("ce1c0bc1-9f94-4d86-adec-d11701033f7d", "ba8e447f-5a33-4f86-b856-e6589ded298a", "4fd784ba-0d38-4db2-81bd-251ed2aa5593"),
    ("ae1e074b-8a9c-430a-a2e9-bf61ac5684de", "208ac89c-187c-479a-b2b1-671fd7f2889e", "f05afb28-95c7-41fe-9f74-b899874a6689"),
)
RSI_KEY_ID = UUID("ca6ae698-e5fd-4859-882b-b54ef10250e3")
CUA_KEY_ID = UUID("fc078ef7-e36d-4b80-b06e-9de39d074a68")
CUA_REQUEST_ID = UUID("08d071a1-8fa6-4133-a862-cd024fe2e63e")
CANARY_KEY_ID = UUID("b8ef4474-41d9-4359-9d7d-6013909c54c2")
CANARY_PROJECT_ID = UUID("bb9a8186-f2f3-4cb2-874c-a461d42d1bf1")
ORGANIZATION = "431bc4f8-c66b-47eb-8d53-2c4622be04e6"
ENVIRONMENT = "dev"  # R52 historical routing input, not current deployment attestation.
MIGRATIONS = Path(__file__).resolve().parents[3] / "model_gateway/managed_auth/alembic.ini"
CREATED = datetime(2026, 10, 1, 12, tzinfo=UTC)
DECIDED = datetime(2026, 10, 2, 12, tzinfo=UTC)
UPDATED = datetime(2026, 10, 3, 12, tzinfo=UTC)


def _upgrade(connection, revision):
    config = Config(str(MIGRATIONS))
    script = ScriptDirectory.from_config(config)
    with EnvironmentContext(
        config, script, fn=lambda current, context: script._upgrade_revs(revision, current),
        destination_rev=revision,
    ) as context:
        context.configure(connection=connection, transactional_ddl=True)
        with context.begin_transaction():
            context.run_migrations()


@asynccontextmanager
async def _isolated_database():
    # A unique search_path keeps Alembic's version table and every managed table
    # outside the integration suite's shared public schema.
    schema = "managed_migration_" + uuid4().hex
    admin = create_engine(os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"], 2)
    engine = create_async_engine(
        os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"],
        connect_args={"server_settings": {"search_path": schema}},
    )
    try:
        async with admin.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        yield engine
    finally:
        await engine.dispose()
        async with admin.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


async def _revision(engine, revision):
    async with engine.connect() as connection:
        await connection.run_sync(_upgrade, revision)


async def _seed_old_population(engine):
    owner = HumanIdentity("https://issuer.example", f"migration-owner-{uuid4()}")
    service_id, group_id = uuid4(), uuid4()
    root, verifier = create_managed_key_secret()
    # Only a disposable credential is used; neither credential nor digest is persisted outside PostgreSQL.
    async with engine.begin() as connection:
        def seed(sync_connection):
            metadata = MetaData()
            names = (
                "managed_humans", "managed_services", "managed_service_managers",
                "managed_model_groups", "managed_human_model_grants", "managed_keys",
                "managed_key_groups", "managed_key_requests", "managed_request_groups",
                "managed_audit",
            )
            old = {name: Table(name, metadata, autoload_with=sync_connection) for name in names}
            sync_connection.execute(old["managed_humans"].insert().values(
                issuer=owner.issuer, subject=owner.subject, enabled=True,
            ))
            sync_connection.execute(old["managed_services"].insert().values(id=service_id, name="migration-service"))
            sync_connection.execute(old["managed_service_managers"].insert().values(
                service_id=service_id, issuer=owner.issuer, subject=owner.subject,
            ))
            sync_connection.execute(old["managed_model_groups"].insert().values(
                id=group_id, name=f"migration-{group_id}",
                permissions=ModelPermissions(
                    allowed_models=["openai/gpt-4o"], allowed_operations=["query"],
                ).model_dump(mode="json"),
                revision=3, created_at=CREATED, updated_at=UPDATED,
            ))
            sync_connection.execute(old["managed_human_model_grants"].insert().values(
                issuer=owner.issuer, subject=owner.subject, group_id=group_id,
            ))
            for index, (key, request, _) in enumerate(ASSIGNMENTS):
                key_id, request_id = UUID(key), UUID(request)
                personal = key_id in (RSI_KEY_ID, CUA_KEY_ID)
                owner_fields = (
                    {"owner_kind": "personal", "owner_issuer": owner.issuer, "owner_subject": owner.subject}
                    if personal else {"owner_kind": "service", "service_id": service_id}
                )
                sync_connection.execute(old["managed_keys"].insert().values(
                    id=key_id, display_name=f"historical-key-{index}",
                    creator_issuer=owner.issuer, creator_subject=owner.subject,
                    **owner_fields, status="pending_collection" if key_id == RSI_KEY_ID else "active",
                    verifier_hash=None if key_id == RSI_KEY_ID else verifier if key_id == CANARY_KEY_ID else f"synthetic-verifier-{uuid4()}",
                    revision=index + 2, created_at=CREATED, updated_at=UPDATED,
                ))
                sync_connection.execute(old["managed_key_groups"].insert().values(
                    key_id=key_id, group_id=group_id,
                ))
                sync_connection.execute(old["managed_key_requests"].insert().values(
                    id=request_id, creator_issuer=owner.issuer, creator_subject=owner.subject,
                    kind="create", display_name=f"historical-key-{index}", reason="synthetic migration fixture",
                    **owner_fields, status="approved", decided_by_issuer=owner.issuer,
                    decided_by_subject=owner.subject, decided_at=DECIDED,
                    key_id=key_id, created_at=CREATED,
                ))
                sync_connection.execute(old["managed_request_groups"].insert().values(
                    request_id=request_id, group_id=group_id, approved=False,
                ))
                sync_connection.execute(old["managed_request_groups"].insert().values(
                    request_id=request_id, group_id=group_id, approved=True,
                ))
            sync_connection.execute(old["managed_audit"].insert().values(
                id=uuid4(), actor_kind="human", actor_issuer=owner.issuer,
                actor_subject=owner.subject, target_kind="key", target_id=CANARY_KEY_ID,
                change="created", occurred_at=CREATED,
            ))
            sync_connection.execute(old["managed_audit"].insert().values(
                id=uuid4(), actor_kind="system", actor_issuer="managed-auth-system",
                actor_subject="eligibility-sync", target_kind="key", target_id=CANARY_KEY_ID,
                change="eligibility_synced", occurred_at=UPDATED,
            ))
        await connection.run_sync(seed)
    return owner, service_id, root, verifier


async def _old_rows(engine, baseline=None):
    async with engine.connect() as connection:
        def snapshot(sync_connection):
            meta = MetaData()
            names = ("managed_keys", "managed_key_requests", "managed_key_groups", "managed_request_groups", "managed_audit", "managed_model_groups", "managed_service_managers")
            result = {}
            for name in names:
                table = Table(name, meta, autoload_with=sync_connection)
                columns = (baseline[name][0] if baseline is not None
                           else tuple(column.name for column in table.c if column.name != "verifier_hash"))
                result[name] = (columns, tuple(tuple(row) for row in sync_connection.execute(
                    select(*(table.c[column] for column in columns)).order_by(*table.primary_key.columns)
                )))
            return result
        return await connection.run_sync(snapshot)


class _Settings:
    def get(self, name, default):
        return {"MODEL_GATEWAY_API_KEYS": json.dumps({"operations": "sk-local"}),
                "MODEL_GATEWAY_HMAC_SECRET": "local-only-secret"}[name]


@pytest.mark.asyncio
async def test_populated_upgrade_preserves_authority_and_existing_child(monkeypatch):
    async with _isolated_database() as engine:
        await _revision(engine, "7e40c64e19a2")
        owner, service_id, root, verifier = await _seed_old_population(engine)
        before = await _old_rows(engine)
        async with engine.connect() as connection:
            old_verifiers = dict((await connection.execute(
                select(tables.keys.c.id, tables.keys.c.verifier_hash)
            )).all())
        redis = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
        namespace = f"managed-migration-{uuid4()}"
        owner_key = f"gateway:managed-auth:{namespace}:owner"
        active_key = f"gateway:managed-auth:{namespace}:active"
        owner_token = uuid4().hex
        first_key = CANARY_KEY_ID
        monkeypatch.setenv("MODEL_GATEWAY_AUTH_NAMESPACE", namespace)
        monkeypatch.setenv("MODEL_GATEWAY_AUTH_POOL_SIZE", "4")
        monkeypatch.setenv("GATEWAY_RUNTIME_ROLE", "control")
        monkeypatch.setenv("GATEWAY_USAGE_LEDGER_MODE", "disabled")
        monkeypatch.setenv("DESCOPE_PROJECT_ID", "local")
        monkeypatch.setenv("DESCOPE_EXPECTED_ISSUER", owner.issuer)
        monkeypatch.setenv("DESCOPE_ORGANIZATION_ID", ORGANIZATION)
        monkeypatch.setenv("DESCOPE_MACHINE_ACCESS_KEY_ID", f"machine-{uuid4()}")
        monkeypatch.setenv("DESCOPE_MACHINE_SERVICE_OWNER_ID", str(uuid4()))
        monkeypatch.setenv("DESCOPE_MANAGEMENT_CREDENTIAL", "local-only")
        child_claims = RunTokenClaims(
            run_id=f"migration-{uuid4()}", task_id="task", attempt_id="attempt",
            allowed_models=["openai/gpt-4o"], allowed_operations=["query"],
            minted_by="historical-producer", managed_key_id=first_key,
        )
        child_lease = None
        try:
            monkeypatch.setattr(token_utils, "redis_client", redis)
            child, child_lease = await mint_run_token(child_claims, 300)
            await _revision(engine, "head")
            assert before == await _old_rows(engine, before)
            async with engine.connect() as connection:
                policies = (await connection.execute(select(tables.project_policies))).all()
                assert len(policies) == 13
                assert {row.project_id for row in policies} == {UUID(p) for _, _, p in ASSIGNMENTS}
                assert all(row.project_organization == ORGANIZATION and row.project_environment == ENVIRONMENT
                           and row.period == "monthly" and row.base_limit_usd == Decimal("10000")
                           and row.revision == 1 for row in policies)
                assert (await connection.execute(select(tables.project_limit_requests))).all() == []
                keys = {row.id: row for row in (await connection.execute(select(tables.keys))).all()}
                requests = {row.id: row for row in (await connection.execute(select(tables.requests))).all()}
                for key, request, project in ASSIGNMENTS:
                    stored_key, stored_request = keys[UUID(key)], requests[UUID(request)]
                    assert stored_key.project_id == stored_request.project_id == UUID(project)
                    assert stored_key.owner_kind == (
                        "personal" if stored_key.id in (RSI_KEY_ID, CUA_KEY_ID) else "service"
                    )
                    assert stored_key.status == (
                        "pending_collection" if stored_key.id == RSI_KEY_ID else "active"
                    )
                    assert stored_key.creator_kind == stored_request.creator_kind == "human"
                    assert stored_key.key_limit_period is None and stored_key.key_limit_usd is None
                    assert stored_request.project_organization == ORGANIZATION
                    assert stored_request.project_environment == ENVIRONMENT
                assert all(keys[key_id].verifier_hash == digest for key_id, digest in old_verifiers.items())
                assert keys[RSI_KEY_ID].verifier_hash is None
                assert keys[RSI_KEY_ID].status == "pending_collection"
                assert keys[CUA_KEY_ID].owner_kind == "personal"
                assert keys[CUA_KEY_ID].status == "active"
                assert keys[CANARY_KEY_ID].service_id == service_id
                actors = (await connection.execute(select(tables.actors))).all()
                assert ("human", owner.issuer, owner.subject) in {tuple(row) for row in actors}
                assert ("system", "managed-auth-system", "eligibility-sync") in {tuple(row) for row in actors}
            store = ManagedAuthStore(engine)
            assert (await store.get_key_by_verifier(verifier)).id == first_key
            assert (await store.get_key(first_key)).project == ProjectReference(CANARY_PROJECT_ID, ORGANIZATION, ENVIRONMENT)
            await redis.set(owner_key, owner_token, nx=True, ex=3600)
            with patch.object(gateway_app, "model_library_settings", _Settings()), patch.object(
                gateway_app, "create_managed_engine", return_value=engine,
            ):
                app = gateway_app.create_app()
                async with app.router.lifespan_context(app):
                    publisher = ManagedAuthorityPublisher(
                        engine, redis, HumanReconciler(store, app.state.managed_auth_verifier, owner.issuer),
                        namespace=namespace,
                    )
                    assert await publisher.publish_once(owner_token)
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app), base_url="http://testserver",
                    ) as client:
                        assert (await client.get("/registry", headers={"Authorization": f"Bearer {root}"})).status_code == 200
                        child_response = await client.get("/registry", headers={"Authorization": f"Bearer {child}"})
                        assert child_response.status_code == 200
        finally:
            await redis.delete(owner_key, active_key)
            if child_lease is not None:
                monkeypatch.setattr(token_utils, "redis_client", redis)
                await revoke_lease(child_lease, child_claims)
            await redis.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("unexpected", ["key", "request"])
async def test_unmapped_population_preflight_refuses_without_schema_or_data_changes(unexpected):
    async with _isolated_database() as engine:
        await _revision(engine, "7e40c64e19a2")
        await _seed_old_population(engine)
        async with engine.begin() as connection:
            if unexpected == "key":
                await connection.execute(text(
                    "INSERT INTO managed_keys (id, display_name, creator_issuer, creator_subject, "
                    "owner_kind, owner_issuer, owner_subject, status, verifier_hash, revision, created_at, updated_at) "
                    "SELECT :id, 'unmapped', creator_issuer, creator_subject, owner_kind, owner_issuer, "
                    "owner_subject, 'pending_collection', NULL, 1, created_at, updated_at "
                    "FROM managed_keys WHERE id = :known"
                ), {"id": uuid4(), "known": CUA_KEY_ID})
            else:
                await connection.execute(text(
                    "INSERT INTO managed_key_requests (id, creator_issuer, creator_subject, kind, display_name, "
                    "reason, owner_kind, owner_issuer, owner_subject, status, decided_by_issuer, "
                    "decided_by_subject, decided_at, key_id, created_at) "
                    "SELECT :id, creator_issuer, creator_subject, kind, display_name, reason, owner_kind, "
                    "owner_issuer, owner_subject, status, decided_by_issuer, decided_by_subject, decided_at, "
                    "key_id, created_at FROM managed_key_requests WHERE id = :known"
                ), {"id": uuid4(), "known": CUA_REQUEST_ID})
        before = await _old_rows(engine)
        with pytest.raises(RuntimeError, match="^Managed keys and requests differ from the approved DEV migration inventory$"):
            await _revision(engine, "head")
        assert before == await _old_rows(engine, before)
        async with engine.connect() as connection:
            assert (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalar_one() == "7e40c64e19a2"
            assert await connection.run_sync(lambda sync: "managed_project_policies" not in inspect(sync).get_table_names())


@pytest.mark.asyncio
async def test_late_old_writer_rolls_back_populated_upgrade_after_ddl():
    async with _isolated_database() as engine:
        await _revision(engine, "7e40c64e19a2")
        await _seed_old_population(engine)
        async with engine.connect() as connection:
            schema = (await connection.execute(text("SELECT current_schema()"))).scalar_one()

        async def schema_shape():
            async with engine.connect() as connection:
                def shape(sync_connection):
                    inspector = inspect(sync_connection)
                    return {
                        name: tuple((column["name"], column["nullable"]) for column in inspector.get_columns(name))
                        for name in inspector.get_table_names()
                    }
                return await connection.run_sync(shape)

        before_schema = await schema_shape()
        before_rows = await _old_rows(engine)
        late_key_id, late_request_id = uuid4(), uuid4()
        writer_engine = create_async_engine(
            os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"],
            connect_args={"server_settings": {"search_path": schema}},
        )
        observed = {"first_ddl": False, "late_commit": False, "policy_count": None}

        async def insert_late_pair():
            async with writer_engine.begin() as writer:
                await writer.execute(text(
                    "INSERT INTO managed_keys (id, display_name, creator_issuer, creator_subject, "
                    "owner_kind, owner_issuer, owner_subject, status, verifier_hash, revision, created_at, updated_at) "
                    "SELECT CAST(:id AS uuid), 'late-old-writer', creator_issuer, creator_subject, "
                    "owner_kind, owner_issuer, owner_subject, 'pending_collection', NULL, 1, created_at, updated_at "
                    "FROM managed_keys WHERE id = :known"
                ), {"id": late_key_id, "known": CUA_KEY_ID})
                await writer.execute(text(
                    "INSERT INTO managed_key_requests (id, creator_issuer, creator_subject, kind, display_name, "
                    "reason, owner_kind, owner_issuer, owner_subject, status, decided_by_issuer, "
                    "decided_by_subject, decided_at, key_id, created_at) "
                    "SELECT CAST(:id AS uuid), creator_issuer, creator_subject, kind, display_name, reason, "
                    "owner_kind, owner_issuer, owner_subject, status, decided_by_issuer, decided_by_subject, "
                    "decided_at, CAST(:key_id AS uuid), created_at FROM managed_key_requests WHERE id = :known"
                ), {"id": late_request_id, "key_id": late_key_id, "known": CUA_REQUEST_ID})
            observed["late_commit"] = True

        def intercept(connection, cursor, statement, parameters, execution_context, executemany):
            normalized = " ".join(statement.upper().split())
            if normalized.startswith("CREATE TABLE MANAGED_ACTORS"):
                observed["first_ddl"] = True
                connection.connection.run_async(lambda _driver: insert_late_pair())
            if "ALTER TABLE MANAGED_KEYS ALTER COLUMN CREATOR_KIND SET NOT NULL" in normalized:
                observed["policy_count"] = connection.exec_driver_sql(
                    "SELECT COUNT(*) FROM managed_project_policies"
                ).scalar_one()

        event.listen(engine.sync_engine, "before_cursor_execute", intercept)
        try:
            with pytest.raises(IntegrityError) as failure:
                await _revision(engine, "head")
            assert failure.value.orig.sqlstate == "23502"
            assert observed == {"first_ddl": True, "late_commit": True, "policy_count": 13}
            assert await schema_shape() == before_schema
            after_rows = await _old_rows(engine, before_rows)
            for name, (_, old_rows) in before_rows.items():
                new_rows = after_rows[name][1]
                assert all(row in new_rows for row in old_rows)
                assert len(new_rows) == len(old_rows) + (
                    1 if name in ("managed_keys", "managed_key_requests") else 0
                )
            async with engine.connect() as connection:
                assert (await connection.execute(text(
                    "SELECT version_num FROM alembic_version"
                ))).scalar_one() == "7e40c64e19a2"
                assert (await connection.execute(text(
                    "SELECT key_id FROM managed_key_requests WHERE id = :id"
                ), {"id": late_request_id})).scalar_one() == late_key_id
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", intercept)
            await writer_engine.dispose()


@pytest.mark.asyncio
async def test_empty_upgrade_does_not_seed_population_specific_policies():
    async with _isolated_database() as engine:
        await _revision(engine, "7e40c64e19a2")
        await _revision(engine, "head")
        async with engine.connect() as connection:
            assert (await connection.execute(select(tables.project_policies))).all() == []
            assert (await connection.execute(select(tables.keys))).all() == []


@pytest.mark.asyncio
async def test_positive_project_policy_upgrade_refuses_invalid_rows_until_corrected():
    async with _isolated_database() as engine:
        await _revision(engine, "c6f10b5ea842")
        uncapped_id, zero_id = uuid4(), uuid4()
        async with engine.begin() as connection:
            await connection.execute(text(
                "INSERT INTO managed_project_policies "
                "(project_id, project_organization, project_environment, period, base_limit_usd, "
                "revision, created_at, updated_at) "
                "VALUES (:id, 'org', 'dev', :period, :amount, 1, now(), now())"
            ), [
                {"id": uncapped_id, "period": None, "amount": None},
                {"id": zero_id, "period": "monthly", "amount": 0},
            ])
        with pytest.raises(IntegrityError) as missing_period:
            await _revision(engine, "head")
        assert missing_period.value.orig.sqlstate == "23502"
        async with engine.connect() as connection:
            assert (await connection.execute(text(
                "SELECT version_num FROM alembic_version"
            ))).scalar_one() == "c6f10b5ea842"
            assert (await connection.execute(text(
                "SELECT project_id, period, base_limit_usd, revision "
                "FROM managed_project_policies ORDER BY project_id"
            ))).all() == sorted([
                (uncapped_id, None, None, 1),
                (zero_id, "monthly", Decimal("0"), 1),
            ])
        async with engine.begin() as connection:
            await connection.execute(text(
                "UPDATE managed_project_policies SET period = 'monthly', base_limit_usd = 10 "
                "WHERE project_id = :id"
            ), {"id": uncapped_id})
        with pytest.raises(IntegrityError) as zero_base:
            await _revision(engine, "head")
        assert zero_base.value.orig.sqlstate == "23514"
        async with engine.connect() as connection:
            assert (await connection.execute(text(
                "SELECT version_num FROM alembic_version"
            ))).scalar_one() == "c6f10b5ea842"
            assert (await connection.execute(text(
                "SELECT base_limit_usd, revision FROM managed_project_policies "
                "WHERE project_id = :id"
            ), {"id": zero_id})).one() == (Decimal("0"), 1)
        async with engine.begin() as connection:
            await connection.execute(text(
                "UPDATE managed_project_policies SET base_limit_usd = 20 "
                "WHERE project_id = :id"
            ), {"id": zero_id})
        await _revision(engine, "head")
        async with engine.connect() as connection:
            assert (await connection.execute(text(
                "SELECT version_num FROM alembic_version"
            ))).scalar_one() == "b8e1f3a6c2d4"
        for invalid in ({"period": None}, {"base_limit_usd": None}, {"base_limit_usd": 0}):
            with pytest.raises(IntegrityError):
                async with engine.begin() as connection:
                    await connection.execute(text(
                        "UPDATE managed_project_policies SET period = :period, "
                        "base_limit_usd = :amount WHERE project_id = :id"
                    ), {
                        "id": uncapped_id,
                        "period": invalid.get("period", "monthly"),
                        "amount": invalid.get("base_limit_usd", 10),
                    })
        with pytest.raises(IntegrityError):
            async with engine.begin() as connection:
                await connection.execute(text(
                    "INSERT INTO managed_project_policies "
                    "(project_id, project_organization, project_environment, period, base_limit_usd, "
                    "revision, created_at, updated_at) "
                    "VALUES (:id, 'org', 'dev', 'monthly', 0, 1, now(), now())"
                ), {"id": uuid4()})


@pytest.mark.asyncio
async def test_requested_budget_reason_pair_rejects_a_reason_without_a_budget():
    async with _isolated_database() as engine:
        await _revision(engine, "head")
        insert = text(
            "INSERT INTO managed_key_requests "
            "(id, creator_kind, creator_issuer, creator_subject, kind, display_name, reason, "
            "owner_kind, owner_issuer, owner_subject, status, created_at, "
            "project_id, project_organization, project_environment, requested_budget, budget_reason) "
            "VALUES (:id, 'human', 'issuer', 'subject', 'create', 'key', 'worker', "
            "'personal', 'issuer', 'subject', 'pending', now(), :project, 'org', 'dev', :budget, :budget_reason)"
        )
        async with engine.begin() as connection:
            await connection.execute(text(
                "INSERT INTO managed_actors (kind, issuer, subject) VALUES ('human', 'issuer', 'subject')"
            ))
            await connection.execute(text(
                "INSERT INTO managed_humans (issuer, subject, enabled) VALUES ('issuer', 'subject', true)"
            ))
            await connection.execute(insert, [
                {"id": uuid4(), "project": uuid4(), "budget": None, "budget_reason": None},
                {"id": uuid4(), "project": uuid4(), "budget": 300, "budget_reason": "nightly sweep"},
            ])
        with pytest.raises(IntegrityError) as reason_only:
            async with engine.begin() as connection:
                await connection.execute(insert, {
                    "id": uuid4(), "project": uuid4(), "budget": None, "budget_reason": "nightly sweep",
                })
        assert reason_only.value.orig.sqlstate == "23514"
        assert "requested_budget_reason_pair" in str(reason_only.value)

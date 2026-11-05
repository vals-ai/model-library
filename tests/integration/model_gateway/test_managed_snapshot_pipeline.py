"""Warehouse facts charge capped projects and managed roots independently of email."""

import asyncio
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch
from uuid import uuid4

import asyncpg
import boto3
import pytest
from redis.asyncio import Redis

from model_gateway.budgets.schema import SpendSnapshot, managed_key_spend_id, period_bounds
from model_gateway.budgets.service import BudgetService
from model_gateway.managed_auth.budget_sources import list_key_spend_sources
from model_gateway.managed_auth.database import create_engine
from model_gateway.managed_auth.runtime import RuntimeAuthority, RuntimeProjectLimit
from model_gateway.managed_auth.store import ManagedAuthStore
from model_gateway.managed_auth.types import (
    HumanIdentity, KeyLimit, MachineIdentity, PersonalOwner, ProjectReference,
    ServiceOwner,
)
from model_gateway.run_token_types import ModelPermissions
from model_gateway.usage_ledger import redshift_export_job
from model_gateway.usage_ledger.redshift_export_job import RedshiftExportJobConfig, _publish_managed_budget_snapshot


class PostgreSQLWarehouseDataAPI:
    """Execute the actual publisher SQL against a local fact table, not Redshift."""

    def __init__(self, url: str):
        self.url = url.replace("postgresql+asyncpg://", "postgresql://")
        self.records = []
        self.statements = []
        self.parameters = []

    def execute_statement(self, **kwargs):
        self.statements.append(kwargs["Sql"])
        parameters = kwargs.get("Parameters", [])
        self.parameters.append({item["name"]: item["value"] for item in parameters})
        sql = kwargs["Sql"]
        for index, item in enumerate(parameters, start=1):
            sql = re.sub(rf":{item['name']}\b", f"${index}::text", sql)

        async def query():
            connection = await asyncpg.connect(self.url)
            try:
                return await connection.fetch(sql, *(item["value"] for item in parameters))
            finally:
                await connection.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            rows = executor.submit(asyncio.run, query()).result()
        self.records = [
            [{"nullValue": True} if value is None else {"stringValue": str(value)} for value in row]
            for row in rows
        ]
        return {"Id": "local-postgresql-fact-query"}

    def describe_statement(self, **kwargs):
        return {"Status": "FINISHED"}

    def get_statement_result(self, **kwargs):
        return {"Records": self.records}


@pytest.mark.asyncio
async def test_managed_fact_spend_publishes_and_refreshes_query_decision(caplog):
    database_url = os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"]
    redis = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    engine = create_engine(database_url, 4)
    store = ManagedAuthStore(engine)
    stage = f"snapshot-{uuid4()}"
    schema = f"snapshot_{uuid4().hex}"
    other_schema = f"snapshot_{uuid4().hex}"
    bucket = f"snapshot-{uuid4()}"
    key = f"managed-{uuid4()}.json"
    alert_key = f"gateway-usage/{stage}/spend/project-alerts-current.json"
    s3 = boto3.client(
        "s3", endpoint_url="http://127.0.0.1:26391", region_name="us-east-1",
        aws_access_key_id="local", aws_secret_access_key="local",
    )
    s3.create_bucket(Bucket=bucket)
    secrets = boto3.client(
        "secretsmanager", endpoint_url="http://127.0.0.1:26391", region_name="us-east-1",
        aws_access_key_id="local", aws_secret_access_key="local",
    )
    secret_id = f"snapshot-database-{uuid4()}"
    secrets.create_secret(Name=secret_id, SecretString=os.environ["MODEL_GATEWAY_AUTH_DATABASE_SECRET"])
    actor = HumanIdentity("https://issuer.example", f"snapshot-{uuid4()}")
    email = f"owner-{uuid4()}@example.com"
    project = ProjectReference(uuid4(), "org", "dev")
    other_project = ProjectReference(uuid4(), "org", "dev")
    empty_project = ProjectReference(uuid4(), "org", "dev")
    optional_key_project = ProjectReference(uuid4(), "org", "dev")
    root_key_id, peer_key_id, other_key_id, empty_key_id, uncapped_key_id = (
        uuid4() for _ in range(5)
    )
    old_budget_a, old_budget_b = uuid4(), uuid4()
    optional_service_key_id = uuid4()
    warehouse = await asyncpg.connect(database_url.replace("postgresql+asyncpg://", "postgresql://"))
    now = datetime.now(UTC)
    managed_redis_key = f"gateway:budgets:{stage}:managed-spend-v2"
    try:
        await store.record_human(actor, True, actor)
        for approved_project, limit, key_id in (
            (project, ("monthly", Decimal("3")), root_key_id),
            (project, None, peer_key_id),
            (other_project, ("monthly", Decimal("20")), other_key_id),
            (empty_project, ("monthly", Decimal("5")), empty_key_id),
            (optional_key_project, ("monthly", Decimal("10")), uncapped_key_id),
        ):
            request_id = uuid4()
            await store.create_request(
                request_id, "snapshot key", "spend attribution",
                actor, PersonalOwner(actor), None, approved_project, None, None,
            )
            await store.decide_request(
                request_id, actor, key_id, limit, None,
            )
        service_id = uuid4()
        await store.create_service(service_id, "snapshot service", actor)
        service_owner = ServiceOwner(service_id)
        tracker = MachineIdentity("https://issuer.example", f"tracker-{uuid4()}")
        for candidate, creator, label, description, revoked in (
            (project, tracker, "Old project label", "Valkyrie project old-handle", True),
            (project, actor, "Competing human label", "human-created", False),
            (project, tracker, "Current project label", "Valkyrie project current-handle", False),
            (project, tracker, "Current project label", "Valkyrie project current-handle", False),
            (other_project, actor, "Manual project label", "Valkyrie project not-a-tracker-handle", False),
            (empty_project, actor, "Empty project label", "manual project", False),
            (optional_key_project, actor, "Optional key project label", "manual project", False),
        ):
            request_id = uuid4()
            service_key_id = optional_service_key_id if candidate == optional_key_project else uuid4()
            await store.create_request(
                request_id, label, description, creator, service_owner, None, candidate,
                None, None,
            )
            issued = await store.decide_request(request_id, actor, service_key_id, None, None)
            assert issued is not None
            if revoked:
                assert await store.revoke_key(service_key_id, issued.revision, actor) is not None
            elif creator == actor and candidate in (project, other_project):
                assert await store.collect_key(
                    service_key_id, issued.revision, uuid4().hex, actor,
                ) is not None
        sources = [policy.project.project_id for policy in await store.list_project_policies()]
        assert {project.project_id, other_project.project_id, empty_project.project_id,
                optional_key_project.project_id} <= set(sources)
        assert root_key_id not in await list_key_spend_sources(engine)
        assert uncapped_key_id not in await list_key_spend_sources(engine)
        await warehouse.execute(f"create schema {schema}")
        await warehouse.execute(f"""create table {schema}.usage_events (
            completed_at timestamptz not null, budget_id varchar(128), project_id varchar(128),
            managed_key_id varchar(128), identity_email varchar(1024),
            requested_model_key varchar(2048) not null, benchmark_name varchar(1024),
            cost_usd numeric(38, 12))""")
        await warehouse.execute(f"""create table {schema}.usage_agg_1d (
            bucket_start_utc timestamptz not null, identity_email varchar(1024),
            provider_model varchar(2048), cost_usd_sum numeric(38, 12),
            cost_recorded_count int)""")
        await warehouse.execute(f"create schema {other_schema}")
        await warehouse.execute(f"create table {other_schema}.usage_events "
                                f"(like {schema}.usage_events)")
        client = PostgreSQLWarehouseDataAPI(database_url)
        managed_control = BudgetService(stage=stage, redis=redis, bucket=bucket, key=key)
        managed_control._s3 = s3
        query = BudgetService(stage=stage, redis=redis)
        config = RedshiftExportJobConfig(
            raw_table_name="local", shards=1, redshift_workgroup_name="local",
            redshift_database_name="local", redshift_secret_arn="local",
            landing_bucket_name=bucket, copy_role_arn="local",
            redshift_schema_name=schema, export_prefix="local",
            budget_snapshot_bucket=bucket,
            managed_budget_snapshot_key=key,
            project_alert_snapshot_key=alert_key,
            managed_database_secret_arn=secret_id,
            stage=stage,
        )

        def publish(data_through, *, idle=False):
            with (
                patch("model_gateway.usage_ledger.budgets.managed_publisher.gateway_registry.fetch_live_models", return_value={
                    "included": {"metadata": {"ignored_for_cost": False}},
                    "second": {"metadata": {"ignored_for_cost": False}},
                    "third": {"metadata": {"ignored_for_cost": False}},
                    "fourth": {"metadata": {"ignored_for_cost": False}},
                    "ignored": {"metadata": {"ignored_for_cost": True}},
                }),
                patch("model_gateway.usage_ledger.redshift_export_job.boto3.client", return_value=secrets),
            ):
                _publish_managed_budget_snapshot(
                    config=config, s3_client=s3, redshift_data_client=client,
                    data_through=data_through, idle=idle,
                )

        backdated = period_bounds("monthly", now)[0] - timedelta(days=10)
        await warehouse.execute(
            "update managed_project_policies set total_started_at = $1 where project_id = $2",
            backdated, other_project.project_id,
        )
        await asyncio.to_thread(publish, now)
        bound = next(list(b.values()) for b in client.parameters if str(other_project.project_id) in b.values())
        assert bound[bound.index(str(other_project.project_id)) + 1] == backdated.isoformat()
        await managed_control.refresh_control_once()
        project_limit = RuntimeProjectLimit("monthly", Decimal("3"), ())
        other_limit = RuntimeProjectLimit("monthly", Decimal("20"), ())
        root_limit = KeyLimit("monthly", Decimal("2"))
        uncapped_limit = KeyLimit("daily", Decimal("1"))

        def authority(
            key_id, target: ProjectReference,
            target_limit: RuntimeProjectLimit, key_limit: KeyLimit | None,
        ) -> RuntimeAuthority:
            return RuntimeAuthority(
                key_id=key_id, owner=PersonalOwner(actor),
                permissions=ModelPermissions(allowed_models=[], allowed_operations=[]),
                project=target, project_limit=target_limit, key_limit=key_limit,
            )

        root_authority = authority(root_key_id, project, project_limit, None)
        peer_authority = authority(peer_key_id, project, project_limit, None)
        other_authority = authority(other_key_id, other_project, other_limit, None)
        assert await query.managed_decision(root_authority, now) == "allowed"
        assert await query.managed_decision(peer_authority, now) == "allowed"
        first_cap = SpendSnapshot.model_validate_json(await redis.get(managed_redis_key))
        assert first_cap.stage == stage
        assert first_cap.revision == 1
        assert first_cap.data_through == now
        first_alert = json.loads(s3.get_object(Bucket=bucket, Key=alert_key)["Body"].read())
        assert first_alert["schema_version"] == 1
        assert first_alert["stage"] == stage
        assert first_alert["revision"] == 1
        assert datetime.fromisoformat(first_alert["data_through"]) == now
        assert len(first_alert["projects"]) == 4
        first_projects = {item["project_id"]: item for item in first_alert["projects"]}
        assert first_projects[str(project.project_id)] == {
            "project_id": str(project.project_id),
            "organization_id": "org",
            "name": "Current project label",
            "handle": "current-handle",
            "environment": "dev",
            "period": "monthly",
            "start": first_cap.periods["monthly"].start.isoformat(),
            "end": first_cap.periods["monthly"].end.isoformat(),
            "limit_usd": "3",
            "spend_usd": "0",
            "contributors": [],
            "models": [],
            "benchmarks": [],
        }
        assert first_projects[str(other_project.project_id)]["name"] == "Manual project label"
        assert first_projects[str(other_project.project_id)]["handle"] is None
        assert first_projects[str(empty_project.project_id)]["name"] == "Empty project label"
        assert first_projects[str(empty_project.project_id)]["handle"] is None
        for period in ("daily", "weekly", "monthly"):
            assert first_cap.periods[period].spend_usd[str(project.project_id)] == 0
            assert first_cap.periods[period].spend_usd[str(other_project.project_id)] == 0
            assert first_cap.periods[period].spend_usd[str(empty_project.project_id)] == 0
            assert first_cap.periods[period].spend_usd[str(optional_key_project.project_id)] == 0
            assert managed_key_spend_id(root_key_id) not in first_cap.periods[period].spend_usd
            assert managed_key_spend_id(uncapped_key_id) not in first_cap.periods[period].spend_usd
        await warehouse.executemany(
            f"insert into {schema}.usage_events values ($1, $2, $3, $4, $5, $6, $7, $8)",
            [
                (now, str(old_budget_a), str(project.project_id), str(root_key_id), email, "included", "alpha", Decimal("1")),
                (now, str(old_budget_b), str(project.project_id), str(root_key_id), email, "second", "beta", Decimal("2")),
                (now, None, str(project.project_id), str(peer_key_id), email, "third", "gamma", Decimal("3")),
                (now, None, str(project.project_id), None, email, "included", "delta", Decimal("2")),
                (now, None, None, None, email, "included", None, Decimal("4")),
                (now, str(old_budget_a), str(other_project.project_id), str(other_key_id), email, "included", "elsewhere", Decimal("5")),
                (now, str(old_budget_a), str(project.project_id), str(root_key_id), email, "ignored", "excluded", Decimal("12")),
                (now, None, str(optional_key_project.project_id), str(uncapped_key_id), email, "included", None, Decimal("1")),
                (now, None, str(project.project_id), None, None, "fourth", None, Decimal("0.5")),
            ],
        )
        await warehouse.execute(
            f"insert into {other_schema}.usage_events values ($1, $2, $3, $4, $5, $6, $7, $8)",
            now, str(old_budget_a), str(project.project_id), str(root_key_id), email,
            "included", "elsewhere", Decimal("100"),
        )
        # The aggregate is materialized from the same facts, not another event insertion.
        await warehouse.execute(f"""insert into {schema}.usage_agg_1d
            select date_trunc('day', completed_at), identity_email, requested_model_key,
                   sum(cost_usd), count(cost_usd)
            from {schema}.usage_events group by 1, 2, 3""")
        assert await warehouse.fetchval(f"select count(*) from {schema}.usage_events") == 9
        # The root attribution was recorded while both keys were still uncapped.
        root = await store.get_key(root_key_id)
        uncapped = await store.get_key(uncapped_key_id)
        assert root is not None and uncapped is not None
        assert await store.update_key_limit(
            root_key_id, root.revision, root_limit, actor,
        ) is not None
        assert await store.update_key_limit(
            uncapped_key_id, uncapped.revision, uncapped_limit, actor,
        ) is not None
        assert {root_key_id, uncapped_key_id} <= set(await list_key_spend_sources(engine))
        root_authority = authority(root_key_id, project, project_limit, root_limit)
        uncapped_authority = authority(uncapped_key_id, optional_key_project, RuntimeProjectLimit("monthly", Decimal("10"), ()), uncapped_limit)
        await asyncio.to_thread(publish, now)
        await managed_control.refresh_control_once()
        assert await query.managed_decision(root_authority, now) == "exhausted"
        assert await query.managed_decision(peer_authority, now) == "exhausted"
        assert await query.managed_decision(other_authority, now) == "allowed"
        policy = await store.get_project_policy(project.project_id)
        assert policy is not None
        raised = await store.update_project_policy(
            project.project_id, policy.revision, "monthly", Decimal("10"), actor,
        )
        assert raised.base_limit_usd == Decimal("10")
        raised_limit = RuntimeProjectLimit("monthly", Decimal("10"), ())
        assert await query.managed_decision(
            authority(root_key_id, project, raised_limit, root_limit), now,
        ) == "key_exhausted"
        assert await query.managed_decision(
            authority(peer_key_id, project, raised_limit, None), now,
        ) == "allowed"
        assert await query.managed_decision(uncapped_authority, now) == "key_exhausted"
        cap = SpendSnapshot.model_validate_json(await redis.get(managed_redis_key))
        assert cap.stage == stage
        assert cap.revision == first_cap.revision + 1
        assert cap.data_through == first_cap.data_through
        for period in ("daily", "weekly", "monthly"):
            assert cap.periods[period].start == first_cap.periods[period].start
            assert cap.periods[period].end == first_cap.periods[period].end
            assert cap.periods[period].spend_usd[str(project.project_id)] == Decimal("8.5")
            assert cap.periods[period].spend_usd[str(other_project.project_id)] == Decimal("5")
            assert cap.periods[period].spend_usd[str(empty_project.project_id)] == 0
            assert cap.periods[period].spend_usd[str(optional_key_project.project_id)] == Decimal("1")
            assert cap.periods[period].spend_usd[managed_key_spend_id(root_key_id)] == Decimal("3")
            assert cap.periods[period].spend_usd[managed_key_spend_id(uncapped_key_id)] == Decimal("1")
        alert = json.loads(s3.get_object(Bucket=bucket, Key=alert_key)["Body"].read())
        alert_projects = {item["project_id"]: item for item in alert["projects"]}
        assert alert["revision"] == first_alert["revision"] + 1
        assert datetime.fromisoformat(alert["data_through"]) == now
        assert alert_projects[str(project.project_id)]["name"] == "Current project label"
        assert alert_projects[str(project.project_id)]["handle"] == "current-handle"
        assert alert_projects[str(other_project.project_id)]["name"] == "Manual project label"
        assert alert_projects[str(other_project.project_id)]["handle"] is None
        assert alert_projects[str(project.project_id)]["spend_usd"] == "8.500000000000"
        assert alert_projects[str(project.project_id)]["contributors"] == [
            {"email": None, "spend_usd": "0.500000000000"},
            {"email": email, "spend_usd": "8.000000000000"},
        ]
        assert alert_projects[str(project.project_id)]["models"] == [
            {"name": "fourth", "spend_usd": "0.500000000000"},
            {"name": "included", "spend_usd": "3.000000000000"},
            {"name": "second", "spend_usd": "2.000000000000"},
            {"name": "third", "spend_usd": "3.000000000000"},
        ]
        assert alert_projects[str(project.project_id)]["benchmarks"] == [
            {"name": None, "spend_usd": "0.500000000000"},
            {"name": "alpha", "spend_usd": "1.000000000000"},
            {"name": "beta", "spend_usd": "2.000000000000"},
            {"name": "delta", "spend_usd": "2.000000000000"},
            {"name": "gamma", "spend_usd": "3.000000000000"},
        ]
        assert all(
            sum(Decimal(entry["spend_usd"]) for entry in alert_projects[str(project.project_id)][dimension])
            == Decimal(alert_projects[str(project.project_id)]["spend_usd"])
            == cap.periods["monthly"].spend_usd[str(project.project_id)]
            for dimension in ("models", "benchmarks", "contributors")
        )
        assert alert_projects[str(other_project.project_id)]["spend_usd"] == "5.000000000000"
        assert alert_projects[str(empty_project.project_id)]["contributors"] == []
        assert alert_projects[str(empty_project.project_id)]["models"] == []
        assert alert_projects[str(empty_project.project_id)]["benchmarks"] == []
        assert set(json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())) == {
            "schema_version", "stage", "revision", "generated_at", "data_through", "periods",
            "total_spend_usd",
        }
        assert cap.total_spend_usd is not None
        assert cap.total_spend_usd[str(project.project_id)] == cap.periods["monthly"].spend_usd[str(project.project_id)]
        policy = await store.get_project_policy(project.project_id)
        assert policy is not None
        await store.change_temporary_increases(
            project.project_id, policy.revision, actor, "burst", "add",
            Decimal("2"), period_bounds(policy.period, datetime.now(UTC)),
        )
        await asyncio.to_thread(publish, now, idle=True)
        changed = json.loads(s3.get_object(Bucket=bucket, Key=alert_key)["Body"].read())
        changed_project = next(
            item for item in changed["projects"] if item["project_id"] == str(project.project_id)
        )
        assert changed["revision"] == alert["revision"] + 1
        assert changed_project["limit_usd"] == "12"
        assert changed_project["spend_usd"] == alert_projects[str(project.project_id)]["spend_usd"]
        assert changed_project["models"] == alert_projects[str(project.project_id)]["models"]
        assert changed_project["benchmarks"] == alert_projects[str(project.project_id)]["benchmarks"]
        assert changed["data_through"] == alert["data_through"]
        assert len(client.statements) == 3

        conflict_request_id, conflict_key_id = uuid4(), uuid4()
        await store.create_request(
            conflict_request_id, "Current project label", "Valkyrie project different-handle",
            tracker, service_owner, None, project, None, None,
        )
        conflict_key = await store.decide_request(conflict_request_id, actor, conflict_key_id, None, None)
        assert conflict_key is not None
        optional_service_key = await store.get_key(optional_service_key_id)
        assert optional_service_key is not None
        assert await store.revoke_key(
            optional_service_key_id, optional_service_key.revision, actor,
        ) is not None
        assert await store.get_project_policy(optional_key_project.project_id) is not None
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="model_gateway.managed_auth.store"):
            await asyncio.to_thread(publish, now, idle=True)
        assert sorted(
            record.getMessage() for record in caplog.records
            if record.name == "model_gateway.managed_auth.store"
            and record.levelno == logging.WARNING
        ) == sorted((
            f"Skipping project alert: project_id={project.project_id} reason=conflicting",
            f"Skipping project alert: project_id={optional_key_project.project_id} reason=missing",
        ))
        unresolved = json.loads(s3.get_object(Bucket=bucket, Key=alert_key)["Body"].read())
        assert unresolved["revision"] == changed["revision"] + 1
        assert unresolved["data_through"] == changed["data_through"]
        assert {item["project_id"] for item in unresolved["projects"]} == {
            str(other_project.project_id), str(empty_project.project_id),
        }
        unresolved_projects = {item["project_id"]: item for item in unresolved["projects"]}
        assert unresolved_projects[str(other_project.project_id)] == alert_projects[str(other_project.project_id)]
        assert unresolved_projects[str(empty_project.project_id)] == alert_projects[str(empty_project.project_id)]
        await managed_control.refresh_control_once()
        unresolved_cap = SpendSnapshot.model_validate_json(await redis.get(managed_redis_key))
        assert unresolved_cap.revision == cap.revision + 2
        assert unresolved_cap.data_through == cap.data_through
        assert unresolved_cap.periods == cap.periods
        assert {project.project_id, other_project.project_id, empty_project.project_id,
                optional_key_project.project_id} <= {
                    policy.project.project_id for policy in await store.list_project_policies()
                }
        assert await query.managed_decision(
            authority(peer_key_id, project, raised_limit, None), now,
        ) == "allowed"
        assert await query.managed_decision(uncapped_authority, now) == "key_exhausted"
        assert await query.managed_decision(other_authority, now) == "allowed"
        assert len(client.statements) == 4

        assert await store.revoke_key(
            conflict_key_id, conflict_key.revision, actor,
        ) is not None
        replacement_request_id, replacement_key_id = uuid4(), uuid4()
        await store.create_request(
            replacement_request_id, "Optional key project label", "manual project",
            actor, service_owner, None, optional_key_project, None, None,
        )
        assert await store.decide_request(
            replacement_request_id, actor, replacement_key_id, None, None,
        ) is not None
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="model_gateway.managed_auth.store"):
            await asyncio.to_thread(publish, now, idle=True)
        assert not [
            record for record in caplog.records
            if record.name == "model_gateway.managed_auth.store"
            and record.levelno == logging.WARNING
        ]
        restored = json.loads(s3.get_object(Bucket=bucket, Key=alert_key)["Body"].read())
        restored_projects = {item["project_id"]: item for item in restored["projects"]}
        assert restored["revision"] == unresolved["revision"] + 1
        assert restored["data_through"] == unresolved["data_through"]
        assert set(restored_projects) == set(first_projects)
        assert restored_projects[str(project.project_id)] == changed_project
        assert restored_projects[str(optional_key_project.project_id)] == alert_projects[str(optional_key_project.project_id)]
        assert len(client.statements) == 5
        # A failed alert read is deferred until the warehouse admission snapshot is refreshed.
        original_sources = ManagedAuthStore.list_project_allowance_sources
        original_token = redshift_export_job.read_alert_publication_token
        alert_body = s3.get_object(Bucket=bucket, Key=alert_key)["Body"].read()
        admission = SpendSnapshot.model_validate_json(
            s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        )
        calls = []

        async def read_sources(self, as_of):
            assert calls == ["token"]
            calls.append("sources")
            return await original_sources(self, as_of)

        token_error = RuntimeError("alert token read failed")

        def fail_token(*args):
            calls.append("token")
            raise token_error

        with (
            patch("model_gateway.usage_ledger.redshift_export_job.read_alert_publication_token",
                  side_effect=fail_token),
            patch.object(ManagedAuthStore, "list_project_allowance_sources", read_sources),
        ):
            with pytest.raises(RuntimeError) as captured:
                await asyncio.to_thread(publish, now, idle=True)
        assert captured.value is token_error
        assert calls == ["token", "sources"]
        assert s3.get_object(Bucket=bucket, Key=alert_key)["Body"].read() == alert_body
        refreshed = SpendSnapshot.model_validate_json(
            s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        )
        assert refreshed.revision == admission.revision + 1
        assert refreshed.periods == admission.periods
        assert len(client.statements) == 6
        await managed_control.refresh_control_once()
        assert SpendSnapshot.model_validate_json(
            await redis.get(managed_redis_key)
        ).revision == refreshed.revision

        calls.clear()
        identity_error = RuntimeError("alert identities read failed")

        def read_token(*args):
            calls.append("token")
            return original_token(*args)

        async def fail_identities(self):
            assert calls == ["token", "sources"]
            published = SpendSnapshot.model_validate_json(
                s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            )
            assert published.revision == refreshed.revision + 1
            calls.append("identities")
            raise identity_error

        with (
            patch("model_gateway.usage_ledger.redshift_export_job.read_alert_publication_token",
                  side_effect=read_token),
            patch.object(ManagedAuthStore, "list_project_allowance_sources", read_sources),
            patch.object(ManagedAuthStore, "list_project_alert_identities", fail_identities),
        ):
            with pytest.raises(RuntimeError) as captured:
                await asyncio.to_thread(publish, now, idle=True)
        assert captured.value is identity_error
        assert calls == ["token", "sources", "identities"]
        assert s3.get_object(Bucket=bucket, Key=alert_key)["Body"].read() == alert_body
        assert SpendSnapshot.model_validate_json(
            s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        ).revision == refreshed.revision + 1
        assert len(client.statements) == 7
        await managed_control.refresh_control_once()
        assert SpendSnapshot.model_validate_json(
            await redis.get(managed_redis_key)
        ).revision == refreshed.revision + 1
    finally:
        await warehouse.execute(f"drop schema if exists {other_schema} cascade")
        await warehouse.execute(f"drop schema if exists {schema} cascade")
        await warehouse.close()
        await redis.delete(managed_redis_key)
        await redis.aclose()
        await engine.dispose()
        s3.delete_object(Bucket=bucket, Key=key)
        s3.delete_object(Bucket=bucket, Key=alert_key)
        s3.delete_bucket(Bucket=bucket)
        secrets.delete_secret(SecretId=secret_id, ForceDeleteWithoutRecovery=True)

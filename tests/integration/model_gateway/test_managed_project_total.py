"""The project-total migration and admission on isolated PostgreSQL and Redis."""

import asyncio
import json
import logging
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import asyncpg
import pytest
from redis.asyncio import Redis
from alembic.config import Config
from alembic.runtime.environment import EnvironmentContext
from alembic.script import ScriptDirectory
from sqlalchemy import text

from model_gateway.budgets.schema import PeriodSpend, SpendSnapshot, period_bounds
from model_gateway.budgets.service import BudgetService
from model_gateway.managed_auth.authority_publisher import ManagedAuthorityPublisher
from model_gateway.managed_auth.store import ManagedAuthStore
from model_gateway.managed_auth.types import HumanIdentity, ProjectReference
from model_gateway.request_budget import BudgetRefusal, budget_refusal
from model_gateway.usage_ledger.budgets.managed_publisher import (
    _managed_spend_statement,
    _query_managed_spend,
)
from tests.integration.model_gateway.test_managed_auth_migration import (
    MIGRATIONS,
    _isolated_database,
    _revision,
)
from tests.integration.model_gateway.test_managed_snapshot_pipeline import PostgreSQLWarehouseDataAPI

BEFORE_TOTALS = "f2b3c4d5e6a7"
STAGE_ONE_HEAD = "b8e1f3a6c2d4"
ISSUER = "https://issuer.example"


async def _policy(connection, project_id, period, base, created_at):
    await connection.execute(
        text(
            "INSERT INTO managed_project_policies (project_id, project_organization, "
            "project_environment, period, base_limit_usd, revision, created_at, updated_at) "
            "VALUES (:id, 'org', 'dev', :period, :base, 1, :created, :created)"
        ),
        {"id": project_id, "period": period, "base": base, "created": created_at},
    )


async def _total_policy(connection, project_id, total, started):
    await connection.execute(
        text(
            "INSERT INTO managed_project_policies (project_id, project_organization, "
            "project_environment, total_limit_usd, total_started_at, revision, created_at, "
            "updated_at) VALUES (:id, 'org', 'dev', :total, :started, 1, :started, :started)"
        ),
        {"id": project_id, "total": total, "started": started},
    )


async def _temporary(connection, project_id, amount, subject, cancelled, during=None):
    now = datetime.now(UTC)
    start, end = period_bounds("monthly", during or now)
    await connection.execute(
        text(
            "INSERT INTO managed_project_limit_requests (id, project_id, kind, increase_usd, "
            "period_start, period_end, reason, creator_issuer, creator_subject, status, "
            "decided_by_issuer, decided_by_subject, decided_at, cancelled_at, "
            "cancelled_by_issuer, cancelled_by_subject, cancellation_reason, created_at) "
            "VALUES (:id, :project, 'temporary', :amount, :start, :end, 'extra', :issuer, "
            ":subject, 'approved', :issuer, :subject, :now, :cancelled, :cancelled_by_issuer, "
            ":cancelled_by_subject, :cancellation_reason, :now)"
        ),
        {
            "id": uuid4(), "project": project_id, "amount": amount, "start": start,
            "end": end, "issuer": ISSUER, "subject": subject, "now": now,
            "cancelled": now if cancelled else None,
            "cancelled_by_issuer": ISSUER if cancelled else None,
            "cancelled_by_subject": subject if cancelled else None,
            "cancellation_reason": "replaced" if cancelled else None,
        },
    )


@pytest.mark.asyncio
async def test_migration_sets_each_total_and_admission_compares_spend_with_it(caplog):
    now = datetime.now(UTC)
    this_month = period_bounds("monthly", now)[0]
    last_month = period_bounds("monthly", this_month - timedelta(days=1))[0]
    this_week = period_bounds("weekly", now)[0]
    under, over, weekly, daily, switched, fresh = uuid4(), uuid4(), uuid4(), uuid4(), uuid4(), uuid4()
    subject = f"admin-{uuid4()}"
    async with _isolated_database() as engine:
        await _revision(engine, BEFORE_TOTALS)
        async with engine.begin() as connection:
            await connection.execute(
                text("INSERT INTO managed_humans VALUES (:issuer, :subject, true)"),
                {"issuer": ISSUER, "subject": subject},
            )
            await _policy(connection, under, "monthly", Decimal("1000"), last_month + timedelta(days=22))
            await _policy(connection, over, "monthly", Decimal("1000"), this_month + timedelta(hours=1))
            await _policy(connection, weekly, "weekly", Decimal("50"), this_week - timedelta(days=13))
            await _policy(connection, daily, "daily", Decimal("10"), now - timedelta(days=2))
            await _temporary(connection, under, Decimal("500"), subject, cancelled=False)
            await _temporary(connection, under, Decimal("9000"), subject, cancelled=True)
            await _temporary(connection, under, Decimal("200"), subject, cancelled=False, during=last_month)
            await _policy(connection, switched, "monthly", Decimal("300"), last_month + timedelta(hours=1))
            await connection.execute(
                text("INSERT INTO managed_actors VALUES ('human', :issuer, :subject)"),
                {"issuer": ISSUER, "subject": subject},
            )
            await connection.execute(
                text(
                    "INSERT INTO managed_audit (id, actor_issuer, actor_subject, actor_kind, "
                    "target_kind, target_id, change, detail, occurred_at) VALUES (:id, :issuer, "
                    ":subject, 'human', 'project_policy', :project, 'policy_changed', "
                    "CAST(:detail AS json), :at)"
                ),
                {"id": uuid4(), "issuer": ISSUER, "subject": subject, "project": switched,
                 "at": this_month,
                 "detail": json.dumps({
                     "before": {"period": "daily", "base_limit_usd": "100"},
                     "after": {"period": "monthly", "base_limit_usd": "300"},
                 })},
            )
        with caplog.at_level(logging.INFO, logger="alembic.runtime.migration"):
            await _revision(engine, STAGE_ONE_HEAD)
        listing = [r.getMessage() for r in caplog.records if r.getMessage().startswith("project total ")]
        assert len(listing) == 5
        assert sorted(line.split()[2] for line in listing) == sorted(
            f"project_id={project}" for project in (under, over, weekly, daily, switched)
        )
        assert any(f"project_id={switched} states=daily:100|monthly:300 " in line for line in listing)
        assert (
            f"project total project_id={under} states=monthly:1000 allowance=2000 temporary=9700 "
            f"total=11700 start={last_month.isoformat()}"
        ) in listing

        async with engine.begin() as connection:
            await _policy(connection, fresh, "monthly", Decimal("300"), now)
        await _revision(engine, "head")
        store = ManagedAuthStore(engine)
        policies = {p.project.project_id: p for p in await store.list_project_policies()}
        days_last_month = (this_month - last_month).days
        assert {project_id: p.total_limit_usd for project_id, p in policies.items()} == {
            under: Decimal("11700"), over: Decimal("1000"), weekly: Decimal("150"), daily: Decimal("30"),
            switched: Decimal("100") * (days_last_month + 1) + Decimal("300"),
            fresh: Decimal("300"),
        }
        assert policies[under].total_started_at == last_month
        assert policies[switched].total_started_at == last_month
        assert policies[fresh].total_started_at == this_month

        redis = Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
        stage = f"total-{uuid4().hex}"
        service = BudgetService(stage=stage, redis=redis)
        try:
            spend = {str(under): Decimal("1200"), str(over): Decimal("10"), str(fresh): Decimal("0")}
            totals_spent = {str(under): Decimal("2750"), str(over): Decimal("1000"), str(fresh): Decimal("299")}
            snapshot = SpendSnapshot(
                schema_version=1, stage=stage, revision=1, generated_at=now, data_through=now,
                periods={
                    period: PeriodSpend(
                        start=period_bounds(period, now)[0], end=period_bounds(period, now)[1],
                        spend_usd=spend,
                    )
                    for period in ("daily", "weekly", "monthly")
                },
                total_spend_usd=totals_spent,
            )
            await redis.set(service.redis_key, snapshot.model_dump_json())
            publisher = ManagedAuthorityPublisher.__new__(ManagedAuthorityPublisher)
            publisher.engine = engine
            async with engine.begin() as connection:
                for project_id in (under, over, fresh):
                    await connection.execute(
                        text(
                            "INSERT INTO managed_keys (id, creator_kind, creator_issuer, "
                            "creator_subject, owner_kind, owner_issuer, owner_subject, status, "
                            "verifier_hash, revision, display_name, description, created_at, "
                            "updated_at, project_id) "
                            "VALUES (:id, 'human', :issuer, :subject, 'personal', :issuer, "
                            ":subject, 'active', :hash, 1, 'k', 'k', :now, :now, :project)"
                        ),
                        {"id": uuid4(), "issuer": ISSUER, "subject": subject,
                         "hash": uuid4().hex, "now": now, "project": project_id},
                    )
            authorities = {a.project.project_id: a for _, a in await publisher.project()}
            decisions = {
                project_id: await service.managed_decision(authority, now)
                for project_id, authority in authorities.items()
            }
            assert decisions == {under: "allowed", over: "exhausted", fresh: "allowed"}
            request = SimpleNamespace(
                app=SimpleNamespace(state=SimpleNamespace(budget_service=service)),
                state=SimpleNamespace(managed_authority=authorities[over]),
            )
            assert await budget_refusal(request) == BudgetRefusal(
                402, "budget_exceeded", "Project total budget of $1000 is spent"
            )
            fresh_limit = next(
                a.project_limit for _, a in await publisher.project()
                if a.project.project_id == fresh
            )
            assert fresh_limit is not None and fresh_limit.total_limit_usd == Decimal("300")
        finally:
            await redis.delete(service.redis_key)
            await redis.aclose()


@pytest.mark.asyncio
async def test_approved_increase_raises_the_total():
    async with _isolated_database() as engine:
        await _revision(engine, "head")
        project_id = uuid4()
        started = datetime(2026, 1, 1, tzinfo=UTC)
        admin = HumanIdentity(ISSUER, f"admin-{uuid4()}")
        async with engine.begin() as connection:
            await connection.execute(
                text("INSERT INTO managed_humans VALUES (:issuer, :subject, true)"),
                {"issuer": admin.issuer, "subject": admin.subject},
            )
            await _total_policy(connection, project_id, Decimal("250"), started)
        store = ManagedAuthStore(engine)
        request = await store.create_limit_request(
            uuid4(), ProjectReference(project_id, "org", "dev"), "temporary",
            Decimal("40"), "rerun", admin,
        )
        await store.approve_limit_request(request.id, admin, 1)
        policy = await store.get_project_policy(project_id)
        assert policy is not None
        assert (policy.total_limit_usd, policy.total_started_at) == (Decimal("290"), started)


@pytest.mark.asyncio
async def test_temporary_increase_routes_move_the_total():
    async with _isolated_database() as engine:
        await _revision(engine, "head")
        project_id = uuid4()
        admin = HumanIdentity(ISSUER, f"admin-{uuid4()}")
        async with engine.begin() as connection:
            await connection.execute(
                text("INSERT INTO managed_humans VALUES (:issuer, :subject, true)"),
                {"issuer": admin.issuer, "subject": admin.subject},
            )
            await _total_policy(connection, project_id, Decimal("250"), datetime.now(UTC))
        store = ManagedAuthStore(engine)
        added = await store.change_temporary_increases(
            project_id, 1, admin, "burst", "add", Decimal("40"),
        )
        assert added is not None and added.total_limit_usd == Decimal("290")
        replaced = await store.change_temporary_increases(
            project_id, 2, admin, "rerun", "set", Decimal("310"),
        )
        assert replaced is not None and replaced.total_limit_usd == Decimal("310")
        with pytest.raises(ValueError, match="at least the total without temporary"):
            await store.change_temporary_increases(
                project_id, 3, admin, "shrink", "set", Decimal("200"),
            )
        removed = await store.change_temporary_increases(
            project_id, 3, admin, "done", "remove", None,
        )
        assert removed is not None and removed.total_limit_usd == Decimal("250")


@pytest.mark.asyncio
async def test_a_temporary_increase_from_an_earlier_month_stays_until_removed():
    async with _isolated_database() as engine:
        await _revision(engine, "head")
        project_id = uuid4()
        admin = HumanIdentity(ISSUER, f"admin-{uuid4()}")
        async with engine.begin() as connection:
            await connection.execute(
                text("INSERT INTO managed_humans VALUES (:issuer, :subject, true)"),
                {"issuer": admin.issuer, "subject": admin.subject},
            )
            await _total_policy(connection, project_id, Decimal("250"), datetime(2026, 1, 1, tzinfo=UTC))
        store = ManagedAuthStore(engine)
        await store.change_temporary_increases(project_id, 1, admin, "burst", "add", Decimal("40"))
        async with engine.begin() as connection:
            await connection.execute(
                text("UPDATE managed_project_limit_requests SET decided_at = :old, created_at = :old"),
                {"old": datetime(2026, 2, 3, tzinfo=UTC)},
            )
        policies, extras = await store.list_project_allowance_sources()
        assert [item.increase_usd for item in extras[project_id]] == [Decimal("40")]
        removed = await store.change_temporary_increases(project_id, 2, admin, "done", "remove", None)
        assert removed is not None and removed.total_limit_usd == Decimal("250")

@pytest.mark.asyncio
async def test_total_spend_counts_every_fact_since_the_budget_start():
    now = datetime.now(UTC)
    this_month = period_bounds("monthly", now)[0]
    budget_start = period_bounds("monthly", this_month - timedelta(days=1))[0]
    project_id = uuid4()
    schema = f"warehouse_{uuid4().hex}"
    url = os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"]
    warehouse = await asyncpg.connect(url.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        await warehouse.execute(f"create schema {schema}")
        await warehouse.execute(f"""create table {schema}.usage_events (
            completed_at timestamptz not null, budget_id varchar(128), project_id varchar(128),
            managed_key_id varchar(128), identity_email varchar(1024),
            requested_model_key varchar(2048) not null, benchmark_name varchar(1024),
            cost_usd numeric(38, 12), served_model_key varchar(2048))""")
        await warehouse.executemany(
            f"insert into {schema}.usage_events values ($1, $2, $3, $4, $5, $6, $7, $8)",
            [
                (budget_start - timedelta(days=1), None, str(project_id), None, None, "m", None, Decimal("7")),
                (budget_start + timedelta(days=3), None, str(project_id), None, None, "m", None, Decimal("20")),
                (now, None, str(project_id), None, None, "m", None, Decimal("5")),
            ],
        )
        bounds = {period: period_bounds(period, now) for period in ("daily", "weekly", "monthly")}
        sql, parameters = _managed_spend_statement(
            bounds, schema, (), {project_id: budget_start}, [],
        )
        assert budget_start.isoformat() not in sql
        spend, totals, *_ = await asyncio.to_thread(
            _query_managed_spend, PostgreSQLWarehouseDataAPI(url), workgroup="local",
            database="local", secret_arn="local", sql=sql, parameters=parameters,
            project_ids=[project_id], key_ids=[],
        )
        assert totals == {str(project_id): Decimal("25")}
        assert spend["monthly"][str(project_id)] == Decimal("5")
    finally:
        await warehouse.execute(f"drop schema if exists {schema} cascade")
        await warehouse.close()


@pytest.mark.asyncio
async def test_fallback_served_by_a_billed_model_counts_toward_project_spend():
    now = datetime.now(UTC)
    project_id = uuid4()
    schema = f"warehouse_{uuid4().hex}"
    url = os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"]
    warehouse = await asyncpg.connect(url.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        await warehouse.execute(f"create schema {schema}")
        await warehouse.execute(f"""create table {schema}.usage_events (
            completed_at timestamptz not null, budget_id varchar(128), project_id varchar(128),
            managed_key_id varchar(128), identity_email varchar(1024),
            requested_model_key varchar(2048) not null, benchmark_name varchar(1024),
            cost_usd numeric(38, 12), served_model_key varchar(2048))""")
        await warehouse.executemany(
            f"insert into {schema}.usage_events values ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            [
                (now, None, str(project_id), None, None, "free", None, Decimal("100"), None),
                (now, None, str(project_id), None, None, "free", None, Decimal("3"), "billed"),
                (now, None, str(project_id), None, None, "billed", None, Decimal("40"), "free"),
                (now, None, str(project_id), None, None, "billed", None, Decimal("2"), None),
            ],
        )
        bounds = {period: period_bounds(period, now) for period in ("daily", "weekly", "monthly")}
        start = bounds["monthly"][0]
        sql, parameters = _managed_spend_statement(
            bounds, schema, ("free",), {project_id: start}, [],
        )
        spend, totals, *_ = await asyncio.to_thread(
            _query_managed_spend, PostgreSQLWarehouseDataAPI(url), workgroup="local",
            database="local", secret_arn="local", sql=sql, parameters=parameters,
            project_ids=[project_id], key_ids=[],
        )
        assert totals == {str(project_id): Decimal("5")}
        assert spend["daily"][str(project_id)] == Decimal("5")
    finally:
        await warehouse.execute(f"drop schema if exists {schema} cascade")
        await warehouse.close()


def _downgrade(connection, revision):
    config = Config(str(MIGRATIONS))
    script = ScriptDirectory.from_config(config)
    with EnvironmentContext(
        config, script, fn=lambda current, context: script._downgrade_revs(revision, current),
        destination_rev=revision,
    ) as context:
        context.configure(connection=connection, transactional_ddl=True)
        with context.begin_transaction():
            context.run_migrations()


@pytest.mark.asyncio
async def test_required_total_fills_each_missing_field_on_its_own():
    created = datetime(2026, 10, 15, 9, tzinfo=UTC)
    kept_start = datetime(2026, 9, 1, tzinfo=UTC)
    start_missing, total_missing = uuid4(), uuid4()
    async with _isolated_database() as engine:
        await _revision(engine, STAGE_ONE_HEAD)
        async with engine.begin() as connection:
            await _policy(connection, start_missing, "weekly", Decimal("50"), created)
            await _policy(connection, total_missing, "daily", Decimal("10"), created)
            await connection.execute(
                text("UPDATE managed_project_policies SET total_limit_usd = 777 WHERE project_id = :id"),
                {"id": start_missing},
            )
            await connection.execute(
                text("UPDATE managed_project_policies SET total_started_at = :start WHERE project_id = :id"),
                {"id": total_missing, "start": kept_start},
            )
        await _revision(engine, "head")
        policies = {p.project.project_id: p for p in await ManagedAuthStore(engine).list_project_policies()}
        assert (policies[start_missing].total_limit_usd, policies[start_missing].total_started_at) == (
            Decimal("777"), period_bounds("weekly", created)[0],
        )
        assert (policies[total_missing].total_limit_usd, policies[total_missing].total_started_at) == (
            Decimal("10"), kept_start,
        )
        async with engine.connect() as connection:
            with pytest.raises(RuntimeError, match="Downgrade requires a period and base"):
                await connection.run_sync(_downgrade, STAGE_ONE_HEAD)


@pytest.mark.asyncio
async def test_spend_statement_runs_as_one_union_with_inline_ids_and_bound_starts():
    now = datetime.now(UTC)
    project_id, key_id = uuid4(), uuid4()
    schema = f"warehouse_{uuid4().hex}"
    url = os.environ["MODEL_GATEWAY_AUTH_DATABASE_URL"]
    warehouse = await asyncpg.connect(url.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        await warehouse.execute(f"create schema {schema}")
        await warehouse.execute(f"""create table {schema}.usage_events (
            completed_at timestamptz not null, budget_id varchar(128), project_id varchar(128),
            managed_key_id varchar(128), identity_email varchar(1024),
            requested_model_key varchar(2048) not null, benchmark_name varchar(1024),
            cost_usd numeric(38, 12), served_model_key varchar(2048))""")
        await warehouse.executemany(
            f"insert into {schema}.usage_events values ($1, $2, $3, $4, $5, $6, $7, $8)",
            [
                (now, None, str(project_id), str(key_id), "a@example.com", "m", "bench", Decimal("2")),
                (now, None, str(project_id), str(key_id), "a@example.com", "o'brien", None, Decimal("9")),
                (now, None, str(project_id), None, None, "m", None, None),
            ],
        )
        bounds = {period: period_bounds(period, now) for period in ("daily", "weekly", "monthly")}
        sql, parameters = _managed_spend_statement(
            bounds, schema, ("o'brien",), {project_id: bounds["monthly"][0]}, [key_id],
        )
        assert sql.startswith("WITH budget_periods AS") and str(project_id) in sql
        assert bounds["monthly"][0].isoformat() in [item["value"] for item in parameters]
        spend, totals, contributors, models, benchmarks = await asyncio.to_thread(
            _query_managed_spend, PostgreSQLWarehouseDataAPI(url), workgroup="local",
            database="local", secret_arn="local", sql=sql, parameters=parameters,
            project_ids=[project_id], key_ids=[key_id],
        )
        for period in bounds:
            assert spend[period][str(project_id)] == Decimal("2")
            assert spend[period][f"key:{key_id}"] == Decimal("2")
            assert contributors[period][str(project_id)] == {"a@example.com": Decimal("2")}
            assert models[period][str(project_id)] == {"m": Decimal("2")}
            assert benchmarks[period][str(project_id)] == {"bench": Decimal("2")}
        assert totals == {str(project_id): Decimal("2")}
    finally:
        await warehouse.execute(f"drop schema if exists {schema} cascade")
        await warehouse.close()

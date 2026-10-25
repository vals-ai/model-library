"""Control views, temporary grants and permanent overrides for email budgets."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from model_gateway.budgets.bump_service import BumpConflict, BumpService
from model_gateway.budgets.override_service import (
    OverrideConflict,
    OverrideMetadataTooLong,
    OverrideService,
)
from model_gateway.budgets.bumps import (
    BumpDocument,
    BumpSnapshot,
    BudgetActor,
    LegacyBumpSnapshot,
    active_grants,
    effective_limit,
    grant_amount,
)
from model_gateway.budgets.schema import (
    BudgetPeriod,
    BudgetPolicy,
    SpendSnapshot,
    period_bounds,
)
from model_gateway.budgets.service import BudgetService
from model_gateway.runtime_config import (
    runtime_admin_api_key_names,
    runtime_budget_policy,
)
from model_gateway.usage_ledger.schema import canonical_email_value


class BumpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str
    expected_revision: StrictInt = Field(ge=0)
    period: BudgetPeriod
    period_start: datetime
    period_end: datetime
    actor: BudgetActor
    reason: str

    @field_validator("reason")
    @classmethod
    def nonblank_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A reason is required for every allowance change")
        return value.strip()


class SetBudgetLimitRequest(BumpRequest):
    limit_usd: str = Field(pattern=r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
    expected_base_limit_usd: str = Field(pattern=r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")


class OverrideWriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str
    expected_configuration_version: StrictInt = Field(ge=1)
    expected_latest_version: StrictInt = Field(ge=1)
    expected_deployment_number: StrictInt = Field(ge=0)
    actor: BudgetActor
    reason: str

    @field_validator("reason")
    @classmethod
    def nonblank_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A reason is required for every override change")
        return value.strip()


class SetOverrideRequest(OverrideWriteRequest):
    base_limit_usd: str = Field(pattern=r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"code": code, "message": message}
    )


def _admin_authorized(request: Request) -> bool:
    key_name = request.state.gateway_api_key_name
    unique_names = cast(frozenset[str], request.app.state.unique_api_key_names)
    return key_name in runtime_admin_api_key_names() and key_name in unique_names


def _budget_view(
    policy: BudgetPolicy,
    snapshot: SpendSnapshot | None,
    bumps: BumpSnapshot | LegacyBumpSnapshot,
    now: datetime,
    stage: str,
) -> dict[str, object]:
    start, end, grants = active_grants(bumps, policy, now)
    period = None if snapshot is None else snapshot.periods[policy.period]
    available = period is not None and period.start == start and period.end == end
    spending: dict[str, Decimal] = (
        period.spend_usd if available and period is not None else {}
    )
    emails = sorted(spending.keys() | policy.email_overrides.keys() | grants.keys())
    rows: list[dict[str, object]] = []
    for email in emails:
        override = policy.email_overrides.get(email)
        base_limit = policy.default_limit_usd if override is None else override
        effective = (
            effective_limit(base_limit, grant_amount(grants[email]))
            if email in grants
            else base_limit
        )
        source = "default" if override is None else "override"
        spent = spending.get(email, Decimal(0)) if available else None
        blocked = spent is not None and spent >= effective
        rows.append(
            {
                "email": email,
                "spent_usd": None if spent is None else format(spent, "f"),
                "limit_usd": format(effective, "f"),
                "limit_source": source,
                "blocked": blocked,
            }
        )
    return {
        "stage": stage,
        "period": policy.period,
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "default_limit_usd": format(policy.default_limit_usd, "f"),
        "email_overrides": {
            email: format(limit, "f") for email, limit in policy.email_overrides.items()
        },
        "spend_available": available,
        "snapshot_revision": None if snapshot is None else snapshot.revision,
        "snapshot_generated_at": None
        if snapshot is None
        else snapshot.generated_at.isoformat(),
        "data_through": None if snapshot is None else snapshot.data_through.isoformat(),
        "emails": rows,
    }


def register_budget_routes(app: FastAPI) -> None:
    @app.get("/admin/budgets", response_model=None)
    async def get_budgets(request: Request) -> dict[str, object] | JSONResponse:
        if not _admin_authorized(request):
            return _error_response(
                403, "forbidden", "This Gateway key cannot view budgets"
            )
        policy = runtime_budget_policy()
        if policy is None:
            return _error_response(
                503, "budget_config_unavailable", "This Gateway serves no budget policy"
            )
        service = cast(BudgetService, request.app.state.budget_service)
        try:
            snapshot = await service.read_shared_snapshot()
        except Exception:
            snapshot = service.snapshot
        bump_service = cast(BumpService, request.app.state.bump_service)
        bumps = bump_service.snapshot
        if bumps is None:
            return _error_response(
                503, "budget_bumps_unavailable", "Budget bump state is unavailable"
            )
        return _budget_view(policy, snapshot, bumps, datetime.now(UTC), service.stage)

    @app.get("/admin/budgets/bumps", response_model=None)
    async def get_bumps(request: Request) -> dict[str, object] | JSONResponse:
        if not _admin_authorized(request):
            return _error_response(
                403, "forbidden", "This Gateway key cannot view budgets"
            )
        policy = runtime_budget_policy()
        if policy is None:
            return _error_response(
                503, "budget_config_unavailable", "This Gateway serves no budget policy"
            )
        service = cast(BumpService, request.app.state.bump_service)
        try:
            snapshot = await service.read_control()
        except Exception:
            snapshot = service.snapshot
        if snapshot is None:
            return _error_response(
                503, "budget_bumps_unavailable", "Budget bump state is unavailable"
            )
        spend_service = cast(BudgetService, request.app.state.budget_service)
        try:
            spend = await spend_service.read_shared_snapshot()
        except Exception:
            spend = spend_service.snapshot
        now = datetime.now(UTC)
        start, end, grants = active_grants(snapshot, policy, now)
        rows: list[dict[str, str]] = []
        for email, grant in sorted(grants.items()):
            base = policy.email_overrides.get(email, policy.default_limit_usd)
            rows.append(
                {
                    "email": email,
                    "base_limit_usd": format(base, "f"),
                    "bump_usd": format(grant_amount(grant), "f"),
                    "limit_usd": format(
                        effective_limit(base, grant_amount(grant)), "f"
                    ),
                }
            )
        return {
            "stage": service.stage,
            "period": policy.period,
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "revision": snapshot.revision,
            "emails": rows,
            "budget": _budget_view(policy, spend, snapshot, now, service.stage),
        }

    @app.get("/admin/budgets/bumps/history", response_model=None)
    async def get_bump_history(
        request: Request,
        email: str,
        period: BudgetPeriod,
        period_start: datetime,
        period_end: datetime,
    ) -> dict[str, object] | JSONResponse:
        if not _admin_authorized(request):
            return _error_response(
                403, "forbidden", "This Gateway key cannot view budgets"
            )
        canonical = canonical_email_value(email)
        if canonical is None:
            return _error_response(400, "invalid_email", "A valid email is required")
        if (
            period_start.tzinfo is None
            or period_start.utcoffset() is None
            or period_end.tzinfo is None
            or period_end.utcoffset() is None
        ):
            return _error_response(
                400, "invalid_period", "Exact UTC calendar period bounds are required"
            )
        if period_bounds(period, period_start) != (period_start, period_end):
            return _error_response(
                400, "invalid_period", "Exact UTC calendar period bounds are required"
            )
        service = cast(BumpService, request.app.state.bump_service)
        try:
            document = await service.read_history()
        except Exception:
            return _error_response(
                503, "budget_bumps_unavailable", "Budget bump history is unavailable"
            )
        events = (
            [
                event.model_dump(mode="json")
                for event in document.events
                if event.email == canonical
                and event.period == period
                and event.period_start == period_start
                and event.period_end == period_end
            ]
            if isinstance(document, BumpDocument)
            else []
        )
        return {
            "stage": service.stage,
            "email": canonical,
            "period": period,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "revision": document.revision,
            "history_started_at": (
                document.history_started_at.isoformat()
                if isinstance(document, BumpDocument)
                else None
            ),
            "events": events,
        }

    async def change_bump(
        request: Request, body: BumpRequest, method: str
    ) -> dict[str, object] | JSONResponse:
        if not _admin_authorized(request):
            return _error_response(
                403, "forbidden", "This Gateway key cannot change budgets"
            )
        policy = runtime_budget_policy()
        if policy is None:
            return _error_response(
                503, "budget_config_unavailable", "This Gateway serves no budget policy"
            )
        email = canonical_email_value(body.email)
        if email is None:
            return _error_response(400, "invalid_email", "A valid email is required")
        start, end = period_bounds(policy.period, datetime.now(UTC))
        if (body.period, body.period_start, body.period_end) != (
            policy.period,
            start,
            end,
        ):
            return _error_response(
                409,
                "budget_bump_conflict",
                "Budget period changed; refresh and try again",
            )
        base = policy.email_overrides.get(email, policy.default_limit_usd)
        if isinstance(body, SetBudgetLimitRequest):
            if Decimal(body.expected_base_limit_usd) != base:
                return _error_response(
                    409,
                    "budget_bump_conflict",
                    "Budget base limit changed; refresh and try again",
                )
            limit = Decimal(body.limit_usd)
            if limit < base:
                return _error_response(
                    400,
                    "invalid_limit",
                    "Temporary limit cannot be below the base limit",
                )
        else:
            limit = None
        service = cast(BumpService, request.app.state.bump_service)
        try:
            snapshot, amount = await service.change(
                email,
                body.expected_revision,
                policy.period,
                start,
                end,
                base,
                body.actor,
                body.reason,
                limit,
                remove=method == "DELETE",
            )
        except BumpConflict:
            return _error_response(
                409,
                "budget_bump_conflict",
                "Budget bump changed; refresh and try again",
            )
        return {
            "email": email,
            "revision": snapshot.revision,
            "period": policy.period,
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "base_limit_usd": format(base, "f"),
            "bump_usd": format(amount, "f"),
            "limit_usd": format(effective_limit(base, amount), "f"),
        }

    @app.post("/admin/budgets/bumps", response_model=None)
    async def post_bump(
        request: Request, body: BumpRequest
    ) -> dict[str, object] | JSONResponse:
        return await change_bump(request, body, "POST")

    @app.put("/admin/budgets/bumps", response_model=None)
    async def put_bump(
        request: Request, body: SetBudgetLimitRequest
    ) -> dict[str, object] | JSONResponse:
        return await change_bump(request, body, "PUT")

    @app.delete("/admin/budgets/bumps", response_model=None)
    async def delete_bump(
        request: Request, body: BumpRequest
    ) -> dict[str, object] | JSONResponse:
        return await change_bump(request, body, "DELETE")

    @app.get("/admin/budgets/overrides", response_model=None)
    async def get_override(
        request: Request, email: str
    ) -> dict[str, object] | JSONResponse:
        if not _admin_authorized(request):
            return _error_response(
                403, "forbidden", "This Gateway key cannot view budgets"
            )
        canonical = canonical_email_value(email)
        if canonical is None:
            return _error_response(400, "invalid_email", "A valid email is required")
        service = cast(OverrideService | None, request.app.state.override_service)
        if service is None:
            return _error_response(
                503,
                "budget_config_unavailable",
                "AppConfig budget management is unavailable",
            )
        try:
            return await service.read(canonical)
        except OverrideConflict:
            return _error_response(
                409,
                "budget_override_conflict",
                "AppConfig Agent policy and hosted version differ; refresh and try again",
            )

    @app.get("/admin/budgets/overrides/history", response_model=None)
    async def get_override_history(
        request: Request, email: str, start: datetime, end: datetime
    ) -> dict[str, object] | JSONResponse:
        if not _admin_authorized(request):
            return _error_response(
                403, "forbidden", "This Gateway key cannot view budgets"
            )
        canonical = canonical_email_value(email)
        if canonical is None:
            return _error_response(400, "invalid_email", "A valid email is required")
        if (
            start.utcoffset() != timedelta(0)
            or end.utcoffset() != timedelta(0)
            or start >= end
        ):
            return _error_response(
                400, "invalid_range", "A half-open UTC interval is required"
            )
        service = cast(OverrideService | None, request.app.state.override_service)
        if service is None:
            return _error_response(
                503,
                "budget_config_unavailable",
                "AppConfig budget management is unavailable",
            )
        return await service.history(canonical, start, end)

    async def change_override(
        request: Request, body: OverrideWriteRequest, action: str
    ) -> dict[str, object] | JSONResponse:
        if not _admin_authorized(request):
            return _error_response(
                403, "forbidden", "This Gateway key cannot change budgets"
            )
        canonical = canonical_email_value(body.email)
        if canonical is None:
            return _error_response(400, "invalid_email", "A valid email is required")
        service = cast(OverrideService | None, request.app.state.override_service)
        if service is None:
            return _error_response(
                503,
                "budget_config_unavailable",
                "AppConfig budget management is unavailable",
            )
        amount = (
            Decimal(body.base_limit_usd)
            if isinstance(body, SetOverrideRequest)
            else None
        )
        try:
            return await service.change(
                email=canonical,
                expected_configuration_version=body.expected_configuration_version,
                expected_latest_version=body.expected_latest_version,
                expected_deployment_number=body.expected_deployment_number,
                actor=body.actor,
                reason=body.reason,
                amount=amount,
                action=action,
            )
        except OverrideConflict:
            return _error_response(
                409,
                "budget_override_conflict",
                "Budget policy or deployment changed; refresh and try again",
            )
        except OverrideMetadataTooLong:
            return _error_response(
                400,
                "override_reason_too_long",
                "Reason exceeds AppConfig record capacity",
            )

    @app.put("/admin/budgets/overrides", response_model=None)
    async def put_override(
        request: Request, body: SetOverrideRequest
    ) -> dict[str, object] | JSONResponse:
        return await change_override(request, body, "set")

    @app.delete("/admin/budgets/overrides", response_model=None)
    async def delete_override(
        request: Request, body: OverrideWriteRequest
    ) -> dict[str, object] | JSONResponse:
        return await change_override(request, body, "remove")

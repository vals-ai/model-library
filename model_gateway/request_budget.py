"""Budget checks and usage attribution shared by every route that spends provider money."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import cast

from fastapi import Request

from model_gateway.budgets.service import BudgetDecision, BudgetService
from model_gateway.managed_auth.runtime import RuntimeAuthority
from model_gateway.managed_auth.types import PersonalOwner, ServiceOwner
from model_gateway.usage_ledger.attribution import ManagedUsageAttribution
from model_gateway.usage_ledger.store import (
    build_managed_usage_event,
    build_usage_event,
)

UsageEventBuilder = Callable[..., dict[str, object]]


@dataclass(frozen=True)
class BudgetRefusal:
    status: int
    code: str
    message: str


def usage_event_builder(request: Request) -> UsageEventBuilder:
    """Builds usage events with the managed key's budget attribution, when the caller has one."""
    authority = cast(RuntimeAuthority | None, request.state.managed_authority)
    if authority is None:
        return build_usage_event
    owner = authority.owner
    project = authority.project
    personal = owner if isinstance(owner, PersonalOwner) else None
    attribution = ManagedUsageAttribution(
        managed_key_id=str(authority.key_id),
        budget_id=None,
        owner_issuer=personal.identity.issuer if personal is not None else None,
        owner_subject=personal.identity.subject if personal is not None else None,
        service_id=str(owner.service_id) if isinstance(owner, ServiceOwner) else None,
        project_id=str(project.project_id),
        project_org_id=project.organization,
        project_environment=project.environment,
    )
    return partial(build_managed_usage_event, attribution=attribution)


async def budget_refusal(request: Request) -> BudgetRefusal | None:
    """Returns why a managed key cannot spend under its project or key limits."""
    service = cast(BudgetService, request.app.state.budget_service)
    now = datetime.now(UTC)
    authority = cast(RuntimeAuthority | None, request.state.managed_authority)
    if authority is None:
        return None
    if authority.project_limit is None and authority.key_limit is None:
        return None
    decision = await service.managed_decision(authority, now)
    if decision == BudgetDecision.ALLOWED:
        return None
    if decision == BudgetDecision.KEY_EXHAUSTED:
        return BudgetRefusal(402, "budget_exceeded", "Managed key budget exceeded")
    if decision == BudgetDecision.EXHAUSTED:
        project_limit = authority.project_limit
        if project_limit is not None and project_limit.total_limit_usd is not None:
            return BudgetRefusal(
                402,
                "budget_exceeded",
                f"Project total budget of ${project_limit.total_limit_usd} is spent",
            )
        return BudgetRefusal(402, "budget_exceeded", "Managed budget exceeded")
    return BudgetRefusal(503, "budget_unavailable", "Managed budget spend unavailable")

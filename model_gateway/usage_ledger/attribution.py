"""Immutable managed-key project attribution captured at query admission."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ManagedUsageAttribution:
    managed_key_id: str
    budget_id: str | None
    owner_issuer: str | None
    owner_subject: str | None
    service_id: str | None
    project_id: str
    project_org_id: str
    project_environment: str

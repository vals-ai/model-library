"""Versioned AppConfig budget overrides and hosted-version audit records."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, TypedDict, cast

import boto3
from botocore.exceptions import ClientError

from model_gateway.budgets.bumps import BudgetActor
from model_gateway.budgets.schema import BudgetPolicy
from model_gateway.runtime_config import AppConfigAgentClient


class OverrideAuditMetadata(TypedDict):
    kind: str
    changed_at: str
    actor: dict[str, str | None]
    reason: str
    email: str
    action: str
    old_base_limit_usd: str
    new_base_limit_usd: str


class OverrideDeploymentHistory(TypedDict):
    number: int
    state: str


class OverrideHistoryEvent(TypedDict):
    configuration_version: int
    changed_at: str
    actor: dict[str, str | None]
    reason: str
    action: str
    old_base_limit_usd: str
    new_base_limit_usd: str
    deployment: OverrideDeploymentHistory | None


class OverrideConflict(Exception):
    """The observed policy or AppConfig locking token changed."""


class OverrideMetadataTooLong(Exception):
    """The full audit description exceeds AppConfig's record capacity."""


class OverrideService:
    def __init__(
        self,
        *,
        stage: str,
        application: str,
        environment: str,
        profile: str,
        strategy: str,
        agent: AppConfigAgentClient,
        client: Any,
    ) -> None:
        self.stage = stage
        self.application = application
        self.environment = environment
        self.profile = profile
        self.strategy = strategy
        self.agent = agent
        self.client = client

    async def close(self) -> None:
        await self.agent.aclose()

    def _hosted(self) -> list[dict[str, Any]]:
        versions: list[dict[str, Any]] = []
        token = None
        while True:
            params = {
                "ApplicationId": self.application,
                "ConfigurationProfileId": self.profile,
            }
            if token is not None:
                params["NextToken"] = token
            response = self.client.list_hosted_configuration_versions(**params)
            versions.extend(response["Items"])
            token = response.get("NextToken")
            if token is None:
                return versions

    def _deployments(self) -> list[dict[str, Any]]:
        deployments: list[dict[str, Any]] = []
        token = None
        while True:
            params = {
                "ApplicationId": self.application,
                "EnvironmentId": self.environment,
            }
            if token is not None:
                params["NextToken"] = token
            response = self.client.list_deployments(**params)
            deployments.extend(response["Items"])
            token = response.get("NextToken")
            if token is None:
                return deployments

    def _version_content(self, version: int) -> bytes:
        response = self.client.get_hosted_configuration_version(
            ApplicationId=self.application,
            ConfigurationProfileId=self.profile,
            VersionNumber=version,
        )
        return response["Content"].read()

    async def _state(
        self,
    ) -> tuple[
        BudgetPolicy, int, int, int, dict[str, Any] | None, list[dict[str, Any]]
    ]:
        versions, deployments, served = await asyncio.gather(
            asyncio.to_thread(self._hosted),
            asyncio.to_thread(self._deployments),
            self.agent.fetch(self.profile),
        )
        configuration_version = int(served.version)
        authoritative_content = await asyncio.to_thread(
            self._version_content, configuration_version
        )
        if served.content != authoritative_content:
            raise OverrideConflict
        policy = BudgetPolicy.model_validate_json(served.content)
        latest_version = max(version["VersionNumber"] for version in versions)
        latest_deployment_number = max(
            (deployment["DeploymentNumber"] for deployment in deployments), default=0
        )
        budget_deployments = [
            deployment
            for deployment in deployments
            if deployment["ConfigurationProfileId"] == self.profile
        ]
        latest_budget = max(
            budget_deployments,
            key=lambda deployment: deployment["DeploymentNumber"],
            default=None,
        )
        return (
            policy,
            configuration_version,
            latest_version,
            latest_deployment_number,
            latest_budget,
            budget_deployments,
        )

    async def read(self, email: str) -> dict[str, Any]:
        (
            policy,
            version,
            latest,
            deployment_number,
            budget_deployment,
            _,
        ) = await self._state()
        override = policy.email_overrides.get(email)
        return {
            "stage": self.stage,
            "email": email,
            "default_limit_usd": format(policy.default_limit_usd, "f"),
            "base_limit_usd": format(
                policy.default_limit_usd if override is None else override, "f"
            ),
            "override_limit_usd": None if override is None else format(override, "f"),
            "configuration_version": version,
            "latest_version": latest,
            "latest_deployment_number": deployment_number,
            "deployment": self._deployment_view(budget_deployment),
        }

    @staticmethod
    def _deployment_view(deployment: dict[str, Any] | None) -> dict[str, Any] | None:
        if deployment is None:
            return None
        return {
            "number": deployment["DeploymentNumber"],
            "configuration_version": int(deployment["ConfigurationVersion"]),
            "state": deployment["State"],
        }

    async def change(
        self,
        *,
        email: str,
        expected_configuration_version: int,
        expected_latest_version: int,
        expected_deployment_number: int,
        actor: BudgetActor,
        reason: str,
        amount: Decimal | None,
        action: str,
    ) -> dict[str, Any]:
        (
            policy,
            version,
            latest,
            deployment_number,
            budget_deployment,
            deployments,
        ) = await self._state()
        if (
            version != expected_configuration_version
            or latest != expected_latest_version
            or deployment_number != expected_deployment_number
            or budget_deployment is None
            or budget_deployment["State"] in {"DEPLOYING", "ROLLING_BACK"}
        ):
            raise OverrideConflict
        completed = max(
            (
                deployment
                for deployment in deployments
                if deployment["State"] == "COMPLETE"
            ),
            key=lambda deployment: deployment["DeploymentNumber"],
            default=None,
        )
        if completed is None or int(completed["ConfigurationVersion"]) != version:
            raise OverrideConflict

        old = policy.email_overrides.get(email, policy.default_limit_usd)
        updated = dict(policy.email_overrides)
        if action == "remove":
            updated.pop(email, None)
            new = policy.default_limit_usd
        else:
            assert amount is not None
            updated[email] = amount
            new = amount
        description = json.dumps(
            {
                "kind": "budget_override",
                "changed_at": datetime.now(UTC).isoformat(),
                "actor": actor.model_dump(),
                "reason": reason,
                "email": email,
                "action": action,
                "old_base_limit_usd": format(old, "f"),
                "new_base_limit_usd": format(new, "f"),
            },
            ensure_ascii=True,
            separators=(",", ":"),
        )
        if len(description) > 1024:
            raise OverrideMetadataTooLong
        content = json.dumps(
            {
                "period": policy.period,
                "default_limit_usd": format(policy.default_limit_usd, "f"),
                "email_overrides": {
                    key: format(value, "f") for key, value in updated.items()
                },
            },
            separators=(",", ":"),
        ).encode()
        return await asyncio.to_thread(
            self._publish, content, description, latest, deployment_number, email
        )

    def _publish(
        self,
        content: bytes,
        description: str,
        latest_version: int,
        deployment_number: int,
        email: str,
    ) -> dict[str, Any]:
        try:
            version = self.client.create_hosted_configuration_version(
                ApplicationId=self.application,
                ConfigurationProfileId=self.profile,
                Content=content,
                ContentType="application/json",
                Description=description,
                LatestVersionNumber=latest_version,
            )["VersionNumber"]
            deployment = self.client.start_deployment(
                ApplicationId=self.application,
                EnvironmentId=self.environment,
                ConfigurationProfileId=self.profile,
                ConfigurationVersion=str(version),
                DeploymentStrategyId=self.strategy,
                LatestDeploymentNumber=deployment_number,
            )
        except ClientError as error:
            if error.response["Error"]["Code"] == "ConflictException":
                raise OverrideConflict from error
            raise
        return {
            "stage": self.stage,
            "email": email,
            "configuration_version": version,
            "deployment_number": deployment["DeploymentNumber"],
            "state": deployment["State"],
        }

    async def history(
        self, email: str, start: datetime, end: datetime
    ) -> dict[str, Any]:
        versions, deployments = await asyncio.gather(
            asyncio.to_thread(self._hosted), asyncio.to_thread(self._deployments)
        )
        deployments_by_version = {
            int(deployment["ConfigurationVersion"]): deployment
            for deployment in sorted(
                deployments, key=lambda item: item["DeploymentNumber"]
            )
            if deployment["ConfigurationProfileId"] == self.profile
        }
        events: list[OverrideHistoryEvent] = []
        for version in versions:
            description = version.get("Description", "")
            try:
                parsed = json.loads(description)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            if cast(dict[str, object], parsed).get("kind") != "budget_override":
                continue
            metadata = cast(OverrideAuditMetadata, parsed)
            changed_at = datetime.fromisoformat(metadata["changed_at"])
            if metadata["email"] != email or not start <= changed_at < end:
                continue
            deployment = deployments_by_version.get(version["VersionNumber"])
            events.append(
                {
                    "configuration_version": version["VersionNumber"],
                    "changed_at": changed_at.isoformat(),
                    "actor": metadata["actor"],
                    "reason": metadata["reason"],
                    "action": metadata["action"],
                    "old_base_limit_usd": metadata["old_base_limit_usd"],
                    "new_base_limit_usd": metadata["new_base_limit_usd"],
                    "deployment": None
                    if deployment is None
                    else {
                        "number": deployment["DeploymentNumber"],
                        "state": deployment["State"],
                    },
                }
            )
        events.sort(
            key=lambda event: (event["changed_at"], event["configuration_version"])
        )
        return {
            "stage": self.stage,
            "email": email,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "events": events,
        }


def override_service_from_env(stage: str) -> OverrideService | None:
    application = os.environ.get("GATEWAY_APPCONFIG_APPLICATION")
    profile = os.environ.get("GATEWAY_APPCONFIG_BUDGET_PROFILE_ID") or None
    if not application or not profile:
        return None
    environment = os.environ["GATEWAY_APPCONFIG_ENVIRONMENT"]
    return OverrideService(
        stage=stage,
        application=application,
        environment=environment,
        profile=profile,
        strategy=os.environ["GATEWAY_APPCONFIG_DEPLOYMENT_STRATEGY"],
        agent=AppConfigAgentClient(
            base_url=os.environ.get("GATEWAY_APPCONFIG_URL", "http://127.0.0.1:2772"),
            application=application,
            environment=environment,
            access_token=os.environ["GATEWAY_APPCONFIG_ACCESS_TOKEN"],
        ),
        client=cast(Callable[[str], Any], boto3.client)("appconfig"),
    )

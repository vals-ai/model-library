"""The key pool document: the one source of the Gateway's key pools.

Every Gateway task reads it from the `pools` AppConfig profile through the same
refresh as the model and credential profiles. `scripts/seed_key_pools.py`
publishes a stage's first document; the control API's PUT publishes each later one.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol, cast

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, ConfigDict, Field

from model_gateway.key_pools import (
    ModelRule,
    PoolConfig,
    internal_key_settings,
    key_setting_provider,
    provider_key_settings,
    provider_of,
)


class ModelPoolRule(BaseModel):
    """Sends each listed model id to `pool`."""

    model_config = ConfigDict(extra="forbid")

    pool: str
    models: list[str]


class KeyPoolReplacement(BaseModel):
    """A PUT body: it replaces the whole document, so every field is required."""

    model_config = ConfigDict(extra="forbid")

    pools: dict[str, list[str]]
    model_pools: list[ModelPoolRule]
    weights: dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    descriptions: dict[str, str]
    """Free text for operators, by pool or key setting name; carries no behaviour."""


class PublishedPoolDocument(KeyPoolReplacement):
    """The document as published, with the revision that orders it.

    Tasks order documents by `revision`, not by the Agent's `Configuration-Version`
    header, which can lag the body by one publish.
    """

    descriptions: dict[str, str] = Field(default_factory=dict[str, str])
    """A document published before descriptions has none."""
    revision: Annotated[int, Field(ge=0)]


@dataclass(frozen=True)
class PoolConfigSnapshot:
    """The published document a task is serving."""

    digest: str
    document: PublishedPoolDocument
    configuration_version: str
    """The Agent's label for this body, kept for logs only."""
    config: PoolConfig = field(init=False, repr=False, compare=False)
    """The document as a `PoolConfig`, built once when the snapshot is made."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", pool_config_from_document(self.document))


def pool_config_from_document(document: KeyPoolReplacement) -> PoolConfig:
    """The `PoolConfig` a published document serves; its rules are the only rules."""
    return PoolConfig(
        pools={name: tuple(keys) for name, keys in document.pools.items()},
        rules=tuple(
            ModelRule(rule.pool, models=frozenset(rule.models))
            for rule in document.model_pools
        ),
        weights=dict(document.weights),
    )


@dataclass(frozen=True)
class KeyCatalog:
    """The key settings an operator may place in pools, by name only."""

    providers: Mapping[str, str]
    internal: frozenset[str]
    pooled_providers: frozenset[str]

    def known_keys(self) -> dict[str, frozenset[str]]:
        by_provider: dict[str, set[str]] = {}
        for key, provider in self.providers.items():
            by_provider.setdefault(provider, set()).add(key)
        return {provider: frozenset(keys) for provider, keys in by_provider.items()}


def key_catalog(
    served_pools: Mapping[str, Sequence[str]],
    managed_fields: Iterable[str],
    get_setting: Callable[[str], object],
) -> KeyCatalog:
    """The key settings with a value on this stage that an operator may pool.

    Keys come from the served pools and the credential profile's fields, each
    named for its provider. Pooled providers are those whose default pool the
    served document holds, so a replacement cannot drop one.
    """
    providers = {
        key: provider_of(pool)
        for pool, keys in served_pools.items()
        for key in keys
        if get_setting(key)
    }
    key_settings = provider_key_settings()
    for setting in [*managed_fields, *key_settings.values()]:
        provider = key_setting_provider(setting, key_settings)
        if provider is not None and get_setting(setting):
            providers.setdefault(setting, provider)
    return KeyCatalog(
        providers=providers,
        internal=internal_key_settings(),
        pooled_providers=frozenset(
            provider_of(pool) for pool in served_pools if pool.endswith("/default")
        ),
    )


class PoolConfigDeploymentInProgressError(Exception):
    """AppConfig is still deploying a version to this environment."""


class PoolConfigPublisher(Protocol):
    async def publish(self, content: bytes) -> str:
        """Publish `content` as a new version, start its deployment, return the version."""
        ...


class _AppConfigClient(Protocol):
    def create_hosted_configuration_version(
        self, **kwargs: Any
    ) -> Mapping[str, Any]: ...

    def start_deployment(self, **kwargs: Any) -> Mapping[str, Any]: ...


class AppConfigPoolConfigPublisher:
    """Publishes to the `pools` profile the way the release pipeline publishes models."""

    def __init__(
        self,
        *,
        application_id: str,
        environment_id: str,
        profile_id: str,
        deployment_strategy_id: str,
        client: _AppConfigClient | None = None,
    ) -> None:
        self._application_id = application_id
        self._environment_id = environment_id
        self._profile_id = profile_id
        self._deployment_strategy_id = deployment_strategy_id
        self._client = client

    async def publish(self, content: bytes) -> str:
        return await asyncio.to_thread(self._publish, content)

    def _publish(self, content: bytes) -> str:
        client = self._appconfig()
        response = client.create_hosted_configuration_version(
            ApplicationId=self._application_id,
            ConfigurationProfileId=self._profile_id,
            Content=content,
            ContentType="application/json",
            Description="Key pools from the Gateway control API",
        )
        version = response.get("VersionNumber")
        if not isinstance(version, int):
            raise ValueError("AppConfig did not return VersionNumber")
        try:
            client.start_deployment(
                ApplicationId=self._application_id,
                EnvironmentId=self._environment_id,
                ConfigurationProfileId=self._profile_id,
                ConfigurationVersion=str(version),
                DeploymentStrategyId=self._deployment_strategy_id,
                Description="Publish key pools from the Gateway control API",
            )
        except ClientError as error:
            code = cast(Mapping[str, Any], error.response).get("Error", {}).get("Code")
            if code == "ConflictException":
                raise PoolConfigDeploymentInProgressError from error
            raise
        return str(version)

    def _appconfig(self) -> _AppConfigClient:
        if self._client is None:
            self._client = cast(_AppConfigClient, cast(Any, boto3).client("appconfig"))
        return self._client


def pool_config_publisher_from_env() -> AppConfigPoolConfigPublisher | None:
    """The AppConfig publisher, or None when AppConfig is off, as for runtime refresh."""
    application = os.environ.get("GATEWAY_APPCONFIG_APPLICATION")
    if not application:
        return None

    required = {
        name: os.environ.get(name, "")
        for name in (
            "GATEWAY_APPCONFIG_ENVIRONMENT",
            "GATEWAY_APPCONFIG_POOL_PROFILE",
            "GATEWAY_APPCONFIG_DEPLOYMENT_STRATEGY",
        )
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            "AppConfig pool publishing is missing required settings: "
            + ", ".join(sorted(missing))
        )
    return AppConfigPoolConfigPublisher(
        application_id=application,
        environment_id=required["GATEWAY_APPCONFIG_ENVIRONMENT"],
        profile_id=required["GATEWAY_APPCONFIG_POOL_PROFILE"],
        deployment_strategy_id=required["GATEWAY_APPCONFIG_DEPLOYMENT_STRATEGY"],
    )

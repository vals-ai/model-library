"""Versioned Gateway runtime snapshots delivered through AWS AppConfig Agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections.abc import Generator, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from types import MappingProxyType
from typing import Literal, Self, cast
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

from model_gateway.budgets.schema import BudgetPolicy
from model_gateway.key_pool_config import PoolConfigSnapshot, PublishedPoolDocument
from model_gateway.metrics import record_runtime_config_refresh
from model_gateway.observability import worker_id
from model_library.register_models import (
    Registries,
    get_registries,
    parse_registry_documents,
    registry_snapshot,
    split_registry_entries,
)
from model_library.settings import managed_settings_snapshot

logger = logging.getLogger("model_proxy_server.runtime_config")
SourceName = Literal["models", "credentials", "pools", "access", "budgets"]

_NO_DEPLOYED_VERSION_STATUSES = frozenset({400, 404})


class _AccessDocument(BaseModel):
    """The `access` profile's document: MODEL_GATEWAY_API_KEYS names, never values."""

    model_config = ConfigDict(extra="forbid")

    admin_api_key_names: list[str]


# The Gateway's own settings; a credential snapshot must never override them.
GATEWAY_OWNED_FIELD_PREFIXES = ("GATEWAY_", "MODEL_GATEWAY_", "AWS_", "OTEL_")
GATEWAY_OWNED_FIELDS = frozenset({"REDIS_URL"})


class SnapshotRejectedError(ValueError):
    """A candidate is valid JSON but cannot replace the current snapshot."""


class VersionedEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    revision: int
    originated_at: datetime

    @field_validator("revision")
    @classmethod
    def non_negative_revision(cls, revision: int) -> int:
        if revision < 0:
            raise ValueError("revision must be non-negative")
        return revision

    @field_validator("originated_at")
    @classmethod
    def timezone_required(cls, originated_at: datetime) -> datetime:
        if originated_at.tzinfo is None:
            raise ValueError("originated_at must include a timezone")
        return originated_at.astimezone(UTC)


class ModelEnvelope(VersionedEnvelope):
    documents: dict[str, str]

    @field_validator("documents")
    @classmethod
    def documents_required(cls, documents: dict[str, str]) -> dict[str, str]:
        if not documents:
            raise ValueError("documents must not be empty")
        if any(not name or not content for name, content in documents.items()):
            raise ValueError("document names and content must not be empty")
        return documents


@dataclass(frozen=True)
class ModelSnapshot:
    revision: int
    originated_at: datetime
    digest: str
    registries: Registries


@dataclass(frozen=True)
class CredentialSnapshot:
    version: str
    generation: int
    digest: str
    managed_fields: frozenset[str]
    values: Mapping[str, SecretStr]


class _RevealedCredentials(Mapping[str, str]):
    """Reveal one managed credential only when a request reads it."""

    def __init__(self, values: Mapping[str, SecretStr]) -> None:
        self._values = values

    def __getitem__(self, key: str) -> str:
        return self._values[key].get_secret_value()

    def __contains__(self, key: object) -> bool:
        return key in self._values

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


@dataclass(frozen=True)
class RuntimeView:
    models: ModelSnapshot
    credentials: CredentialSnapshot | None
    pools: PoolConfigSnapshot | None
    budgets: BudgetPolicy | None = None
    admin_api_key_names: frozenset[str] = frozenset()
    """From the access document; control tasks only, empty until one is published."""

    @property
    def cache_namespace(self) -> str:
        credential_revision = (
            "bootstrap"
            if self.credentials is None
            else f"{self.credentials.version}:{self.credentials.generation}"
        )
        namespace = f"models={self.models.revision}:credentials={credential_revision}"
        # Pool changes rebuild rate-limit monitor probes for the new keys.
        return (
            namespace
            if self.pools is None
            else f"{namespace}:pools={self.pools.document.revision}"
        )


@dataclass(frozen=True)
class InstallResult:
    applied: frozenset[SourceName] = frozenset()
    unchanged: frozenset[SourceName] = frozenset()
    rejected: frozenset[SourceName] = frozenset()


_runtime_view_context: ContextVar[RuntimeView | None] = ContextVar(
    "gateway_runtime_view", default=None
)


def runtime_cache_namespace() -> str:
    view = _runtime_view_context.get()
    return "bootstrap" if view is None else view.cache_namespace


def runtime_pool_snapshot() -> PoolConfigSnapshot | None:
    """The published pool config this request runs with; None without AppConfig."""
    view = _runtime_view_context.get()
    return None if view is None else view.pools


def runtime_budget_policy() -> BudgetPolicy | None:
    view = _runtime_view_context.get()
    return None if view is None else view.budgets


def runtime_admin_api_key_names() -> frozenset[str]:
    """Gateway key names this request may use on /admin routes; empty without a document."""
    view = _runtime_view_context.get()
    return frozenset() if view is None else view.admin_api_key_names


def runtime_credential_fields() -> frozenset[str]:
    """The provider secret's field names this request runs with; empty without AppConfig."""
    view = _runtime_view_context.get()
    if view is None or view.credentials is None:
        return frozenset()
    return view.credentials.managed_fields


class RuntimeConfigState:
    def __init__(self, view: RuntimeView) -> None:
        self._view = view
        self._lock = Lock()

    @classmethod
    def bootstrap(cls) -> Self:
        return cls(
            RuntimeView(
                models=ModelSnapshot(
                    revision=0,
                    originated_at=datetime.fromtimestamp(0, tz=UTC),
                    digest="bootstrap",
                    registries=get_registries(),
                ),
                credentials=None,
                pools=None,
            )
        )

    @property
    def current(self) -> RuntimeView:
        with self._lock:
            return self._view

    @contextmanager
    def capture(self) -> Generator[RuntimeView]:
        view = self.current
        token = _runtime_view_context.set(view)
        try:
            with ExitStack() as stack:
                stack.enter_context(registry_snapshot(view.models.registries))
                if view.credentials is not None:
                    stack.enter_context(
                        managed_settings_snapshot(
                            _RevealedCredentials(view.credentials.values),
                            view.credentials.managed_fields,
                        )
                    )
                yield view
        finally:
            _runtime_view_context.reset(token)

    def install(
        self,
        models: ModelSnapshot | None = None,
        credentials: CredentialSnapshot | None = None,
        pools: PoolConfigSnapshot | None = None,
        admin_api_key_names: frozenset[str] | None = None,
        budgets: BudgetPolicy | None = None,
    ) -> InstallResult:
        """Install valid sources independently in one immutable runtime view."""
        with self._lock:
            rejected: set[SourceName] = set()
            if models is not None:
                try:
                    _validate_successor(
                        self._view.models.revision,
                        self._view.models.digest,
                        models.revision,
                        models.digest,
                    )
                except SnapshotRejectedError:
                    rejected.add("models")
            current_pools = self._view.pools
            if pools is not None and current_pools is not None:
                try:
                    _validate_successor(
                        current_pools.document.revision,
                        current_pools.digest,
                        pools.document.revision,
                        pools.digest,
                    )
                except SnapshotRejectedError:
                    rejected.add("pools")

            current_credentials = self._view.credentials
            if (
                credentials is not None
                and current_credentials is not None
                and credentials.version == current_credentials.version
                and credentials.digest != current_credentials.digest
            ):
                rejected.add("credentials")

            applied: set[SourceName] = set()
            unchanged: set[SourceName] = set()
            next_models = self._view.models
            next_credentials = current_credentials
            if models is not None and "models" not in rejected:
                if models.revision == self._view.models.revision:
                    unchanged.add("models")
                else:
                    applied.add("models")
                    next_models = models
            if credentials is not None and "credentials" not in rejected:
                if (
                    current_credentials is not None
                    and credentials.version == current_credentials.version
                ):
                    unchanged.add("credentials")
                else:
                    applied.add("credentials")
                    next_credentials = credentials
            next_pools = current_pools
            if pools is not None and "pools" not in rejected:
                if (
                    current_pools is not None
                    and pools.document.revision == current_pools.document.revision
                ):
                    unchanged.add("pools")
                else:
                    applied.add("pools")
                    next_pools = pools
            next_admin_names = self._view.admin_api_key_names
            if admin_api_key_names is not None:
                if admin_api_key_names == next_admin_names:
                    unchanged.add("access")
                else:
                    applied.add("access")
                    next_admin_names = admin_api_key_names
            next_budgets = self._view.budgets
            if budgets is not None:
                if budgets == next_budgets:
                    unchanged.add("budgets")
                else:
                    applied.add("budgets")
                    next_budgets = budgets

            if applied:
                self._view = RuntimeView(
                    models=next_models,
                    credentials=next_credentials,
                    pools=next_pools,
                    budgets=next_budgets,
                    admin_api_key_names=next_admin_names,
                )
            return InstallResult(
                applied=frozenset(applied),
                unchanged=frozenset(unchanged),
                rejected=frozenset(rejected),
            )


def _validate_successor(
    current_revision: int,
    current_digest: str,
    candidate_revision: int,
    candidate_digest: str,
) -> None:
    if candidate_revision < current_revision:
        raise SnapshotRejectedError("candidate revision is older than current")
    if candidate_revision == current_revision and candidate_digest != current_digest:
        raise SnapshotRejectedError("candidate changes an existing revision")


@dataclass(frozen=True)
class AgentConfiguration:
    content: bytes
    version: str


class AppConfigAgentClient:
    def __init__(
        self,
        *,
        base_url: str,
        application: str,
        environment: str,
        access_token: str | None,
        http_client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._application = application
        self._environment = environment
        self._access_token = access_token
        self._client = http_client or httpx.AsyncClient()
        self._timeout_seconds = timeout_seconds

    async def fetch(self, profile: str) -> AgentConfiguration:
        path = "/".join(
            quote(part, safe="")
            for part in (
                "applications",
                self._application,
                "environments",
                self._environment,
                "configurations",
                profile,
            )
        )
        headers = (
            {"Authorization": f"Bearer {self._access_token}"}
            if self._access_token
            else None
        )
        response = await self._client.get(
            f"{self._base_url}/{path}",
            headers=headers,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        version = response.headers.get("Configuration-Version")
        if not version:
            raise ValueError("AppConfig Agent response has no Configuration-Version")
        return AgentConfiguration(content=response.content, version=version)

    async def aclose(self) -> None:
        await self._client.aclose()


@dataclass(frozen=True)
class RefreshResult:
    rejected: frozenset[SourceName] = frozenset()
    failed: frozenset[SourceName] = frozenset()


class RuntimeConfigManager:
    def __init__(
        self,
        *,
        state: RuntimeConfigState,
        agent: AppConfigAgentClient,
        model_profile: str,
        credential_profile: str,
        pool_profile: str,
        access_profile: str | None = None,
        budget_profile: str | None = None,
        poll_interval_seconds: float = 5.0,
        source_timeout_seconds: float = 3.0,
    ) -> None:
        self.state = state
        self._agent = agent
        self._model_profile = model_profile
        self._credential_profile = credential_profile
        self._pool_profile = pool_profile
        self._access_profile = access_profile
        self._budget_profile = budget_profile
        self._poll_interval_seconds = poll_interval_seconds
        self._source_timeout_seconds = source_timeout_seconds
        self._refresh_lock = asyncio.Lock()

    async def refresh_once(self) -> RefreshResult:
        async with self._refresh_lock:
            return await self._refresh_once()

    async def _refresh_once(self) -> RefreshResult:
        failed: set[SourceName] = set()
        rejected: set[SourceName] = set()
        candidates: dict[
            SourceName,
            ModelSnapshot
            | CredentialSnapshot
            | PoolConfigSnapshot
            | frozenset[str]
            | BudgetPolicy,
        ] = {}

        async def read(source: SourceName, profile: str) -> None:
            try:
                delivered = await asyncio.wait_for(
                    self._agent.fetch(profile), timeout=self._source_timeout_seconds
                )
                if source == "models":
                    current_models = self.state.current.models
                    if (
                        hashlib.sha256(delivered.content).hexdigest()
                        == current_models.digest
                    ):
                        candidates[source] = current_models
                    else:
                        candidates[source] = await asyncio.to_thread(
                            _parse_model_snapshot, delivered.content
                        )
                elif source == "pools":
                    candidates[source] = _parse_pool_snapshot(
                        delivered.content,
                        version=delivered.version,
                        current=self.state.current.pools,
                    )
                elif source == "access":
                    candidates[source] = frozenset(
                        _AccessDocument.model_validate_json(
                            delivered.content
                        ).admin_api_key_names
                    )
                elif source == "budgets":
                    candidates[source] = BudgetPolicy.model_validate_json(
                        delivered.content
                    )
                else:
                    candidates[source] = _parse_credential_snapshot(
                        delivered.content,
                        version=delivered.version,
                        current=self.state.current.credentials,
                    )
            except SnapshotRejectedError:
                rejected.add(source)
            except Exception as error:
                if source == "access":
                    candidates[source] = frozenset()
                    if (
                        isinstance(error, httpx.HTTPStatusError)
                        and error.response.status_code in _NO_DEPLOYED_VERSION_STATUSES
                    ):
                        return
                if (
                    source == "budgets"
                    and isinstance(error, httpx.HTTPStatusError)
                    and error.response.status_code in _NO_DEPLOYED_VERSION_STATUSES
                ):
                    return
                failed.add(source)
                record_runtime_config_refresh(source=source, outcome="failed")
                logger.warning(
                    "Runtime configuration refresh failed source=%s error_type=%s",
                    source,
                    type(error).__name__,
                )

        reads = [
            read("models", self._model_profile),
            read("credentials", self._credential_profile),
            read("pools", self._pool_profile),
        ]
        if self._access_profile is not None:
            reads.append(read("access", self._access_profile))
        if self._budget_profile is not None:
            reads.append(read("budgets", self._budget_profile))
        await asyncio.gather(*reads)
        models = cast(ModelSnapshot | None, candidates.get("models"))
        credentials = cast(CredentialSnapshot | None, candidates.get("credentials"))
        pools = cast(PoolConfigSnapshot | None, candidates.get("pools"))
        admin_names = cast(frozenset[str] | None, candidates.get("access"))
        budgets = cast(BudgetPolicy | None, candidates.get("budgets"))
        installed = self.state.install(models, credentials, pools, admin_names, budgets)
        rejected.update(installed.rejected)
        for source in rejected:
            record_runtime_config_refresh(source=source, outcome="rejected")
            logger.warning("Rejected runtime configuration snapshot source=%s", source)

        for source, candidate in candidates.items():
            if source not in installed.applied:
                if source in installed.unchanged:
                    record_runtime_config_refresh(
                        source=source,
                        outcome="unchanged",
                    )
                continue
            if isinstance(candidate, ModelSnapshot):
                source_propagation_seconds = max(
                    0.0,
                    (datetime.now(UTC) - candidate.originated_at).total_seconds(),
                )
                logger.info(
                    "Applied runtime configuration snapshot source=%s revision=%s "
                    "worker_id=%s propagation_seconds=%.3f",
                    source,
                    candidate.revision,
                    worker_id(),
                    source_propagation_seconds,
                )
                record_runtime_config_refresh(
                    source=source,
                    outcome="applied",
                    propagation_seconds=source_propagation_seconds,
                )
            elif isinstance(candidate, PoolConfigSnapshot):
                logger.info(
                    "Applied runtime configuration snapshot source=pools "
                    "revision=%s configuration_version=%s worker_id=%s",
                    candidate.document.revision,
                    candidate.configuration_version,
                    worker_id(),
                )
                record_runtime_config_refresh(source=source, outcome="applied")
            elif isinstance(candidate, frozenset):
                logger.info(
                    "Applied runtime configuration snapshot source=access "
                    "admin_api_key_names=%s worker_id=%s",
                    ",".join(sorted(candidate)),
                    worker_id(),
                )
                record_runtime_config_refresh(source=source, outcome="applied")
            elif isinstance(candidate, BudgetPolicy):
                logger.info(
                    "Applied runtime configuration snapshot source=budgets worker_id=%s",
                    worker_id(),
                )
                record_runtime_config_refresh(source=source, outcome="applied")
            else:
                logger.info(
                    "Applied runtime configuration snapshot source=credentials "
                    "version=%s generation=%s worker_id=%s observed_at=%s",
                    candidate.version,
                    candidate.generation,
                    worker_id(),
                    datetime.now(UTC).isoformat(),
                )
                record_runtime_config_refresh(
                    source=source,
                    outcome="applied",
                )

        return RefreshResult(
            rejected=frozenset(rejected),
            failed=frozenset(failed),
        )

    async def run(self, stop: asyncio.Event) -> None:
        # Startup awaits refresh_once() before starting this loop, so wait one
        # interval before the first background refresh.
        started = time.monotonic()
        while not stop.is_set():
            remaining = max(
                0.0,
                self._poll_interval_seconds - (time.monotonic() - started),
            )
            try:
                await asyncio.wait_for(stop.wait(), timeout=remaining)
                return
            except TimeoutError:
                pass
            started = time.monotonic()
            await self.refresh_once()

    async def aclose(self) -> None:
        await self._agent.aclose()


def runtime_config_manager_from_env(
    state: RuntimeConfigState,
) -> RuntimeConfigManager | None:
    application = os.environ.get("GATEWAY_APPCONFIG_APPLICATION")
    if not application:
        return None

    required = {
        name: os.environ.get(name, "")
        for name in (
            "GATEWAY_APPCONFIG_ENVIRONMENT",
            "GATEWAY_APPCONFIG_MODEL_PROFILE",
            "GATEWAY_APPCONFIG_CREDENTIAL_PROFILE",
            "GATEWAY_APPCONFIG_POOL_PROFILE",
            "GATEWAY_APPCONFIG_ACCESS_TOKEN",
        )
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            "AppConfig runtime refresh is missing required settings: "
            + ", ".join(sorted(missing))
        )

    poll_interval = float(os.environ.get("GATEWAY_RUNTIME_REFRESH_SECONDS", "5"))
    if poll_interval <= 0:
        raise RuntimeError("GATEWAY_RUNTIME_REFRESH_SECONDS must be positive")

    access_profile = os.environ.get("GATEWAY_APPCONFIG_ACCESS_PROFILE") or None
    budget_profile = os.environ.get("GATEWAY_APPCONFIG_BUDGET_PROFILE_ID") or None

    return RuntimeConfigManager(
        state=state,
        agent=AppConfigAgentClient(
            base_url=os.environ.get("GATEWAY_APPCONFIG_URL", "http://127.0.0.1:2772"),
            application=application,
            environment=required["GATEWAY_APPCONFIG_ENVIRONMENT"],
            access_token=required["GATEWAY_APPCONFIG_ACCESS_TOKEN"],
        ),
        model_profile=required["GATEWAY_APPCONFIG_MODEL_PROFILE"],
        credential_profile=required["GATEWAY_APPCONFIG_CREDENTIAL_PROFILE"],
        pool_profile=required["GATEWAY_APPCONFIG_POOL_PROFILE"],
        access_profile=access_profile,
        budget_profile=budget_profile,
        poll_interval_seconds=poll_interval,
    )


def _parse_model_snapshot(payload: bytes) -> ModelSnapshot:
    envelope = ModelEnvelope.model_validate_json(payload)
    entries = parse_registry_documents(envelope.documents)
    return ModelSnapshot(
        revision=envelope.revision,
        originated_at=envelope.originated_at,
        digest=hashlib.sha256(payload).hexdigest(),
        registries=split_registry_entries(entries),
    )


def _is_gateway_owned(field: str) -> bool:
    return field in GATEWAY_OWNED_FIELDS or field.startswith(
        GATEWAY_OWNED_FIELD_PREFIXES
    )


def _parse_credential_snapshot(
    payload: bytes,
    *,
    version: str,
    current: CredentialSnapshot | None,
) -> CredentialSnapshot:
    """Every field except Gateway-owned ones is managed, and stays managed after removal.

    Keeping removed fields managed for the worker's lifetime stops a deleted
    credential from resolving to a stale startup environment value.
    """
    digest = hashlib.sha256(payload).hexdigest()
    if current is not None and version == current.version and digest == current.digest:
        return current
    decoded: object = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("credential snapshot must be a JSON object")
    raw = cast(dict[str, object], decoded)
    owned = sorted(field for field in raw if _is_gateway_owned(field))
    if owned:
        logger.warning(
            "Ignored Gateway-owned credential snapshot fields=%s", ",".join(owned)
        )
        raw = {field: value for field, value in raw.items() if field not in owned}
    non_string = sorted(
        field
        for field, value in raw.items()
        if value is not None and not isinstance(value, str)
    )
    if non_string:
        logger.warning(
            "Skipped non-string credential snapshot fields=%s", ",".join(non_string)
        )
        raw = {f: v for f, v in raw.items() if f not in non_string}
    managed_fields = frozenset(raw)
    if current is not None:
        managed_fields |= current.managed_fields
    values: dict[str, SecretStr] = {}
    for field, value in raw.items():
        if isinstance(value, str) and value:
            values[field] = SecretStr(value)
    if current is not None and version == current.version:
        raise SnapshotRejectedError("credential version changed contents")
    return CredentialSnapshot(
        version=version,
        generation=1 if current is None else current.generation + 1,
        digest=digest,
        managed_fields=managed_fields,
        values=MappingProxyType(values),
    )


def _parse_pool_snapshot(
    payload: bytes, *, version: str, current: PoolConfigSnapshot | None
) -> PoolConfigSnapshot:
    digest = hashlib.sha256(payload).hexdigest()
    if current is not None and digest == current.digest:
        return current
    return PoolConfigSnapshot(
        digest=digest,
        document=PublishedPoolDocument.model_validate_json(payload),
        configuration_version=version,
    )

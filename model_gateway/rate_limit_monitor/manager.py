"""Background provider polling for the shared rate-limit monitor."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from anthropic import APIError as AnthropicAPIError
from httpx import HTTPError as HttpxHTTPError
from openai import APIError as OpenAIAPIError
from pydantic import SecretStr

from model_gateway.metrics import (
    RateLimitMonitorPollOutcome,
    record_rate_limit_monitor_ownership,
    record_rate_limit_monitor_poll,
    record_rate_limit_monitor_publish,
)
from model_gateway.model_helpers import managed_api_keys
from model_gateway.observability import log_gateway_event
from model_gateway.rate_limit_monitor.state import MonitorSourceUpdate
from model_gateway.rate_limit_monitor.types import (
    MonitorActivationResponse,
    MonitorListResponse,
    MonitorSourceName,
)
from model_library import model_library_settings
from model_library.base import LLM, LLMConfig
from model_library.registry_utils import get_registry_config, get_registry_model

DISCOVERY_SECONDS = 1.0
OWNERSHIP_RETRY_SECONDS = 5.0
HEARTBEAT_SECONDS = 10.0
POLL_INTERVAL_SECONDS = 5.0
PROVIDER_TIMEOUT_SECONDS = 30.0


class MonitorInvalidModel(ValueError):
    pass


class _MonitorStore(Protocol):
    async def activate(
        self,
        model: str,
        expected_sources: tuple[MonitorSourceName, ...],
    ) -> MonitorActivationResponse: ...

    async def list_states(self) -> MonitorListResponse: ...

    async def discover_active(self) -> set[str]: ...

    async def claim_owner(
        self,
        model: str,
        token: str,
    ) -> tuple[MonitorSourceName, ...] | None: ...

    async def renew_owner(
        self,
        model: str,
        token: str,
    ) -> tuple[str, float] | None: ...

    async def release_owner(self, model: str, token: str) -> None: ...

    async def publish_source(
        self,
        model: str,
        token: str,
        captured_generation: str,
        attempted_at: float,
        source: MonitorSourceUpdate,
    ) -> bool: ...


@dataclass(frozen=True)
class _SourceSpec:
    source: MonitorSourceName
    key_setting: str | None


@dataclass(frozen=True)
class _SourceProbe:
    source: MonitorSourceName
    provider: str
    probe: LLM


def _source_specs(model: str) -> tuple[_SourceSpec, ...]:
    managed_keys = managed_api_keys(model)
    if not managed_keys:
        return (_SourceSpec("default", None),)
    return tuple(
        _SourceSpec(managed_key.source, managed_key.key_setting)
        for managed_key in managed_keys
    )


class RateLimitMonitor:
    """Own shared activation and one process's leased background poll tasks."""

    def __init__(self, store: _MonitorStore):
        self._store = store
        self._model_tasks: dict[str, asyncio.Task[None]] = {}
        self._discovery_task: asyncio.Task[None] | None = None
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        self._monotonic: Callable[[], float] = time.monotonic
        self._closed = False

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("rate-limit monitor is closed")
        if self._discovery_task is None:
            self._discovery_task = asyncio.create_task(
                self._discovery_loop(),
                name="rate-limit-monitor-discovery",
            )

    def check_health(self) -> None:
        discovery_task = self._discovery_task
        if discovery_task is None or self._closed:
            raise RuntimeError("rate-limit monitor is not running")
        if discovery_task.done():
            discovery_task.result()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        discovery_task = self._discovery_task
        self._discovery_task = None
        if discovery_task is not None:
            discovery_task.cancel()
            await asyncio.gather(discovery_task, return_exceptions=True)
        await self._cancel_model_tasks()

    async def activate(self, model: str) -> MonitorActivationResponse:
        config = get_registry_config(model)
        if (
            config is None
            or config.rate_limit is None
            or not config.rate_limit.supports_live_monitoring
        ):
            raise MonitorInvalidModel("model is not eligible for monitoring")
        expected_sources: tuple[MonitorSourceName, ...] = tuple(
            spec.source for spec in _source_specs(model)
        )
        return await self._store.activate(model, expected_sources)

    async def list_states(self) -> MonitorListResponse:
        return await self._store.list_states()

    async def _discovery_loop(self) -> None:
        while True:
            active = await self._store.discover_active()
            for model, task in list(self._model_tasks.items()):
                if task.done():
                    if self._model_tasks.get(model) is task:
                        self._model_tasks.pop(model)
                    task.result()
                elif model not in active:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    finally:
                        if self._model_tasks.get(model) is task:
                            self._model_tasks.pop(model)
            for model in active:
                self._ensure_model_task(model)
            await self._sleep(DISCOVERY_SECONDS)

    def _ensure_model_task(self, model: str) -> None:
        if self._closed:
            return
        task = self._model_tasks.get(model)
        if task is not None and not task.done():
            return
        self._model_tasks[model] = asyncio.create_task(
            self._model_worker(model),
            name=f"rate-limit-monitor-{model}",
        )

    async def _cancel_model_tasks(self) -> None:
        tasks = list(self._model_tasks.items())
        for _, task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(
                *(task for _, task in tasks),
                return_exceptions=True,
            )
        for model, task in tasks:
            if self._model_tasks.get(model) is task:
                self._model_tasks.pop(model, None)

    async def _model_worker(self, model: str) -> None:
        token = secrets.token_urlsafe(24)
        claimed = False
        clean_exit = False
        children: list[asyncio.Task[None]] = []
        try:
            while True:
                source_names = await self._store.claim_owner(model, token)
                if source_names is None:
                    record_rate_limit_monitor_ownership("contended")
                    await self._sleep(OWNERSHIP_RETRY_SECONDS)
                    continue

                claimed = True
                record_rate_limit_monitor_ownership("acquired")
                local_source_names = tuple(spec.source for spec in _source_specs(model))
                if source_names == local_source_names:
                    break

                record_rate_limit_monitor_ownership("lost")
                clean_exit = True
                return

            sources = self._build_sources(model, source_names)
            children = [
                asyncio.create_task(
                    self._heartbeat(model, token),
                    name=f"rate-limit-monitor-heartbeat-{model}",
                ),
                *[
                    asyncio.create_task(
                        self._source_loop(model, token, source),
                        name=f"rate-limit-monitor-{model}-{source.source}",
                    )
                    for source in sources
                ],
            ]
            done, _ = await asyncio.wait(
                children,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                task.result()
            clean_exit = True
        except asyncio.CancelledError:
            clean_exit = claimed
            raise
        finally:
            for task in children:
                task.cancel()
            child_results = await asyncio.gather(*children, return_exceptions=True)
            if clean_exit:
                for result in child_results:
                    if isinstance(result, BaseException) and not isinstance(
                        result,
                        asyncio.CancelledError,
                    ):
                        clean_exit = False
                        raise result
            if claimed and clean_exit:
                await self._store.release_owner(model, token)

    async def _renew_owner(
        self,
        model: str,
        token: str,
    ) -> tuple[str, float] | None:
        renewal = await self._store.renew_owner(model, token)
        if renewal is None:
            record_rate_limit_monitor_ownership("lost")
        return renewal

    async def _heartbeat(self, model: str, token: str) -> None:
        while True:
            await self._sleep(HEARTBEAT_SECONDS)
            if await self._renew_owner(model, token) is None:
                return

    async def _source_loop(
        self,
        model: str,
        token: str,
        source: _SourceProbe,
    ) -> None:
        next_poll_at = self._monotonic()
        while True:
            while (delay := next_poll_at - self._monotonic()) > 0:
                await self._sleep(delay)

            renewal = await self._renew_owner(model, token)
            if renewal is None:
                return
            generation, attempted_at = renewal

            poll_started = time.perf_counter()
            try:
                async with asyncio.timeout(PROVIDER_TIMEOUT_SECONDS):
                    rate_limit = await source.probe.get_rate_limit()
            except (
                TimeoutError,
                HttpxHTTPError,
                OpenAIAPIError,
                AnthropicAPIError,
            ) as exc:
                poll_outcome: RateLimitMonitorPollOutcome = "provider_error"
                self._log_provider_failure(model, source.source, exc)
                result = MonitorSourceUpdate(
                    source=source.source,
                    status="error",
                )
            else:
                if rate_limit is None:
                    poll_outcome = "unsupported"
                    result = MonitorSourceUpdate(
                        source=source.source,
                        status="unsupported",
                    )
                else:
                    poll_outcome = "ok"
                    result = MonitorSourceUpdate(
                        source=source.source,
                        status="ok",
                        rate_limit=rate_limit,
                    )
            record_rate_limit_monitor_poll(
                provider=source.provider,
                source=source.source,
                outcome=poll_outcome,
                latency_ms=(time.perf_counter() - poll_started) * 1000,
            )

            published = await self._store.publish_source(
                model,
                token,
                generation,
                attempted_at,
                result,
            )
            record_rate_limit_monitor_publish("accepted" if published else "rejected")

            next_poll_at += POLL_INTERVAL_SECONDS
            while next_poll_at <= self._monotonic():
                next_poll_at += POLL_INTERVAL_SECONDS

    def _build_sources(
        self,
        model: str,
        expected_sources: tuple[MonitorSourceName, ...],
    ) -> list[_SourceProbe]:
        specs = _source_specs(model)
        if tuple(spec.source for spec in specs) != expected_sources:
            raise ValueError(
                "monitor source configuration does not match persisted state"
            )
        config = get_registry_config(model)
        if config is None:
            raise MonitorInvalidModel("monitored model is not configured")
        return [
            self._construct_source(model, config.provider_name, spec) for spec in specs
        ]

    def _construct_source(
        self,
        model: str,
        provider: str,
        spec: _SourceSpec,
    ) -> _SourceProbe:
        if spec.key_setting is None:
            probe = get_registry_model(model)
        else:
            api_key = model_library_settings.get(spec.key_setting)
            if not api_key:
                raise RuntimeError(
                    f"rate-limit monitor API key is not configured: {spec.key_setting}"
                )
            probe = get_registry_model(
                model,
                LLMConfig(custom_api_key=SecretStr(api_key)),
            )
        return _SourceProbe(spec.source, provider, probe)

    @staticmethod
    def _log_provider_failure(
        model: str,
        source: MonitorSourceName,
        exc: BaseException,
    ) -> None:
        log_gateway_event(
            "gateway.rate_limit_monitor.provider_failure",
            level=logging.WARNING,
            model=model,
            source=source,
            code="provider_error",
            exception_type=exc.__class__.__name__,
        )

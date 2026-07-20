import logging
import time
from typing import cast

from model_gateway.benchmark_admission_types import (
    BenchmarkAdmissionOutcome,
    BenchmarkAdmissionResponse,
)
from model_library.retriers.token.benchmark_queue import (
    EARLY_RELEASE_GRACE_PERIOD,
    HEARTBEAT_INTERVAL,
    HOURS_24,
    BenchmarkQueueKeys,
    clear_benchmark_notification,
    control_benchmark_run,
    early_release_benchmark_run,
    initialize_benchmark_run,
    mark_benchmark_slot_acquired,
    refresh_benchmark_heartbeat,
    release_benchmark_run,
    wait_for_benchmark_slot,
)
from model_library.retriers.token.utils import KEY_PREFIX, AsyncRedisClient

ADMISSION_OPERATION_LOCK_TTL = HEARTBEAT_INTERVAL * 10


class BenchmarkAdmissionConflict(Exception):
    """Raised when one live run ID is used with incompatible admission inputs."""


def get_benchmark_run_pointer_key(run_id: str) -> str:
    return f"{KEY_PREFIX}:benchmark:admission:{run_id}"


class BenchmarkAdmissionStore:
    """Request-driven benchmark admission state backed only by shared Redis."""

    def __init__(self, redis_client: AsyncRedisClient, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger

    async def acquire(
        self,
        *,
        model: str,
        model_registry_key: tuple[str, str],
        run_id: str,
        effective_token_limit: int,
        total_requests: int | None,
        early_release: bool,
        immediate_queue_release: bool,
    ) -> BenchmarkAdmissionResponse:
        pointer_key = get_benchmark_run_pointer_key(run_id)
        requested_keys = BenchmarkQueueKeys.for_run(model_registry_key, run_id)

        async with self.redis.lock(
            f"{pointer_key}:lock",
            timeout=ADMISSION_OPERATION_LOCK_TTL,
        ):
            pointer = await self.redis.hgetall(pointer_key)
            if pointer:
                current_keys = BenchmarkQueueKeys.for_base(pointer["base"], run_id)
                current_meta = await self.redis.hgetall(current_keys.run_meta)

                if pointer.get("initializing") == "1":
                    if (
                        pointer.get("model") != model
                        or pointer.get("base") != requested_keys.base
                    ):
                        raise BenchmarkAdmissionConflict(
                            "Initializing benchmark admission inputs do not match the existing run"
                        )
                else:
                    current_outcome = current_meta.get("outcome")
                    current_alive = bool(await self.redis.exists(current_keys.alive))

                    if current_outcome is None and current_alive:
                        self._require_compatible_live_acquire(
                            model=model,
                            requested_keys=requested_keys,
                            pointer=pointer,
                            meta=current_meta,
                            effective_token_limit=effective_token_limit,
                            total_requests=total_requests,
                            early_release=early_release,
                            immediate_queue_release=immediate_queue_release,
                        )
                        await refresh_benchmark_heartbeat(self.redis, current_keys)
                        await control_benchmark_run(
                            self.redis,
                            current_keys,
                            run_id,
                            self.logger,
                            self_promote=False,
                        )
                        current_meta = await self.redis.hgetall(current_keys.run_meta)
                        return self._response(model, run_id, current_meta)

                    if current_outcome is None:
                        await self._release_locked(
                            model=model,
                            run_id=run_id,
                            keys=current_keys,
                            outcome="failed",
                            meta=current_meta,
                        )
                    elif "completed_at" not in current_meta:
                        await self._release_locked(
                            model=model,
                            run_id=run_id,
                            keys=current_keys,
                            outcome=cast(BenchmarkAdmissionOutcome, current_outcome),
                            meta=current_meta,
                        )

            # Persist recoverable run ownership before queue effects. If this
            # worker stops during initialization, a retry can repair the same
            # pointer instead of creating an orphan in another queue.
            await self.redis.hset(
                pointer_key,
                mapping={
                    "model": model,
                    "base": requested_keys.base,
                    "initializing": "1",
                },
            )
            await self.redis.expire(pointer_key, HOURS_24)

            await clear_benchmark_notification(self.redis, requested_keys)
            await initialize_benchmark_run(
                self.redis,
                requested_keys,
                run_id,
                total_requests,
                self.logger,
            )
            await self.redis.hset(
                requested_keys.run_meta,
                mapping={
                    "effective_token_limit": effective_token_limit,
                    "early_release": "1" if early_release else "0",
                    "immediate_queue_release": (
                        "1" if immediate_queue_release else "0"
                    ),
                },
            )
            await self.redis.hdel(pointer_key, "initializing")
            meta = await self.redis.hgetall(requested_keys.run_meta)
            return self._response(model, run_id, meta)

    async def wait(
        self,
        *,
        model: str,
        run_id: str,
        timeout_seconds: float,
    ) -> BenchmarkAdmissionResponse:
        pointer_key = get_benchmark_run_pointer_key(run_id)
        initial = await self._inspect_live(pointer_key, model, run_id)
        if initial.state != "waiting":
            return initial

        pointer = await self._require_pointer(pointer_key, model, run_id)
        keys = BenchmarkQueueKeys.for_base(pointer["base"], run_id)
        acquired = await wait_for_benchmark_slot(
            self.redis,
            keys,
            run_id,
            self.logger,
            None,
            timeout_seconds=timeout_seconds,
        )

        async with self.redis.lock(
            f"{pointer_key}:lock",
            timeout=ADMISSION_OPERATION_LOCK_TTL,
        ):
            pointer = await self._require_pointer(pointer_key, model, run_id)
            keys = BenchmarkQueueKeys.for_base(pointer["base"], run_id)
            meta = await self.redis.hgetall(keys.run_meta)
            outcome = meta.get("outcome")
            if outcome is not None:
                return self._response(model, run_id, meta)
            if not await self.redis.exists(keys.alive):
                return await self._release_locked(
                    model=model,
                    run_id=run_id,
                    keys=keys,
                    outcome="failed",
                    meta=meta,
                )
            if acquired:
                await mark_benchmark_slot_acquired(self.redis, keys)
                meta = await self.redis.hgetall(keys.run_meta)
            return self._response(model, run_id, meta)

    async def renew(
        self,
        *,
        model: str,
        run_id: str,
    ) -> BenchmarkAdmissionResponse:
        pointer_key = get_benchmark_run_pointer_key(run_id)
        async with self.redis.lock(
            f"{pointer_key}:lock",
            timeout=ADMISSION_OPERATION_LOCK_TTL,
        ):
            pointer = await self._require_pointer(pointer_key, model, run_id)
            keys = BenchmarkQueueKeys.for_base(pointer["base"], run_id)
            meta = await self.redis.hgetall(keys.run_meta)
            if meta.get("outcome") is not None:
                return self._response(model, run_id, meta)
            if not await self.redis.exists(keys.alive):
                return await self._release_locked(
                    model=model,
                    run_id=run_id,
                    keys=keys,
                    outcome="failed",
                    meta=meta,
                )

            await refresh_benchmark_heartbeat(self.redis, keys)
            await self._maybe_early_release(keys, run_id, meta)
            await control_benchmark_run(
                self.redis,
                keys,
                run_id,
                self.logger,
                self_promote=meta.get("slot_acquired") != "1",
            )
            meta = await self.redis.hgetall(keys.run_meta)
            return self._response(model, run_id, meta)

    async def release(
        self,
        *,
        model: str,
        run_id: str,
        outcome: BenchmarkAdmissionOutcome,
    ) -> BenchmarkAdmissionResponse:
        pointer_key = get_benchmark_run_pointer_key(run_id)
        async with self.redis.lock(
            f"{pointer_key}:lock",
            timeout=ADMISSION_OPERATION_LOCK_TTL,
        ):
            pointer = await self._require_pointer(pointer_key, model, run_id)
            keys = BenchmarkQueueKeys.for_base(pointer["base"], run_id)
            meta = await self.redis.hgetall(keys.run_meta)
            return await self._release_locked(
                model=model,
                run_id=run_id,
                keys=keys,
                outcome=outcome,
                meta=meta,
            )

    async def _inspect_live(
        self,
        pointer_key: str,
        model: str,
        run_id: str,
    ) -> BenchmarkAdmissionResponse:
        async with self.redis.lock(
            f"{pointer_key}:lock",
            timeout=ADMISSION_OPERATION_LOCK_TTL,
        ):
            pointer = await self._require_pointer(pointer_key, model, run_id)
            keys = BenchmarkQueueKeys.for_base(pointer["base"], run_id)
            meta = await self.redis.hgetall(keys.run_meta)
            if meta.get("outcome") is not None:
                return self._response(model, run_id, meta)
            if not await self.redis.exists(keys.alive):
                return await self._release_locked(
                    model=model,
                    run_id=run_id,
                    keys=keys,
                    outcome="failed",
                    meta=meta,
                )
            return self._response(model, run_id, meta)

    async def _maybe_early_release(
        self,
        keys: BenchmarkQueueKeys,
        run_id: str,
        meta: dict[str, str],
    ) -> None:
        if (
            meta.get("slot_acquired") != "1"
            or meta.get("early_release") != "1"
            or "popped_at" in meta
        ):
            return

        total_requests = int(meta.get("total_requests", "0"))
        if total_requests == 0:
            return
        if await self.redis.scard(keys.dispatched) < total_requests:
            return

        if meta.get("immediate_queue_release") == "1":
            await early_release_benchmark_run(self.redis, keys, run_id, self.logger)
            return

        now = time.time()
        deadline_value = meta.get("early_release_deadline")
        if deadline_value is None:
            deadline = now + EARLY_RELEASE_GRACE_PERIOD
            await self.redis.hset(
                keys.run_meta,
                mapping={"early_release_deadline": deadline},
            )
        else:
            deadline = float(deadline_value)
        if now >= deadline:
            await early_release_benchmark_run(self.redis, keys, run_id, self.logger)

    async def _release_locked(
        self,
        *,
        model: str,
        run_id: str,
        keys: BenchmarkQueueKeys,
        outcome: BenchmarkAdmissionOutcome,
        meta: dict[str, str],
    ) -> BenchmarkAdmissionResponse:
        existing_outcome = meta.get("outcome")
        terminal_outcome = (
            cast(BenchmarkAdmissionOutcome, existing_outcome)
            if existing_outcome is not None
            else outcome
        )
        if existing_outcome is None:
            await self.redis.hset(keys.run_meta, mapping={"outcome": outcome})

        # A retry must finish cleanup if the first release persisted its outcome
        # and then lost its worker before removing queue ownership.
        if "completed_at" not in meta:
            await release_benchmark_run(
                self.redis,
                keys,
                run_id,
                self.logger,
                notify_next=meta.get("slot_acquired") == "1",
            )
        return BenchmarkAdmissionResponse(
            model=model,
            run_id=run_id,
            state="released",
            effective_token_limit=int(meta["effective_token_limit"]),
            outcome=terminal_outcome,
        )

    async def _require_pointer(
        self,
        pointer_key: str,
        model: str,
        run_id: str,
    ) -> dict[str, str]:
        pointer = await self.redis.hgetall(pointer_key)
        if not pointer or pointer.get("model") != model:
            raise BenchmarkAdmissionConflict(
                f"Run {run_id} does not have a live admission for model {model}"
            )
        return pointer

    @staticmethod
    def _require_compatible_live_acquire(
        *,
        model: str,
        requested_keys: BenchmarkQueueKeys,
        pointer: dict[str, str],
        meta: dict[str, str],
        effective_token_limit: int,
        total_requests: int | None,
        early_release: bool,
        immediate_queue_release: bool,
    ) -> None:
        compatible = (
            pointer.get("model") == model
            and pointer.get("base") == requested_keys.base
            and meta.get("effective_token_limit") == str(effective_token_limit)
            and meta.get("total_requests", "0") == str(total_requests or 0)
            and meta.get("early_release", "0") == ("1" if early_release else "0")
            and meta.get("immediate_queue_release", "0")
            == ("1" if immediate_queue_release else "0")
        )
        if not compatible:
            raise BenchmarkAdmissionConflict(
                "Live benchmark admission inputs do not match the existing run"
            )

    @staticmethod
    def _response(
        model: str,
        run_id: str,
        meta: dict[str, str],
    ) -> BenchmarkAdmissionResponse:
        outcome = meta.get("outcome")
        if outcome is not None:
            return BenchmarkAdmissionResponse(
                model=model,
                run_id=run_id,
                state="released",
                effective_token_limit=int(meta["effective_token_limit"]),
                outcome=cast(BenchmarkAdmissionOutcome, outcome),
            )
        return BenchmarkAdmissionResponse(
            model=model,
            run_id=run_id,
            state="acquired" if meta.get("slot_acquired") == "1" else "waiting",
            effective_token_limit=int(meta["effective_token_limit"]),
        )

import asyncio
import logging

from model_gateway.managed_auth.identity import HumanVerifier
from model_gateway.managed_auth.store import ManagedAuthStore


logger = logging.getLogger(__name__)


class HumanReconciler:
    def __init__(
        self, store: ManagedAuthStore, verifier: HumanVerifier, expected_issuer: str
    ):
        self._store = store
        self._verifier = verifier
        self._expected_issuer = expected_issuer

    async def reconcile_once(self) -> None:
        for identity in await self._store.list_known_humans(self._expected_issuer):
            try:
                enabled = await self._verifier.current_eligibility(identity)
            except Exception:
                logger.exception(
                    "Unable to refresh current human eligibility for %s",
                    identity.subject,
                )
                continue
            await self._store.record_system_human_eligibility(identity, enabled)

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.reconcile_once()
            except Exception:
                logger.exception("Unable to reconcile current human eligibility")
            try:
                await asyncio.wait_for(stop.wait(), timeout=60)
            except TimeoutError:
                pass

"""Router-provider mixin: task session ids and served-model pricing.

Router-only members live here rather than on `LLM`, so the base class does not
collect a member per model type. `LLM._calculate_cost` dispatches `is_router`
queries to `RouterLLM.router_cost`.
"""

from __future__ import annotations

from collections.abc import Sequence

from model_library.base.input import InputItem, ToolDefinition
from model_library.base.output import QueryResultCost, QueryResultMetadata
from model_library.base.query_ids import prompt_cache_key_from_query_ids


class RouterLLM:
    """Mixin for providers that route to a served model.

    Inherit alongside the provider's usual base (`OpenAIModel`, `DelegateOnly`).
    `router_session_id` is the shared per-task session id each router sends in
    its own field. A provider opts into served-model pricing by overriding
    `_served_model_cost_key` and, optionally, `_router_fee`.
    """

    provider: str
    model_name: str
    _registry_key: str | None

    def router_session_id(
        self, input: Sequence[InputItem], tools: Sequence[ToolDefinition]
    ) -> str:
        """One session id per task, hashed from the registry key and task ids (or the prompt prefix without them)."""
        return prompt_cache_key_from_query_ids(
            model_name=self._registry_key or f"{self.provider}/{self.model_name}",
            input=input,
            tools=tools,
        )

    def _served_model_cost_key(self, served_model: str) -> str | None:
        """Registry key to price the served model under. `None` prices nothing."""
        return None

    def _router_fee(self, metadata: QueryResultMetadata) -> float:
        """Per-call fee added on top of the served model's price."""
        return 0.0

    def router_cost(
        self,
        metadata: QueryResultMetadata,
        *,
        batch: bool,
        bill_reasoning: bool,
    ) -> QueryResultCost | None:
        """Price the served model from the registry; `None` without a cost key."""
        from model_library.registry_utils import compute_model_cost

        key = metadata.served_model and self._served_model_cost_key(
            metadata.served_model
        )
        if not key:
            return None
        cost = compute_model_cost(
            key, metadata, batch=batch, bill_reasoning=bill_reasoning
        )
        if cost is None or not (fee := self._router_fee(metadata)):
            return cost
        return QueryResultCost(input=0, output=0, total_override=cost.total + fee)

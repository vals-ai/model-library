"""Anthropic cost of billed declined fallback hops."""

from typing import cast

import pytest

from model_library.base.output.result import (
    FallbackHop,
    FallbackInfo,
    QueryResultMetadata,
)
from model_library.providers.anthropic import AnthropicModel
from model_library.registry_utils import compute_model_cost


def _fallback_metadata(declined_out_tokens: int, category: str) -> QueryResultMetadata:
    return QueryResultMetadata(
        in_tokens=1000,
        out_tokens=100,
        fallback=FallbackInfo(
            requested_model="anthropic/claude-opus-4-8",
            served_model="anthropic/claude-opus-5",
            hops=[
                FallbackHop(
                    model="anthropic/claude-opus-4-8",
                    served=False,
                    trigger="refusal",
                    category=category,
                    usage=QueryResultMetadata(
                        in_tokens=1000, out_tokens=declined_out_tokens
                    ),
                ),
                FallbackHop(
                    model="anthropic/claude-opus-5",
                    served=True,
                    usage=QueryResultMetadata(in_tokens=1000, out_tokens=100),
                ),
            ],
        ),
    )


@pytest.mark.parametrize(
    ("declined_out_tokens", "category", "billed"),
    [(50, "cyber", True), (0, "cyber", False), (0, "bio", True)],
)
async def test_cost_adds_billed_declined_hops(
    declined_out_tokens: int, category: str, billed: bool
):
    metadata = _fallback_metadata(declined_out_tokens, category)
    model = AnthropicModel("claude-opus-4-8")

    cost = await model._calculate_cost(metadata)  # pyright: ignore[reportPrivateUsage]

    served = compute_model_cost("anthropic/claude-opus-5", metadata)
    declined = compute_model_cost(
        "anthropic/claude-opus-4-8", cast(FallbackInfo, metadata.fallback).hops[0].usage
    )
    assert cost is not None and served is not None and declined is not None
    assert cost.total == pytest.approx(
        served.total + (declined.total if billed else 0)
    )

"""Live smoke tests for the typed-decision models, called directly. Skipped unless the route's key is set."""

from __future__ import annotations

import os

import pytest

from model_library.providers.delegates.openrouter.decision import decision_request, parse_decision_output
from model_library.registry_utils import get_registry_model

pytestmark = pytest.mark.integration

OPTIONS = {
    "billing": "Charges, invoices, and refunds",
    "engineering": "Bugs, errors, and outages",
    "support": "How-to and account questions",
}
REQUEST = decision_request(
    "The API returns 500 on every call.",
    {"team": {"type": "choice", "instructions": "Which team should handle this ticket?", "criteria": OPTIONS}},
)


@pytest.mark.parametrize(
    ("model_key", "key_env"),
    [
        ("decision/jev-1.13.0", "OPENROUTER_API_KEY"),
        ("decision/mercury-decide", "OPENROUTER_API_KEY"),
        ("decision/microsoft-decision-1", "OPENROUTER_API_KEY"),
        ("decision/gpt-6-luna", "OPENAI_API_KEY"),
    ],
)
async def test_live_decision_call(model_key: str, key_env: str) -> None:
    if not os.environ.get(key_env):
        pytest.skip(f"{key_env} not set")

    result = await get_registry_model(model_key).query(REQUEST)

    answer = parse_decision_output(result.output_text)["team"]
    assert answer["choice"] in OPTIONS
    assert set(answer["probabilities"]) == set(OPTIONS)
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=0.01)
    assert result.metadata.in_tokens > 0
    assert result.metadata.cost is not None and result.metadata.cost.total > 0

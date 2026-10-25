from decimal import Decimal
from unittest.mock import AsyncMock, patch

from model_gateway.run_tokens import RunTokenClaims, sha256_hex
from model_gateway.valsmith_catalog import VALSMITH_OVERRIDES, ValSmithPolicy
from model_library.register_models import get_model_registry, model_config_from_json
from tests.unit.model_gateway._support import HEADERS, _make_client


def test_catalog_preserves_prices_families_and_caller_scope(monkeypatch):
    root_key, high_key, alias_key = (
        "openai/future",
        "openai/future-high",
        "openai/future-alias",
    )
    template = get_model_registry()["openai/gpt-4o"]
    registry = {
        key: template.model_copy(deep=True, update={"full_key": key, "alternative_keys": []})
        for key in (root_key, high_key, alias_key)
    }
    registry[root_key].alternative_keys = [high_key, alias_key]
    for model in registry.values():
        model.default_parameters.reasoning_effort = "ultra"
    registry[high_key].default_parameters.reasoning_effort = "high"
    base_rate = "0.10000000000000000000000000001"
    alias_rate = "0.10000000000000000000000000002"
    monkeypatch.setitem(
        VALSMITH_OVERRIDES,
        root_key,
        ValSmithPolicy(credits_per_task=Decimal(base_rate), required_plan="enterprise"),
    )
    monkeypatch.setitem(
        VALSMITH_OVERRIDES,
        alias_key,
        ValSmithPolicy(credits_per_task=Decimal(alias_rate)),
    )
    monkeypatch.setattr(
        "model_gateway.routes.models.get_registries", lambda: (registry, {})
    )

    client = _make_client()
    assert client.get("/registry?include_valsmith=true").status_code == 401
    response = client.get("/registry?include_valsmith=true", headers=HEADERS)
    assert response.status_code == 200
    catalog = response.json()
    models = catalog["models"]
    assert set(models) == set(registry)
    for key, rate in ((root_key, base_rate), (high_key, base_rate), (alias_key, alias_rate)):
        assert models[key]["valsmith"] == {
            "credits_per_task": rate,
            "required_plan": "enterprise",
        }
    assert models[root_key]["catalog"]["thinking_options"] == [
        {"value": "high", "label": "High", "model_key": high_key},
        {"value": "ultra", "label": "Ultra", "model_key": root_key},
    ]
    assert models[high_key]["catalog"]["family_key"] == root_key
    assert models[alias_key]["catalog"]["family_key"] == alias_key

    regular = client.get("/registry", headers=HEADERS).json()["models"][root_key]
    assert "valsmith" not in regular
    with patch("model_library.register_models.logger.warning") as warning:
        assert model_config_from_json(regular).full_key == root_key
        warning.assert_not_called()

    registry[root_key].label = "Updated label"
    renamed = client.get("/registry?include_valsmith=true", headers=HEADERS).json()
    assert renamed["catalog_revision"] != catalog["catalog_revision"]
    assert renamed["pricing_revision"] == catalog["pricing_revision"]

    registry[root_key].metadata.available_for_everyone = False
    claims = RunTokenClaims(
        run_id="test-run",
        task_id="test-task",
        attempt_id="test-attempt",
        allowed_models=[high_key],
        minted_by="test",
        minted_with=sha256_hex("sk-test"),
    )
    with patch("model_gateway.auth.get_run_token", AsyncMock(return_value=claims)):
        scoped = client.get(
            "/registry?include_valsmith=true",
            headers={"Authorization": "Bearer mgwt_test"},
        )
    assert scoped.status_code == 200
    scoped_models = scoped.json()["models"]
    assert set(scoped_models) == {high_key}
    assert scoped_models[high_key]["valsmith"] == models[high_key]["valsmith"]
    assert scoped_models[high_key]["catalog"] == {
        "family_key": high_key,
        "thinking_options": [],
    }

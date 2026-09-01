from datetime import date

from scripts.deprecate_model import (
    model_to_deprecated_entry_data,
    model_to_yaml_entry,
)


class FakeModelConfig:
    def __init__(self, costs: dict[str, object] | None = None):
        self._costs = costs

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "python"
        return {
            "costs_per_million_token": self._costs,
            "company": "Google",
            "label": "Gemini 3.1 Flash Lite Preview",
            "release_date": date(2026, 3, 3),
            "provider_name": "google",
            "full_key": "google/gemini-3.1-flash-lite-preview",
            "slug": "google-gemini-3-1-flash-lite-preview",
            "provider_properties": {},
            "metadata": {"deprecated": True, "available_for_everyone": True},
        }


def test_deprecated_entry_preserves_release_date_as_yaml_date_scalar():
    model_data = model_to_deprecated_entry_data(FakeModelConfig())
    yaml_entry = model_to_yaml_entry(
        "google/gemini-3.1-flash-lite-preview",
        model_data,
    )

    assert "release_date: 2026-03-03" in yaml_entry
    assert "release_date: '2026-03-03'" not in yaml_entry
    assert "provider_name" not in yaml_entry
    assert "provider_properties" not in yaml_entry
    assert "deprecated" not in yaml_entry


def test_deprecated_entry_keeps_null_costs_so_the_entry_stays_loadable():
    """`costs_per_million_token` is nullable but required by RawModelConfig."""
    model_data = model_to_deprecated_entry_data(FakeModelConfig())
    assert model_data["costs_per_million_token"] is None
    assert "costs_per_million_token: null" in model_to_yaml_entry(
        "google/gemini-3.1-flash-lite-preview", model_data
    )

    priced = model_to_deprecated_entry_data(FakeModelConfig({"input": 0.1}))
    assert priced["costs_per_million_token"] == {"input": 0.1}

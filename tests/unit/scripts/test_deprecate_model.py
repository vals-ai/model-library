from datetime import date

from scripts.deprecate_model import (
    model_to_deprecated_entry_data,
    model_to_yaml_entry,
)


class FakeModelConfig:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "python"
        return {
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

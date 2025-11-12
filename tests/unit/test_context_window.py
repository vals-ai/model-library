"""Unit tests for context window helper function"""

from model_library.utils import get_context_window_for_model


def test_context_window_helper():
    """Test that the context window helper function works correctly"""

    expected_context_lengths = {
        "gpt-4.1-mini-2025-04-14": 1_047_576,
        "gpt-4.1-2025-04-14": 1_047_576,
        "gpt-4.1-nano-2025-04-14": 1_047_576,
        "gpt-4o-mini-2024-07-18": 128_000,
        "gpt-4o-2024-08-06": 128_000,
        "gpt-4o-2024-11-20": 128_000,
        "gpt-4o-2024-05-13": 128_000,
        "o1-2024-12-17": 200_000,
        "o3-mini-2025-01-31": 200_000,
        "o3-deep-research-2025-06-26": 200_000,
        "o3-deep-research": 200_000,
        "o3-pro-2025-06-10": 200_000,
        "o3-2025-04-16": 200_000,
        "o4-mini-deep-research-2025-06-26": 200_000,
        "o4-mini-deep-research": 200_000,
        "o4-mini-2025-04-16": 200_000,
        "gpt-5-2025-08-07": 400_000,
        "gpt-5-codex": 400_000,
        # "gpt-oss-120b": 131_072, # inference done only by others like fireworks
        # "gpt-oss-20b": 131_072, # so not tested here
    }

    # Test each model (allow up to 4k difference)
    for model_name, expected in expected_context_lengths.items():
        model_key = f"openai/{model_name}"
        actual = get_context_window_for_model(model_key)
        diff = abs(actual - expected)
        assert diff <= 4000, (
            f"{model_key}: expected {expected}, got {actual} (diff: {diff})"
        )


def test_context_window_helper_default():
    """Test default behavior for unknown models"""
    assert get_context_window_for_model("openai/non-existent") == 128_000
    assert get_context_window_for_model("openai/non-existent", default=64_000) == 64_000

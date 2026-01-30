"""Unit tests for model_library/utils.py"""

from model_library.utils import get_context_window_for_model


def test_get_context_window_for_existing_model():
    """Test that context window is correctly fetched for a model that exists."""
    context_window = get_context_window_for_model("openai/gpt-4o-mini")
    assert context_window == 128_000


def test_get_context_window_for_nonexistent_model():
    """Test that None is returned for a model that doesn't exist."""
    context_window = get_context_window_for_model("nonexistent/fake-model-xyz")
    assert context_window is None

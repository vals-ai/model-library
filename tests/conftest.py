"""
Test configuration and fixtures.
"""

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load from root .env file
root_env = Path(__file__).parent.parent / ".env"
load_dotenv(root_env)


@pytest.fixture
def google_models() -> Dict[str, Dict[str, Any]]:
    """Google model configurations for testing."""
    return {
        "thinking": {
            "name": "gemini-2.5-flash-thinking",
            "supports_thinking": True,
            "default_budget": 24576,
        },
        "2.5-pro": {
            "name": "gemini-2.5-pro",
            "supports_thinking": True,
            "default_budget": -1,
        },
        "2.5-flash": {
            "name": "gemini-2.5-flash",
            "supports_thinking": True,
            "default_budget": -1,
        },
        "2.5-flash-lite": {
            "name": "gemini-2.5-flash-lite",
            "supports_thinking": True,
            "default_budget": 0,
        },
        "1.5-pro": {
            "name": "gemini-1.5-pro",
            "supports_thinking": False,
            "default_budget": None,
        },
    }


@pytest.fixture
def provider_api_keys() -> Dict[str, str | None]:
    """API keys from environment for different providers."""
    return {
        "google": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "azure": os.getenv("AZURE_API_KEY"),
    }


@pytest.fixture
def mock_google_response():
    """Mock Google API response for unit tests."""
    from google.genai.types import GenerateContentResponse

    def _create_response(
        output_text: str = "Test response",
        reasoning_text: str = "",
        in_tokens: int = 10,
        out_tokens: int = 20,
        reasoning_tokens: int | None = None,
    ):
        response = MagicMock(spec=GenerateContentResponse)
        response.candidates = [MagicMock()]

        parts: list[Any] = []
        if reasoning_text:
            reasoning_part = MagicMock()
            reasoning_part.text = reasoning_text
            reasoning_part.thought = True
            parts.append(reasoning_part)

        if output_text:
            output_part = MagicMock()
            output_part.text = output_text
            output_part.thought = False
            parts.append(output_part)

        response.candidates[0].content.parts = parts
        response.candidates[0].finish_reason = None

        class UsageMetadata:
            def __init__(self):
                self.prompt_token_count = in_tokens
                self.candidates_token_count = out_tokens
                if reasoning_tokens is not None:
                    self.thoughts_token_count = reasoning_tokens

        response.usage_metadata = UsageMetadata()

        return response

    return _create_response


@pytest.fixture
def mock_google_client(mock_google_response: Any):
    """Mock Google client for unit tests.

    Returns a dict with all three mocked clients so tests can check which was called.
    For backwards compatibility, accessing the fixture directly returns the genai client.
    """

    with (
        patch("model_library.providers.google.GoogleModel.get_client") as mock_client,
    ):
        # Create a single shared mock instance that all clients will return
        shared_client_instance = MagicMock()
        shared_client_instance.aio.models.generate_content = AsyncMock(
            return_value=mock_google_response()
        )
        shared_client_instance.aio.models.generate_content_stream = AsyncMock()

        # All client constructors return the same instance
        mock_client.return_value = shared_client_instance

        yield mock_client


@pytest.fixture
def mock_model_settings():
    """Mock model library settings to avoid real API key access."""
    with (
        patch(
            "model_library.providers.google.google.model_library_settings"
        ) as mock_settings,
    ):
        # GenAI settings
        mock_settings.GOOGLE_API_KEY = "test_key"
        mock_settings.OPENAI_API_KEY = "test_openai_key"

        yield mock_settings


def pytest_configure(config: Any):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests that don't require API access")
    config.option.asyncio_default_fixture_loop_scope = "function"
    config.addinivalue_line(
        "markers", "integration: Integration tests that require API access"
    )
    config.addinivalue_line("markers", "slow: Slow tests that might take time")
    config.addinivalue_line(
        "markers", "requires_google_api: Tests that require Google API key"
    )
    config.addinivalue_line(
        "markers", "requires_openai_api: Tests that require OpenAI API key"
    )


def has_google_api_key():
    """Check if any form of Google/Gemini API key is available."""
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return True

    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        return True

    return False


requires_google_api = pytest.mark.skipif(
    not has_google_api_key(), reason="Google API key not available"
)

requires_openai_api = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available"
)

requires_anthropic_api = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not available"
)


def has_mercury_api_key() -> bool:
    """Check if Mercury API key is available."""
    mercury_key = os.getenv("MERCURY_API_KEY")
    return bool(mercury_key)


requires_mercury_api = pytest.mark.skipif(
    not has_mercury_api_key(), reason="Mercury API key not available"
)

requires_any_api = pytest.mark.skipif(
    not any(
        [
            os.getenv("GEMINI_API_KEY"),
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("MERCURY_API_KEY"),
        ]
    ),
    reason="No API keys available for testing",
)


@pytest.fixture
def assert_thinking_config() -> Callable[[Dict[str, Any], bool, Optional[int]], None]:
    """Helper to assert thinking configuration."""

    def _assert(
        config: Dict[str, Any],
        should_have_thinking: bool,
        expected_budget: Optional[int] = None,
    ) -> None:
        # get_api_call_config normalizes config to a Dict[str, Any]
        cfg: Dict[str, Any] = config
        if should_have_thinking:
            assert "thinking_config" in cfg and cfg["thinking_config"] is not None
            tc = cast(Dict[str, Any], cfg["thinking_config"])
            if expected_budget is not None:
                budget_any = tc.get("thinking_budget")
                assert isinstance(budget_any, int)
                assert budget_any == expected_budget
            include_any = tc.get("include_thoughts")
            assert isinstance(include_any, bool)
            assert include_any is True
        else:
            assert ("thinking_config" not in cfg) or (
                cfg.get("thinking_config") is None
            )

    return _assert


@pytest.fixture
def create_test_input() -> object:
    """Helper to create test inputs."""
    from model_library.base import TextInput

    def _create(text: str = "Test input", files: object = None) -> list[object]:
        inputs: list[object] = [TextInput(text=text)]
        if files:
            pass
        return inputs

    return _create


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()

    yield loop

    if not loop.is_closed():
        loop.close()


@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment after each test."""
    yield

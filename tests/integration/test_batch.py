"""
Integration tests for batch processing (real API when available).
Tests batch operations for providers that support batch processing.
"""

import uuid
from typing import Any

import pytest

from tests.conftest import (
    requires_anthropic_api,
    requires_google_api,
    requires_openai_api,
)
from model_library.base import LLMConfig, TextInput
from model_library.providers.anthropic import AnthropicModel
from model_library.providers.openai import OpenAIModel
from model_library.providers.google import GoogleModel


@pytest.mark.integration
class TestBatch:
    """Test batch operations across different providers."""

    @requires_google_api
    async def test_google_batch_single_request(self):
        """Test Google batch submission with a single request."""
        model = GoogleModel("gemini-2.5-flash", config=LLMConfig(supports_batch=True))

        if not model.batch:
            pytest.skip("Batch not supported for this model")

        request = await model.batch.create_batch_query_request(
            custom_id="test-1",
            input=[TextInput(text="What is 2 + 2? Reply with just the number.")],
        )

        batch_name = f"test-batch-{uuid.uuid4().hex[:8]}"
        job_id = await model.batch.batch_query(batch_name, [request])

        assert job_id is not None
        assert isinstance(job_id, str)
        assert len(job_id) > 0

        try:
            status = await model.batch.get_batch_status(job_id)
            assert status is not None
            print(
                f"Batch job created successfully with ID: {job_id}, initial status: {status}"
            )
        except Exception as e:
            print(f"Batch job created with ID: {job_id}, status check failed: {e}")

        try:
            await model.batch.cancel_batch_request(job_id)
        except Exception:
            pass

    @requires_google_api
    async def test_google_batch_multiple_requests(self):
        """Test Google batch submission with multiple requests."""
        model = GoogleModel(
            "gemini-2.5-flash-lite", config=LLMConfig(supports_batch=True)
        )

        if not model.batch:
            pytest.skip("Batch not supported for this model")

        test_cases = [
            ("math-1", "What is 10 + 15? Reply with just the number."),
            (
                "capital-1",
                "What is the capital of France? Reply with just the city name.",
            ),
            (
                "color-1",
                "What color is the sky on a clear day? Reply with just the color.",
            ),
        ]

        requests: list[dict[str, Any]] = []

        for custom_id, prompt in test_cases:
            request = await model.batch.create_batch_query_request(
                custom_id=custom_id,
                input=[TextInput(text=prompt)],
            )
            requests.append(request)

        batch_name = f"test-multi-{uuid.uuid4().hex[:8]}"
        job_id = await model.batch.batch_query(batch_name, requests)

        assert job_id is not None
        assert isinstance(job_id, str)
        assert len(job_id) > 0
        assert len(requests) == 3

        print(
            f"Batch job with {len(requests)} requests created successfully, ID: {job_id}"
        )

        await model.batch.cancel_batch_request(job_id)

    @requires_openai_api
    async def test_openai_batch_single_request(self):
        """Test OpenAI batch submission with a single request."""
        model = OpenAIModel("gpt-4o-mini", config=LLMConfig(supports_batch=True))

        if not model.batch:
            pytest.skip("Batch not supported for this model")

        request = await model.batch.create_batch_query_request(
            custom_id="openai-test-1",
            input=[TextInput(text="What is 3 + 3? Reply with just the number.")],
        )

        batch_name = f"openai-test-{uuid.uuid4().hex[:8]}"
        job_id = await model.batch.batch_query(batch_name, [request])

        assert job_id is not None
        assert isinstance(job_id, str)
        assert len(job_id) > 0

        print(f"OpenAI batch job created successfully with ID: {job_id}")

    @requires_google_api
    async def test_google_batch_with_reasoning(self):
        """Test batch submission with reasoning enabled."""
        model = GoogleModel(
            "gemini-2.5-flash-lite",
            config=LLMConfig(reasoning=True, supports_batch=True),
        )

        if not model.batch:
            pytest.skip("Batch not supported for this model")

        request = await model.batch.create_batch_query_request(
            custom_id="reasoning-test",
            input=[TextInput(text="Step by step, calculate: (15 * 4) + (12 / 3) - 7")],
            thinking_budget=-1,
        )

        assert "request" in request
        assert "generationConfig" in request["request"]

        batch_name = f"test-reasoning-{uuid.uuid4().hex[:8]}"
        job_id = await model.batch.batch_query(batch_name, [request])

        assert job_id is not None
        assert isinstance(job_id, str)
        assert len(job_id) > 0

        print(f"Batch job with reasoning created successfully, ID: {job_id}")

        try:
            await model.batch.cancel_batch_request(job_id)
        except Exception:
            pass

    @requires_anthropic_api
    async def test_anthropic_batch_single_request(self):
        """Test Anthropic batch submission with a single request."""
        model = AnthropicModel(
            "claude-sonnet-4-20250514", config=LLMConfig(supports_batch=True)
        )

        if not model.batch:
            pytest.skip("Batch not supported for this model")

        request = await model.batch.create_batch_query_request(
            custom_id="anthropic-test-1",
            input=[TextInput(text="What is 5 + 7? Reply with just the number.")],
        )

        batch_name = f"anthropic-test-{uuid.uuid4().hex[:8]}"
        batch_id = await model.batch.batch_query(batch_name, [request])

        assert batch_id is not None
        assert isinstance(batch_id, str)
        assert len(batch_id) > 0

        print(f"Anthropic batch job created successfully with ID: {batch_id}")

        # Test status checking
        status = await model.batch.get_batch_status(batch_id)
        assert status is not None
        print(f"Batch status: {status}")

    @requires_anthropic_api
    async def test_anthropic_batch_with_thinking(self):
        """Test Anthropic batch submission with extended thinking enabled."""
        model = AnthropicModel(
            "claude-sonnet-4-20250514",
            config=LLMConfig(reasoning=True, supports_batch=True),
        )

        if not model.batch:
            pytest.skip("Batch not supported for this model")

        request = await model.batch.create_batch_query_request(
            custom_id="thinking-test",
            input=[TextInput(text="Step by step, calculate: (25 * 8) + (36 / 4) - 15")],
        )

        # Verify the request has thinking enabled
        assert "params" in request
        assert "thinking" in request["params"]

        batch_name = f"anthropic-thinking-{uuid.uuid4().hex[:8]}"
        batch_id = await model.batch.batch_query(batch_name, [request])

        assert batch_id is not None
        assert isinstance(batch_id, str)
        assert len(batch_id) > 0

        print(f"Anthropic batch job with thinking created successfully, ID: {batch_id}")

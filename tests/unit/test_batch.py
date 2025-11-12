import json
from unittest.mock import MagicMock, patch

import pytest
from google.genai.types import JobState

from model_library.base import LLMConfig, TextInput
from model_library.providers.google import GoogleModel
from model_library.providers.google.batch import GoogleBatchMixin


@pytest.mark.unit
class TestBatch:
    @staticmethod
    def async_mock(return_value=None):
        """Helper to create async mock."""

        async def _mock(*args, **kwargs):
            return return_value

        return _mock

    @pytest.fixture
    def mock_batch_job(self):
        job = MagicMock()
        job.name = "projects/test-project/locations/us-central1/batchPredictionJobs/456"
        job.state = MagicMock()
        job.state.name = "JOB_STATE_PENDING"
        return job

    @pytest.fixture
    def mock_google_client(self, mock_batch_job):
        """Mock the Vertex AI client used by batch operations."""
        with (
            patch(
                "model_library.providers.google.GoogleModel.get_client"
            ) as mock_google,
        ):
            client = MagicMock()

            # Setup sync methods
            client.batches.create = MagicMock(return_value=mock_batch_job)
            client.batches.get = MagicMock(return_value=mock_batch_job)
            client.batches.cancel = MagicMock()

            # Setup async methods with default async mocks
            client.aio.batches.create = TestBatch.async_mock(mock_batch_job)
            client.aio.batches.get = TestBatch.async_mock(mock_batch_job)
            client.aio.batches.cancel = TestBatch.async_mock(None)
            client.aio.files.download = TestBatch.async_mock(b"")

            # Both mocks return the same client instance
            mock_google.return_value = client

            yield mock_google

    async def test_batch_request_creation(self, mock_google_client):
        model = GoogleModel("gemini-2.5-flash", config=LLMConfig(supports_batch=True))
        model.temperature = 0.7
        model.max_tokens = 100
        batch = GoogleBatchMixin(model)
        request = await batch.create_batch_query_request(
            custom_id="test-123",
            input=[TextInput(text="Test prompt")],
        )

        assert "request" in request
        assert "labels" in request["request"]
        assert request["request"]["labels"]["qa_pair_id"] == "test-123"
        assert "contents" in request["request"]
        assert "generationConfig" in request["request"]
        assert request["request"]["generationConfig"]["temperature"] == 0.7
        assert request["request"]["generationConfig"]["maxOutputTokens"] == 100

    async def test_batch_submission(self, mock_google_client, mock_batch_job):
        # Setup async mock for batch creation
        mock_google_client.return_value.aio.batches.create = self.async_mock(
            mock_batch_job
        )
        mock_batch_job.state = JobState.JOB_STATE_SUCCEEDED

        model = GoogleModel("gemini-2.5-flash", config=LLMConfig(supports_batch=True))
        batch = GoogleBatchMixin(model)

        requests = [
            {
                "request": {
                    "labels": {"qa_pair_id": "test-1"},
                    "model": "gemini-2.5-flash",
                    "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
                    "generationConfig": {"temperature": 0.7},
                }
            }
        ]

        job_name = await batch.batch_query("test-batch", requests)

        assert job_name == mock_batch_job.name

    async def test_batch_status_states(self, mock_google_client, mock_batch_job):
        model = GoogleModel("gemini-2.5-flash", config=LLMConfig(supports_batch=True))
        batch = GoogleBatchMixin(model)

        test_states = [
            ("JOB_STATE_PENDING", False, False),
            ("JOB_STATE_RUNNING", False, False),
            ("JOB_STATE_SUCCEEDED", True, False),
            ("JOB_STATE_FAILED", True, True),
            ("JOB_STATE_CANCELLED", True, False),
        ]

        for state_name, is_completed, is_failed in test_states:
            mock_batch_job.state.name = state_name
            # The async mock was already set in the fixture and returns mock_batch_job
            # which we're modifying above, so it should work

            status = await batch.get_batch_status("test-job-id")
            assert status == state_name

            assert batch.is_batch_status_completed(status) == is_completed
            assert batch.is_batch_status_failed(status) == is_failed

    async def test_batch_results_parsing(self, mock_google_client):
        output_jsonl = "\n".join(
            [
                json.dumps(
                    {
                        "key": "request-0",
                        "response": {
                            "candidates": [
                                {"content": {"parts": [{"text": "Response 1"}]}}
                            ],
                            "usageMetadata": {
                                "promptTokenCount": 10,
                                "candidatesTokenCount": 5,
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "key": "request-1",
                        "response": {
                            "candidates": [
                                {
                                    "content": {
                                        "parts": [
                                            {"text": "Thinking...", "thought": True},
                                            {"text": "Response 2"},
                                        ]
                                    }
                                }
                            ],
                            "usageMetadata": {
                                "promptTokenCount": 12,
                                "candidatesTokenCount": 8,
                                "thoughtsTokenCount": 3,
                            },
                        },
                    }
                ),
            ]
        )

        # Setup mock job before creating the model
        mock_job = MagicMock()
        mock_job.dest = MagicMock()
        mock_job.dest.file_name = "test-batch/output/predictions.jsonl"
        mock_job.dest.inlined_responses = None
        mock_job.state = JobState.JOB_STATE_SUCCEEDED

        # Update the fixture's async mocks to return our test data
        mock_google_client.return_value.aio.files.download = self.async_mock(
            output_jsonl.encode("utf-8")
        )
        mock_google_client.return_value.aio.batches.get = self.async_mock(mock_job)

        model = GoogleModel("gemini-2.5-flash", config=LLMConfig(supports_batch=True))
        batch = GoogleBatchMixin(model)

        results = await batch.get_batch_results("test-batch")

        assert len(results) == 2
        assert results[0].custom_id == "request-0"
        assert results[0].output.output_text == "Response 1"
        assert results[0].output.reasoning is None
        assert results[0].error_message is None
        assert results[1].custom_id == "request-1"
        assert results[1].output.output_text == "Response 2"
        assert results[1].error_message is None

    @pytest.mark.asyncio
    async def test_batch_error_handling(self, mock_google_client):
        output_jsonl = json.dumps(
            {"key": "request-error", "error": {"message": "Invalid request format"}}
        )

        # Setup mock job before creating the model
        mock_job = MagicMock()
        mock_job.dest = MagicMock()
        mock_job.dest.file_name = "test-batch/output/predictions.jsonl"
        mock_job.dest.inlined_responses = None
        mock_job.state = JobState.JOB_STATE_SUCCEEDED

        # Update the fixture's async mocks to return our test data
        mock_google_client.return_value.aio.files.download = self.async_mock(
            output_jsonl.encode("utf-8")
        )
        mock_google_client.return_value.aio.batches.get = self.async_mock(mock_job)

        model = GoogleModel("gemini-2.5-flash", config=LLMConfig(supports_batch=True))
        batch = GoogleBatchMixin(model)

        results = await batch.get_batch_results("test-batch")

        assert len(results) == 1
        assert results[0].custom_id == "request-error"
        assert results[0].error_message == "Invalid request format"

    async def test_batch_with_reasoning_config(self, mock_google_client):
        model = GoogleModel(
            "gemini-2.5-flash",
            config=LLMConfig(
                reasoning=True, supports_batch=True, temperature=0.5, max_tokens=200
            ),
        )
        batch = GoogleBatchMixin(model)
        request = await batch.create_batch_query_request(
            custom_id="reasoning-test",
            input=[TextInput(text="Complex problem")],
            thinking_budget=8192,
        )

        assert "request" in request
        assert request["request"]["labels"]["qa_pair_id"] == "reasoning-test"
        assert request["request"]["generationConfig"]["maxOutputTokens"] == 200

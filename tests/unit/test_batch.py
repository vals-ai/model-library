import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai.types import JobState

from model_library.base import LLMConfig, TextInput
from model_library.base.input import FileWithBase64
from model_library.providers.google import GoogleModel
from model_library.providers.google.batch import GoogleBatchMixin
from model_library.providers.google.google import GoogleConfig
from model_library.registry_utils import get_registry_model

# TODO: batching is currently out of data, need to update to include
pytest.skip("Batching is currently out of date", allow_module_level=True)


def mock_batch_response():
    response = [
        json.dumps(
            {
                "key": "request-0",
                "response": {
                    "candidates": [{"content": {"parts": [{"text": "Response 1"}]}}],
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
                                    {
                                        "text": "Thinking...",
                                        "thought": True,
                                    },
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
    return "\n".join(response).encode("utf-8")


@pytest.fixture()
def mock_batch_job() -> MagicMock:
    job: MagicMock = MagicMock()
    job.name = "projects/test-project/locations/us-central1/batchPredictionJobs/456"
    job.state = MagicMock()
    job.state.name = JobState.JOB_STATE_PENDING
    return job


@pytest.fixture(autouse=True)
def mock_google_client(mock_batch_job: MagicMock):
    with (
        patch("model_library.providers.google.GoogleModel.get_client") as mock_google,
    ):
        client = MagicMock()

        # batch methods
        client.batches.create = MagicMock(return_value=mock_batch_job)
        client.batches.get = MagicMock(return_value=mock_batch_job)
        client.batches.cancel = MagicMock()

        # async batch methods
        client.aio.batches.create = AsyncMock(return_value=mock_batch_job)
        client.aio.batches.get = AsyncMock(return_value=mock_batch_job)
        client.aio.batches.cancel = AsyncMock(return_value=None)
        client.aio.files.download = AsyncMock(return_value=mock_batch_response())

        mock_google.return_value = client

        yield mock_google


async def test_google_vertex_no_batch():
    model = get_registry_model(
        "google/gemini-2.5-flash",
        override_config=LLMConfig(
            temperature=0.5,
            max_tokens=100,
            provider_config=GoogleConfig(use_vertex=True),
        ),
    )
    assert model.batch is None


async def test_batch_request_creation_basic():
    model = get_registry_model(
        "google/gemini-2.5-flash",
        override_config=LLMConfig(
            temperature=0.5,
            max_tokens=100,
        ),
    )
    assert model.batch

    request = await model.batch.create_batch_query_request(
        custom_id="test-1",
        input=[
            TextInput(text="Test prompt"),
            FileWithBase64(
                type="image", name="test.png", mime="image/png", base64="dGVzdA=="
            ),
        ],
    )

    assert request["request"]["labels"]["qa_pair_id"] == "test-1"
    assert request["request"]["contents"][0]["role"] == "user"

    assert request["request"]["contents"][0]["parts"][0]["text"] == "Test prompt"
    assert (
        request["request"]["contents"][0]["parts"][1]["inlineData"]["mimeType"]
        == "image/image/png"
    )
    assert (
        request["request"]["contents"][0]["parts"][1]["inlineData"]["data"]
        == "dGVzdA=="
    )

    assert request["request"]["generationConfig"]["temperature"] == 0.5
    assert request["request"]["generationConfig"]["maxOutputTokens"] == 100
    assert len(request["request"]["safetySettings"]) == 4


async def test_batch_request_creation_advanced():
    # TODO: when batch tool inputs are added for batch
    return


async def test_batch_submission(mock_batch_job: MagicMock):
    model = get_registry_model(
        "google/gemini-2.5-flash",
        override_config=LLMConfig(
            temperature=0.5,
            max_tokens=100,
        ),
    )
    assert model.batch

    request1 = await model.batch.create_batch_query_request(
        custom_id="test-1",
        input=[
            TextInput(text="Test prompt"),
            FileWithBase64(
                type="image", name="test.png", mime="image/png", base64="dGVzdA=="
            ),
        ],
    )
    request2 = await model.batch.create_batch_query_request(
        custom_id="test-2",
        input=[
            TextInput(text="Test prompt"),
            FileWithBase64(
                type="image", name="test.png", mime="image/png", base64="dGVzdA=="
            ),
        ],
    )
    batch_id = await model.batch.batch_query("test-batch", [request1, request2])

    assert (
        mock_batch_job.name
        == "projects/test-project/locations/us-central1/batchPredictionJobs/456"
    )

    status = await model.batch.get_batch_status(batch_id)
    assert status == "JOB_STATE_PENDING"

    mock_batch_job.state = JobState.JOB_STATE_RUNNING
    status = await model.batch.get_batch_status(batch_id)
    assert status == JobState.JOB_STATE_RUNNING

    mock_batch_job.state = JobState.JOB_STATE_SUCCEEDED
    status = await model.batch.get_batch_status(batch_id)
    assert status == JobState.JOB_STATE_SUCCEEDED

    results = await model.batch.get_batch_results(batch_id)

    assert len(results) == 2

    # first
    # assert results[0].custom_id == "request-0"
    # assert results[0].output.output_text == "Response 1"
    # assert results[0].output.reasoning == None
    # assert results[0].output.metadata.in_tokens == 10
    # assert results[0].output.metadata.out_tokens == 5
    # assert results[0].error_message is None
    #
    # # second
    # assert results[1].custom_id == "request-1"
    # assert results[1].output.output_text == "Response 2"
    # assert results[1].output.reasoning == None
    # assert results[1].output.metadata.in_tokens == 12
    # assert results[1].output.metadata.out_tokens == 8
    # assert results[1].error_message is None
    #


@pytest.mark.parametrize(
    "state_name,is_completed,is_failed",
    [
        ("JOB_STATE_PENDING", False, False),
        ("JOB_STATE_RUNNING", False, False),
        ("JOB_STATE_SUCCEEDED", True, False),
        ("JOB_STATE_FAILED", True, True),
        ("JOB_STATE_CANCELLED", True, False),
    ],
)
async def test_batch_status_states(mock_batch_job, state_name, is_completed, is_failed):
    model = GoogleModel("gemini-2.5-flash", config=LLMConfig(supports_batch=True))
    batch = GoogleBatchMixin(model)

    mock_batch_job.state.name = state_name

    status = await batch.get_batch_status("batches/mock-batch-job-id")
    assert status == state_name
    assert batch.is_batch_status_completed(status) == is_completed
    assert batch.is_batch_status_failed(status) == is_failed


async def test_batch_results_parsing(mock_google_client):
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
    mock_google_client.return_value.aio.files.download = AsyncMock(
        return_value=output_jsonl.encode("utf-8")
    )
    mock_google_client.return_value.aio.batches.get = AsyncMock(return_value=mock_job)

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
async def test_batch_error_handling(mock_google_client):
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
    mock_google_client.return_value.aio.files.download = AsyncMock(
        return_value=output_jsonl.encode("utf-8")
    )
    mock_google_client.return_value.aio.batches.get = AsyncMock(return_value=mock_job)

    model = GoogleModel("gemini-2.5-flash", config=LLMConfig(supports_batch=True))
    batch = GoogleBatchMixin(model)

    results = await batch.get_batch_results("test-batch")

    assert len(results) == 1
    assert results[0].custom_id == "request-error"
    assert results[0].error_message == "Invalid request format"


async def test_batch_with_reasoning_config(mock_google_client):
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

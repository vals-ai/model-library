import io
import json
from typing import TYPE_CHECKING, Any, Final, Sequence, cast

from typing_extensions import override

from google.genai.types import (
    BatchJob,
    Content,
    GenerateContentConfig,
    JobState,
    UploadFileConfig,
)
from model_library.base import BatchResult, InputItem, LLMBatchMixin

if TYPE_CHECKING:
    from model_library.providers.google.google import GoogleModel

import base64

from google.genai.types import (
    Part,
    SafetySetting,
)


def extract_text_from_json_response(response: dict[str, Any]) -> str:
    """Extract concatenated non-thought text from a JSON response structure."""
    # TODO: fix the typing we always ignore
    text = ""
    for candidate in response.get("candidates", []) or []:  # type: ignore
        content = (candidate or {}).get("content") or {}  # type: ignore
        for part in content.get("parts", []) or []:  # type: ignore
            if not part.get("thought", False):  # type: ignore
                text += part.get("text", "")  # type: ignore
    return text  # type: ignore


def parse_predictions_jsonl(jsonl: str) -> list[BatchResult]:
    """Parse Google batch predictions JSONL into BatchResult list."""
    from model_library.base import BatchResult, QueryResult

    results: list[BatchResult] = []
    for line in jsonl.strip().split("\n"):
        if not line:
            continue
        data = json.loads(line)
        custom_id = data.get("key", "unknown")
        if "response" in data:
            response = data["response"]
            text = extract_text_from_json_response(response)
            output = QueryResult()
            output.output_text = text
            if "usageMetadata" in response:
                output.metadata.in_tokens = response["usageMetadata"].get(
                    "promptTokenCount", 0
                )
                output.metadata.out_tokens = response["usageMetadata"].get(
                    "candidatesTokenCount", 0
                )
            results.append(BatchResult(custom_id=custom_id, output=output))
        else:
            error = data.get("error", {}).get("message", "Unknown error")
            results.append(
                BatchResult(
                    custom_id=custom_id,
                    output=QueryResult(output_text=""),
                    error_message=error,
                )
            )
    return results


def serialize_part_to_json(p: Part) -> dict[str, Any]:
    if p.text:
        return {"text": p.text}

    inline = p.inline_data
    if inline is not None:
        mime_type = inline.mime_type
        data_val: Any = inline.data
        if isinstance(data_val, (bytes, bytearray, memoryview)):
            data_val = base64.b64encode(cast(bytes, data_val)).decode("utf-8")
        return {"inlineData": {"mimeType": mime_type, "data": data_val}}

    return {}


def serialize_content_to_json(c: Content) -> dict[str, Any]:
    ser_parts = [serialize_part_to_json(p) for p in (c.parts or [])]
    return {"role": c.role or "user", "parts": ser_parts}


def serialize_safety_settings(
    safety: list[SafetySetting] | None,
) -> list[dict[str, Any]]:
    if not safety:
        return []

    items: list[dict[str, Any]] = []
    for s in safety:
        items.append({"category": s.category, "threshold": s.threshold})
    return items


def build_generation_config(cfg: GenerateContentConfig) -> dict[str, Any]:
    gen: dict[str, Any] = {}

    if cfg.max_output_tokens is not None:
        gen["maxOutputTokens"] = cfg.max_output_tokens
    if cfg.temperature is not None:
        gen["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        gen["topP"] = cfg.top_p

    thinking_cfg = cfg.thinking_config
    if thinking_cfg is not None and thinking_cfg.thinking_budget is not None:
        gen["thinkingConfig"] = {"thinkingBudget": thinking_cfg.thinking_budget}

    return gen


class GoogleBatchMixin(LLMBatchMixin):
    COMPLETED_STATES: Final[set[str]] = {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_PAUSED,
        JobState.JOB_STATE_EXPIRED,
    }

    FAILED_STATE = JobState.JOB_STATE_FAILED
    CANCELLED_STATE = JobState.JOB_STATE_CANCELLED
    BATCH_STATUS_TIMEOUT: Final[int] = 30

    def __init__(self, google: "GoogleModel"):
        self._root = google

    @override
    async def create_batch_query_request(
        self,
        custom_id: str,
        input: Sequence[InputItem],
        **kwargs: object,
    ) -> dict[str, Any]:
        self._root.logger.debug(f"Creating batch request for custom_id: {custom_id}")
        body = await self._root.create_body(input, tools=[], **kwargs)

        contents_any = body["contents"]
        serialized_contents: list[dict[str, Any]] = [
            serialize_content_to_json(content)
            for content in contents_any
            if isinstance(content, Content)
        ]

        config_obj = cast(GenerateContentConfig, body.pop("config"))
        gen_cfg: dict[str, Any] = build_generation_config(config_obj)

        formatted_body: dict[str, Any] = {
            "contents": serialized_contents,
            "generationConfig": gen_cfg,
        }

        if config_obj.safety_settings:
            formatted_body["safetySettings"] = serialize_safety_settings(
                config_obj.safety_settings
            )

        system_instruction = config_obj.system_instruction
        if system_instruction:
            formatted_body["systemInstruction"] = {
                "role": "system",
                "parts": [{"text": str(system_instruction)}],
            }

        return {
            "request": {
                "labels": {"qa_pair_id": custom_id},
                **formatted_body,
            }
        }

    @override
    async def batch_query(
        self,
        batch_name: str,
        requests: list[dict[str, Any]],
    ) -> str:
        self._root.logger.info(
            f"Starting batch query '{batch_name}' with {len(requests)} requests"
        )
        jsonl_lines: list[str] = []
        for i, req in enumerate(requests):
            request_data = req.get("request", req)
            labels = request_data.pop("labels", {})
            custom_id = labels.get("qa_pair_id", f"request-{i}")
            jsonl_lines.append(json.dumps({"key": custom_id, "request": request_data}))

        batch_request_file = self._root.client.files.upload(
            file=io.StringIO("\n".join(jsonl_lines)),
            config=UploadFileConfig(mime_type="application/jsonl"),
        )

        if not batch_request_file.name:
            raise Exception("Failed to upload batch jsonl")

        try:
            job: BatchJob = await self._root.client.aio.batches.create(
                model=self._root.model_name,
                src=batch_request_file.name,
                config={"display_name": batch_name},
            )
        except Exception as e:
            self._root.logger.error(f"Error creating batch job: {e}")
            raise

        if not job.name:
            raise Exception("Failed to create batch job - no job name returned")

        self._root.logger.info(f"Batch job created successfully: {job.name}")
        return job.name

    @override
    async def get_batch_results(self, batch_id: str) -> list[BatchResult]:
        self._root.logger.info(f"Retrieving batch results for {batch_id}")

        job = await self._root.client.aio.batches.get(name=batch_id)

        results: list[BatchResult] = []

        if job.state == JobState.JOB_STATE_SUCCEEDED:
            if job.dest and job.dest.file_name:
                results_file_name = job.dest.file_name
                file_content = await self._root.client.aio.files.download(
                    file=results_file_name
                )
                decoded = file_content.decode("utf-8")
                results.extend(parse_predictions_jsonl(decoded))
            else:
                self._root.logger.warning(f"No results found for batch {batch_id}")

        else:
            self._root.logger.error(f"Batch {batch_id} did not succeed: {job.state}")
            if job.error:
                self._root.logger.error(f"Error: {job.error}")

        self._root.logger.info(f"Retrieved {len(results)} batch results")
        return results

    @override
    async def cancel_batch_request(self, batch_id: str):
        self._root.logger.info(f"Cancelling batch {batch_id}")
        await self._root.client.aio.batches.cancel(name=batch_id)

    @override
    async def get_batch_progress(self, batch_id: str) -> int:
        return 0

    @override
    async def get_batch_status(self, batch_id: str) -> str:
        import asyncio

        try:
            self._root.logger.debug(f"Checking batch status for {batch_id}")
            job: BatchJob = await self._root.client.aio.batches.get(name=batch_id)
            state = job.state

            if not state:
                raise Exception("Failed to get batch job status")

            return state.name
        except asyncio.TimeoutError:
            self._root.logger.error(
                f"Timeout getting status for batch {batch_id} after {self.BATCH_STATUS_TIMEOUT}s"
            )
            raise
        except Exception as e:
            import traceback

            self._root.logger.error(
                f"Error getting batch status for {batch_id}: {e}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            raise

    @override
    @classmethod
    def is_batch_status_completed(cls, batch_status: str) -> bool:
        return batch_status in cls.COMPLETED_STATES

    @override
    @classmethod
    def is_batch_status_failed(cls, batch_status: str) -> bool:
        return batch_status == cls.FAILED_STATE

    @override
    @classmethod
    def is_batch_status_cancelled(cls, batch_status: str) -> bool:
        return batch_status == cls.CANCELLED_STATE

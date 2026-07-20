"""GatewayLLM — routes queries through a remote model-proxy server."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import random
from collections.abc import Generator, Mapping, Sequence
from typing import Any, Literal, NoReturn, cast

import httpx
import model_library
import model_library.telemetry as telemetry
from openai.lib._pydantic import to_strict_json_schema
from openai.types.moderation_create_response import ModerationCreateResponse
from pydantic import BaseModel
from typing_extensions import override

from model_library.base.base import (
    LLM,
    LLMConfig,
    TokenRetryParams,
)
from model_library.base.query_ids import resolve_query_ids
from model_library.base.query_logging import (
    log_query_completed,
    log_query_started,
    scoped_query_logger,
)
from model_library.base.input import (
    FileInput,
    FileWithId,
    InputItem,
    RawInput,
    RawResponse,
    ToolDefinition,
    normalize_query_input,
)
from model_library.base.output import QueryResult, RateLimit
from model_library.exceptions import GatewayMethodNotSupported, GatewayProviderError
from model_library.utils import gateway_httpx_client


GATEWAY_HTTP_MAX_ATTEMPTS = 8
GATEWAY_HTTP_RETRY_INITIAL_SECONDS = 0.5
GATEWAY_HTTP_RETRY_MAX_SECONDS = 30.0
GATEWAY_RETRY_LOGGER = logging.getLogger("llm.gateway")


def _dump_items(items: Sequence[InputItem]) -> list[dict[str, Any]]:
    """Dump InputItems to JSON-ready dicts.

    Normal items use model_dump(). RawResponse/RawInput are built manually
    because their fields hold opaque blobs (base64+HMAC from the server)
    that must be echoed as-is — model_dump() would fail on provider SDK objects.
    """
    result: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, RawResponse):
            result.append({"kind": "raw_response", "response": item.response})
        elif isinstance(item, RawInput):
            result.append({"kind": "raw_input", "input": item.input})
        else:
            result.append(item.model_dump())
    return result


def _request_items(items: Sequence[InputItem]) -> list[InputItem]:
    return cast(list[InputItem], _dump_items(items))


def _request_tools(tools: list[ToolDefinition]) -> list[ToolDefinition]:
    return cast(list[ToolDefinition], [t.model_dump() for t in tools])


def _clean_optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dump_request(request: BaseModel) -> dict[str, Any]:
    data = request.model_dump(mode="json", exclude_none=True)
    config_dict = getattr(request, "config_dict", None)
    if callable(config_dict):
        data["config"] = config_dict()
    return data


def _gateway_correlation_headers(body: Mapping[str, Any]) -> dict[str, str]:
    return {
        "X-Run-Id": str(body["run_id"])[:128],
        "X-Question-Id": str(body["question_id"])[:128],
        "X-Query-Id": str(body["query_id"])[:128],
    }


def _gateway_retry_log_ids(
    body: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None, str | None]:
    identity = body.get("identity")
    return (
        _clean_optional(body.get("run_id")),
        _clean_optional(body.get("question_id")),
        _clean_optional(body.get("query_id")),
        json.dumps(identity, sort_keys=True, separators=(",", ":"))
        if identity is not None
        else None,
    )


def _gateway_error_detail(resp: httpx.Response) -> str:
    detail = resp.text
    try:
        error_body_obj: object = resp.json()
    except Exception:
        return detail

    if not isinstance(error_body_obj, dict):
        return detail

    error_body = cast(dict[str, object], error_body_obj)
    raw_detail = error_body.get("message")
    if raw_detail is None:
        raw_detail = error_body.get("detail")
    if isinstance(raw_detail, str):
        detail = raw_detail
    return detail


def _is_retryable_gateway_response(resp: httpx.Response) -> bool:
    return resp.status_code == 429 or resp.status_code >= 500


def _raise_for_gateway_error_envelope(data: dict[str, Any]) -> None:
    if "error" not in data:
        return

    raw_error = data["error"]
    if not isinstance(raw_error, dict):
        raise GatewayProviderError(
            error_type="GatewayError",
            code="malformed_error_envelope",
            message=f"Gateway response error: {raw_error}",
            provider=None,
            raw_error=raw_error,
        )

    error = cast(dict[str, Any], raw_error)
    raw_type = error.get("type")
    raw_code = error.get("code")
    raw_message = error.get("message")
    raw_provider = error.get("provider")
    raw_exception_type = error.get("exception_type")
    raw_status_code = error.get("status_code")
    error_type = raw_type if isinstance(raw_type, str) else "GatewayError"
    code = raw_code if isinstance(raw_code, str) else None
    message = raw_message if isinstance(raw_message, str) else "Unknown gateway error"
    provider = raw_provider if isinstance(raw_provider, str) else None
    exception_type = raw_exception_type if isinstance(raw_exception_type, str) else None
    status_code = (
        raw_status_code
        if isinstance(raw_status_code, int) and not isinstance(raw_status_code, bool)
        else None
    )
    raise GatewayProviderError(
        error_type=error_type,
        code=code,
        message=message,
        provider=provider,
        raw_error=error,
        exception_type=exception_type,
        status_code=status_code,
    )


def _decode_gateway_success(resp: httpx.Response) -> dict[str, Any]:
    try:
        raw_data: object = resp.json()
    except ValueError as exc:
        raise GatewayProviderError(
            error_type="GatewayError",
            code="malformed_gateway_response",
            message="Gateway response must be a JSON object",
            provider=None,
            raw_error=resp.text,
        ) from exc
    if not isinstance(raw_data, dict):
        raise GatewayProviderError(
            error_type="GatewayError",
            code="malformed_gateway_response",
            message="Gateway response must be a JSON object",
            provider=None,
            raw_error=raw_data,
        )
    data = cast(dict[str, Any], raw_data)
    _raise_for_gateway_error_envelope(data)
    return data


def _parse_query_result(
    data: dict[str, Any], schema_model: type[BaseModel] | None
) -> QueryResult:
    result_data = dict(data)

    # Raw fields stay as base64+HMAC blobs. The client decodes the signed JSON
    # structure but never restores provider-native objects.
    history_json = result_data.pop("signed_history", None)
    history_items: list[InputItem] = []
    if history_json:
        from pydantic import TypeAdapter

        adapter = TypeAdapter(list[InputItem])
        history_items = list(adapter.validate_json(history_json))
    result_data["history"] = history_items

    output_parsed: dict[str, Any] | BaseModel | None = result_data.get("output_parsed")
    if schema_model is not None:
        if output_parsed is not None:
            result_data["output_parsed"] = schema_model.model_validate(output_parsed)
        elif result_data.get("output_text"):
            result_data["output_parsed"] = schema_model.model_validate_json(
                result_data["output_text"]
            )

    finish_reason = result_data.get("finish_reason")
    if finish_reason is None:
        result_data.pop("finish_reason", None)
    else:
        if not isinstance(finish_reason, dict):
            raise TypeError("Gateway finish_reason must be an object")
        finish_reason_data = dict(cast(dict[str, object], finish_reason))
        finish_reason_data.setdefault("raw", None)
        result_data["finish_reason"] = finish_reason_data

    return QueryResult.model_validate(result_data)


def _gateway_retry_delay_seconds(attempt: int) -> float:
    capped = min(
        GATEWAY_HTTP_RETRY_INITIAL_SECONDS * (2**attempt),
        GATEWAY_HTTP_RETRY_MAX_SECONDS,
    )
    return random.uniform(0, capped)


async def _sleep_before_gateway_retry(
    attempt: int, *, delay_seconds: float | None = None
) -> None:
    delay = (
        _gateway_retry_delay_seconds(attempt)
        if delay_seconds is None
        else delay_seconds
    )
    await asyncio.sleep(delay)


class GatewayUnsupportedBatch:
    def __getattr__(self, name: str) -> NoReturn:
        raise GatewayMethodNotSupported(
            "batch is not supported in gateway mode; use the gateway query API instead."
        )


class GatewayLLM(LLM):
    """LLM that forwards all queries to a remote model-proxy server.

    Does not initialize any provider client or validate API keys locally.
    The server handles model instantiation, retries, and provider auth.

    Usage:
        llm = GatewayLLM("gpt-4o", "openai")
        result = await llm.query([TextInput(text="hi")])
    """

    gateway_mode: bool = True

    def _get_default_api_key(self) -> str:
        return model_library.model_library_settings.MODEL_GATEWAY_API_KEY

    @override
    def _client_initialization(self, config: LLMConfig) -> tuple[str, str | None]:
        return self._get_default_api_key(), None

    @override
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> httpx.AsyncClient:
        if not self.has_client():
            assert api_key
            self.assign_client(
                gateway_httpx_client(
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
            )
        return super().get_client()

    def __init__(
        self,
        model_name: str,
        provider: str,
        *,
        config: LLMConfig | None = None,
        identity: object | None = None,
    ) -> None:
        self.gateway_config = config if config is not None else LLMConfig()
        self.identity = (
            telemetry.normalize_identity(identity) if identity is not None else None
        )
        super().__init__(model_name, provider, config=config)
        self.provider_config = self.gateway_config.provider_config
        if self.supports_batch:
            self.batch = cast(Any, GatewayUnsupportedBatch())

    @property
    def gateway_model_key(self) -> str:
        return self._registry_key or f"{self.provider}/{self.model_name}"

    async def post_gateway(
        self,
        path: str,
        request_body: BaseModel | dict[str, Any],
        *,
        timeout: float | httpx.Timeout | None = None,
    ) -> dict[str, Any]:
        body = (
            _dump_request(request_body)
            if isinstance(request_body, BaseModel)
            else request_body
        )
        client: httpx.AsyncClient = self.get_client()
        request_timeout = client.timeout if timeout is None else timeout
        url = f"{model_library.model_library_settings.MODEL_GATEWAY_URL.rstrip('/')}{path}"

        headers = _gateway_correlation_headers(body) if path == "/query" else None
        run_id, question_id, query_id, identity = _gateway_retry_log_ids(body)
        for attempt in range(GATEWAY_HTTP_MAX_ATTEMPTS):
            is_last_attempt = attempt == GATEWAY_HTTP_MAX_ATTEMPTS - 1
            try:
                resp = await client.post(
                    url,
                    content=json.dumps(body, default=str),
                    headers=headers,
                    timeout=request_timeout,
                )
            except httpx.TransportError as exc:
                if is_last_attempt:
                    raise
                delay = _gateway_retry_delay_seconds(attempt)
                GATEWAY_RETRY_LOGGER.warning(
                    "gateway_http_retry path=%s attempt=%s max_attempts=%s "
                    "run_id=%s question_id=%s query_id=%s identity=%s "
                    "error_type=%s retry_after_s=%.3f",
                    path,
                    attempt + 1,
                    GATEWAY_HTTP_MAX_ATTEMPTS,
                    run_id,
                    question_id,
                    query_id,
                    identity,
                    type(exc).__name__,
                    delay,
                )
                await _sleep_before_gateway_retry(attempt, delay_seconds=delay)
                continue

            if resp.status_code == 200:
                return _decode_gateway_success(resp)

            if _is_retryable_gateway_response(resp) and not is_last_attempt:
                delay = _gateway_retry_delay_seconds(attempt)
                GATEWAY_RETRY_LOGGER.warning(
                    "gateway_http_retry path=%s attempt=%s max_attempts=%s "
                    "run_id=%s question_id=%s query_id=%s identity=%s "
                    "status_code=%s retry_after_s=%.3f",
                    path,
                    attempt + 1,
                    GATEWAY_HTTP_MAX_ATTEMPTS,
                    run_id,
                    question_id,
                    query_id,
                    identity,
                    resp.status_code,
                    delay,
                )
                await _sleep_before_gateway_retry(attempt, delay_seconds=delay)
                continue

            detail = _gateway_error_detail(resp)
            raise Exception(f"Gateway error ({resp.status_code}): {detail}")

        raise AssertionError("unreachable")

    def __rich_repr__(self) -> Generator[tuple[str, Any], None, None]:
        attrs = vars(self).copy()
        attrs.pop("_metadata", None)
        attrs.pop("custom_retrier", None)
        attrs.pop("instance_logger", None)
        attrs.pop("gateway_config", None)
        for name, value in attrs.items():
            yield name, value

    @override
    async def query(
        self,
        input: Sequence[InputItem] | str,
        *,
        history: Sequence[InputItem] = [],
        tools: list[ToolDefinition] = [],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        logger: logging.Logger | None = None,
        **kwargs: object,
    ) -> QueryResult:
        if self.custom_retrier is not None:
            raise GatewayMethodNotSupported(
                "custom_retrier is not supported in gateway mode; "
                "retries run server-side."
            )

        identity = kwargs.pop("identity", None)
        if identity is None:
            identity = self.identity
        raw_identity = _clean_optional(
            model_library.model_library_settings.get("IDENTITY", None)
        )
        if identity is None and raw_identity is not None:
            try:
                identity = telemetry.normalize_identity(json.loads(raw_identity))
            except (json.JSONDecodeError, telemetry.IdentityValidationError):
                identity = None

        run_id, question_id, query_id = resolve_query_ids(
            run_id=kwargs.pop("run_id", None),
            question_id=kwargs.pop("question_id", None),
            query_id=kwargs.pop("query_id", None),
        )
        in_agent = bool(kwargs.pop("in_agent", False))

        base_logger = logger or self.instance_logger.getChild(f"<run={run_id}>")
        query_logger = scoped_query_logger(
            base_logger,
            question_id=question_id,
            query_id=query_id,
            in_agent=in_agent,
        )
        current_input = normalize_query_input(input, kwargs=kwargs)
        all_input = normalize_query_input(current_input, history=history)

        schema_model: type[BaseModel] | None = (
            output_schema
            if output_schema is not None and not isinstance(output_schema, dict)
            else None
        )
        request_body = await self.build_body(
            all_input,
            tools=tools,
            output_schema=output_schema,
            run_id=run_id,
            question_id=question_id,
            query_id=query_id,
            identity=identity,
            in_agent=in_agent,
            **kwargs,
        )

        log_query_started(
            query_logger,
            input=current_input,
            all_input=all_input,
            history=history,
            tools=tools,
            kwargs=kwargs,
        )
        data = await self.post_gateway("/query", request_body)
        result = _parse_query_result(data, schema_model)
        log_query_completed(query_logger, result)
        return result

    @override
    async def init_token_retry(self, token_retry_params: TokenRetryParams) -> None:
        self.token_retry_params = token_retry_params
        self._resolved_token_retry_params = None

    @override
    async def count_tokens(
        self,
        input: Sequence[InputItem],
        *,
        history: Sequence[InputItem] = [],
        tools: list[ToolDefinition] = [],
        **kwargs: object,
    ) -> int:
        from model_gateway.types import TokenCountRequest

        all_input = normalize_query_input(input, history=history, kwargs=kwargs)
        data = await self.post_gateway(
            "/tokens/count",
            TokenCountRequest(
                model=self.gateway_model_key,
                inputs=_request_items(all_input),
                tools=_request_tools(tools),
                config=self.gateway_config,
            ),
        )
        return int(data["tokens"])

    @override
    async def get_rate_limit(self) -> RateLimit | None:
        from model_gateway.types import RateLimitRequest

        data = await self.post_gateway(
            "/rate-limit",
            RateLimitRequest(
                model=self.gateway_model_key,
                config=self.gateway_config,
            ),
        )
        raw_rate_limit = data.get("rate_limit")
        return RateLimit(**raw_rate_limit) if raw_rate_limit else None

    @override
    async def _query_impl(self, *args: Any, **kwargs: Any) -> Any:
        raise GatewayMethodNotSupported

    @override
    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition] = [],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> dict[str, Any]:
        from model_gateway.types import QueryRequest

        gateway_owned_fields = {
            "model",
            "inputs",
            "tools",
            "config",
            "output_schema",
            "token_retry_params",
        }
        supported_kwargs = set(QueryRequest.model_fields) - gateway_owned_fields
        unsupported_kwargs = sorted(set(kwargs) - supported_kwargs)
        if unsupported_kwargs:
            fields = ", ".join(unsupported_kwargs)
            raise GatewayMethodNotSupported(
                f"Gateway query does not support extra parameter(s): {fields}"
            )

        if isinstance(output_schema, dict):
            schema = output_schema
        elif output_schema is None:
            schema = None
        else:
            schema = to_strict_json_schema(output_schema)

        request = QueryRequest.model_validate(
            {
                "model": self.gateway_model_key,
                "inputs": _request_items(input),
                "tools": _request_tools(tools),
                "config": self.gateway_config,
                "output_schema": schema,
                "token_retry_params": self.token_retry_params,
                **kwargs,
            }
        )
        return _dump_request(request)

    @override
    async def parse_input(self, *args: Any, **kwargs: Any) -> Any:
        raise GatewayMethodNotSupported

    @override
    async def parse_image(self, image: FileInput) -> Any:
        raise GatewayMethodNotSupported

    @override
    async def parse_file(self, file: FileInput) -> Any:
        raise GatewayMethodNotSupported

    @override
    async def parse_tools(self, tools: list[ToolDefinition]) -> Any:
        raise GatewayMethodNotSupported

    @override
    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        from model_gateway.types import UploadFileRequest

        data = await self.post_gateway(
            "/files/upload",
            UploadFileRequest(
                model=self.gateway_model_key,
                name=name,
                mime=mime,
                content_base64=base64.b64encode(bytes.getvalue()).decode(),
                type=type,
                config=self.gateway_config,
            ),
        )
        return FileWithId(**data["file"])

    async def get_embedding(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> list[float]:
        from model_gateway.types import EmbeddingRequest

        data = await self.post_gateway(
            "/embeddings",
            EmbeddingRequest(
                model=self.gateway_model_key,
                text=text,
                embedding_model=model,
                config=self.gateway_config,
            ),
        )
        return cast(list[float], data["embedding"])

    async def moderate_content(self, text: str) -> ModerationCreateResponse:
        from model_gateway.types import ModerationRequest

        data = await self.post_gateway(
            "/moderation",
            ModerationRequest(
                model=self.gateway_model_key,
                text=text,
                config=self.gateway_config,
            ),
        )
        return ModerationCreateResponse.model_validate(data["response"])

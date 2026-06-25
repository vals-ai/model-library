"""GatewayLLM — routes queries through a remote model-proxy server."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import random
from collections.abc import Generator, Sequence
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
    dump_gateway_config,
)
from model_library.base.query_ids import resolve_query_ids
from model_library.base.input import (
    FileInput,
    FileWithId,
    InputItem,
    RawInput,
    RawResponse,
    ToolCall,
    ToolDefinition,
    normalize_query_input,
)
from model_library.base.output import (
    FinishReasonInfo,
    QueryResult,
    QueryResultExtras,
    QueryResultMetadata,
    RateLimit,
)
from model_library.exceptions import GatewayMethodNotSupported, GatewayProviderError
from model_library.utils import default_httpx_client


GATEWAY_HTTP_MAX_ATTEMPTS = 3
GATEWAY_HTTP_RETRY_INITIAL_SECONDS = 0.5
GATEWAY_HTTP_RETRY_MAX_SECONDS = 5.0


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


def _gateway_config(config: LLMConfig | None) -> LLMConfig:
    return config if config is not None else LLMConfig()


def _clean_optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _request_id(value: object | None, name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be blank")
    return text


def _dump_request(request: BaseModel) -> dict[str, Any]:
    data = request.model_dump(mode="json", exclude_none=True)
    config_dict = getattr(request, "config_dict", None)
    if callable(config_dict):
        data["config"] = config_dict()
    return data


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
    error_type = raw_type if isinstance(raw_type, str) else "GatewayError"
    code = raw_code if isinstance(raw_code, str) else "unknown"
    message = raw_message if isinstance(raw_message, str) else "Unknown gateway error"
    provider = raw_provider if isinstance(raw_provider, str) else None
    raise GatewayProviderError(
        error_type=error_type,
        code=code,
        message=message,
        provider=provider,
        raw_error=error,
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
    # Parse history items from server response. Raw fields stay as
    # base64+HMAC blobs — the client never unpickles, just echoes them back.
    history_json = data.get("signed_history")
    history_items: list[InputItem] = []
    if history_json:
        from pydantic import TypeAdapter

        adapter = TypeAdapter(list[InputItem])
        history_items = list(adapter.validate_json(history_json))

    output_parsed: dict[str, Any] | BaseModel | None = data.get("output_parsed")
    if schema_model is not None:
        if output_parsed is not None:
            output_parsed = schema_model.model_validate(output_parsed)
        elif data.get("output_text"):
            output_parsed = schema_model.model_validate_json(data["output_text"])

    finish_reason = data.get("finish_reason")
    result_kwargs: dict[str, Any] = {
        "output_text": data.get("output_text"),
        "output_parsed": output_parsed,
        "reasoning": data.get("reasoning"),
        "tool_calls": [ToolCall(**tc) for tc in data.get("tool_calls", [])],
        "history": history_items,
        "metadata": QueryResultMetadata(**data.get("metadata", {})),
        "extras": QueryResultExtras(**data.get("extras", {})),
    }
    if finish_reason is not None:
        if not isinstance(finish_reason, dict):
            raise TypeError("Gateway finish_reason must be an object")
        finish_reason_data = dict(cast(dict[str, object], finish_reason))
        finish_reason_data.setdefault("raw", None)
        result_kwargs["finish_reason"] = FinishReasonInfo.model_validate(
            finish_reason_data
        )

    return QueryResult(**result_kwargs)


async def _sleep_before_gateway_retry(attempt: int) -> None:
    capped = min(
        GATEWAY_HTTP_RETRY_INITIAL_SECONDS * (2**attempt),
        GATEWAY_HTTP_RETRY_MAX_SECONDS,
    )
    await asyncio.sleep(random.uniform(0, capped))


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
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> httpx.AsyncClient:
        if not self.has_client():
            assert api_key
            self.assign_client(
                default_httpx_client(
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
            )
        return super().get_client()

    def __init__(self, model_name: str, provider: str, **kwargs: Any):
        self._gateway_metadata_loaded = True
        self._override_config = kwargs.get("config")
        init_kwargs = dict(kwargs)
        init_config = self._override_config
        if isinstance(init_config, LLMConfig):
            init_kwargs["config"] = LLMConfig(**dump_gateway_config(init_config))
        super().__init__(model_name, provider, **init_kwargs)
        self._gateway_metadata_loaded = False
        self.token_retry_params: TokenRetryParams | None = None
        if self.supports_batch:
            self.batch = cast(Any, GatewayUnsupportedBatch())

    @property
    def gateway_input_context_window(self) -> int | None:
        return self.input_context_window

    @override
    async def ensure_metadata_loaded(self) -> None:
        if self._gateway_metadata_loaded:
            return
        await self.sync_model_metadata()

    async def _post_gateway(
        self, path: str, request_body: BaseModel | dict[str, Any]
    ) -> dict[str, Any]:
        body = (
            _dump_request(request_body)
            if isinstance(request_body, BaseModel)
            else request_body
        )
        client: httpx.AsyncClient = self.get_client()
        url = f"{model_library.model_library_settings.MODEL_GATEWAY_URL.rstrip('/')}{path}"

        for attempt in range(GATEWAY_HTTP_MAX_ATTEMPTS):
            is_last_attempt = attempt == GATEWAY_HTTP_MAX_ATTEMPTS - 1
            try:
                resp = await client.post(
                    url,
                    content=json.dumps(body, default=str),
                )
            except httpx.TransportError:
                if is_last_attempt:
                    raise
                await _sleep_before_gateway_retry(attempt)
                continue

            if resp.status_code == 200:
                return _decode_gateway_success(resp)

            if _is_retryable_gateway_response(resp) and not is_last_attempt:
                await _sleep_before_gateway_retry(attempt)
                continue

            detail = _gateway_error_detail(resp)
            raise Exception(f"Gateway error ({resp.status_code}): {detail}")

        raise AssertionError("unreachable")

    async def aresolve_model(self) -> dict[str, Any]:
        """Resolve gateway-side effective config and registry config for this model."""
        from model_gateway.types import ModelResolveRequest

        return await self._post_gateway(
            "/models/resolve",
            ModelResolveRequest(
                model=f"{self.provider}/{self.model_name}",
                config=_gateway_config(self._override_config),
            ),
        )

    async def sync_model_metadata(self) -> None:
        """Load gateway-authoritative model metadata onto this instance."""
        data = await self.aresolve_model()
        if not data.get("exists", False):
            raise Exception(f"Model {self.provider}/{self.model_name} not found")

        effective_config = cast(dict[str, Any] | None, data.get("effective_config"))
        if effective_config is None:
            raise Exception("Gateway resolve response did not include effective_config")

        registry_config = cast(dict[str, Any] | None, data.get("registry_config"))
        if registry_config is None:
            raise Exception("Gateway resolve response did not include registry_config")

        from model_library.register_models import ModelConfig

        # The gateway returns JSON-shaped registry data. Keep JSON-mode validation
        # so fields such as release_date are coerced from strings.
        metadata = ModelConfig.model_validate_json(json.dumps(registry_config))
        self._metadata = metadata
        self._registry_key = metadata.full_key

        config = LLMConfig.model_validate(effective_config)
        for field, value in dump_gateway_config(
            config, mode="python", exclude_none=False, exclude_unset=False
        ).items():
            setattr(self, field, value)
        self.batch = (
            cast(Any, GatewayUnsupportedBatch()) if self.supports_batch else None
        )
        self._gateway_metadata_loaded = True

    def __rich_repr__(self) -> Generator[tuple[str, Any], None, None]:
        attrs = vars(self).copy()
        if not self._gateway_metadata_loaded:
            for field in dump_gateway_config(
                _gateway_config(None),
                mode="python",
                exclude_none=False,
                exclude_unset=False,
            ):
                if field in attrs:
                    attrs[field] = "<unloaded: call ensure_metadata_loaded()>"
        attrs.pop("_metadata", None)
        attrs.pop("custom_retrier", None)
        attrs.pop("instance_logger", None)
        yield from attrs.items()

    @override
    async def query(
        self,
        input: Sequence[InputItem] | str,
        *,
        history: Sequence[InputItem] = [],
        tools: list[ToolDefinition] = [],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> QueryResult:
        if self.custom_retrier is not None:
            raise GatewayMethodNotSupported(
                "custom_retrier is not supported in gateway mode; "
                "retries run server-side."
            )

        kwargs.pop("logger", None)
        identity = kwargs.pop("identity", None)
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

        all_input = normalize_query_input(input, history=history, kwargs=kwargs)

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

        data = await self._post_gateway("/query", request_body)

        return _parse_query_result(data, schema_model)

    @override
    async def init_token_retry(self, token_retry_params: TokenRetryParams) -> None:
        self.token_retry_params = token_retry_params

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
        data = await self._post_gateway(
            "/tokens/count",
            TokenCountRequest(
                model=f"{self.provider}/{self.model_name}",
                inputs=_request_items(all_input),
                tools=_request_tools(tools),
                config=_gateway_config(self._override_config),
            ),
        )
        return int(data["tokens"])

    @override
    async def get_rate_limit(self) -> RateLimit | None:
        from model_gateway.types import RateLimitRequest

        data = await self._post_gateway(
            "/rate-limit",
            RateLimitRequest(
                model=f"{self.provider}/{self.model_name}",
                config=_gateway_config(self._override_config),
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
        supported_kwargs = {"run_id", "question_id", "query_id", "identity", "in_agent"}
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

        from model_gateway.types import QueryRequest

        identity = kwargs.get("identity")
        run_id = kwargs.get("run_id")
        question_id = kwargs.get("question_id")
        request = QueryRequest(
            model=f"{self.provider}/{self.model_name}",
            inputs=_request_items(input),
            tools=_request_tools(tools),
            config=_gateway_config(self._override_config),
            output_schema=schema,
            run_id=_request_id(run_id, "run_id"),
            question_id=_request_id(question_id, "question_id"),
            query_id=_clean_optional(kwargs.get("query_id")),
            identity=identity,
            in_agent=bool(kwargs.get("in_agent", False)),
            token_retry_params=self.token_retry_params,
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

        data = await self._post_gateway(
            "/files/upload",
            UploadFileRequest(
                model=f"{self.provider}/{self.model_name}",
                name=name,
                mime=mime,
                content_base64=base64.b64encode(bytes.getvalue()).decode(),
                type=type,
                config=_gateway_config(self._override_config),
            ),
        )
        return FileWithId(**data["file"])

    async def get_embedding(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> list[float]:
        from model_gateway.types import EmbeddingRequest

        data = await self._post_gateway(
            "/embeddings",
            EmbeddingRequest(
                model=f"{self.provider}/{self.model_name}",
                text=text,
                embedding_model=model,
                config=_gateway_config(self._override_config),
            ),
        )
        return cast(list[float], data["embedding"])

    async def moderate_content(self, text: str) -> ModerationCreateResponse:
        from model_gateway.types import ModerationRequest

        data = await self._post_gateway(
            "/moderation",
            ModerationRequest(
                model=f"{self.provider}/{self.model_name}",
                text=text,
                config=_gateway_config(self._override_config),
            ),
        )
        return ModerationCreateResponse.model_validate(data["response"])

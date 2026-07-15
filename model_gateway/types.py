"""Wire types for the model proxy server API."""

from typing import Any, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from model_library import telemetry
from model_library.base import LLMConfig, TokenRetryParams, dump_llm_config
from model_library.base.input import FileWithId, InputItem, ToolDefinition
from model_library.base.output import QueryResult


class GatewayRequestBase(BaseModel):
    """Common gateway request fields and config normalization."""

    model_config = ConfigDict(extra="forbid")

    model: str
    config: LLMConfig = Field(default_factory=LLMConfig)

    @model_validator(mode="before")
    @classmethod
    def _normalize_config(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        raw_data = cast(dict[str, Any], data)
        raw_config_value = raw_data.get("config")
        if not isinstance(raw_config_value, dict):
            return raw_data
        raw_config = cast(dict[str, Any], raw_config_value)
        unknown_fields = set(raw_config) - set(LLMConfig.model_fields)
        if unknown_fields:
            fields = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unknown LLM config field(s): {fields}")

        normalized: dict[str, Any] = dict(raw_data)
        normalized["config"] = raw_config
        return normalized

    def config_dict(self) -> dict[str, Any]:
        return dump_llm_config(self.config)


class QueryRequest(GatewayRequestBase):
    inputs: list[InputItem]
    tools: list[ToolDefinition] = Field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    run_id: str | None = None
    question_id: str | None = None
    query_id: str | None = None
    identity: object | None = None
    in_agent: bool = False
    token_retry_params: TokenRetryParams | None = None

    @field_validator("run_id", "question_id")
    @classmethod
    def _normalize_request_id(
        cls, value: str | None, info: ValidationInfo
    ) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            raise ValueError(f"{info.field_name} must not be blank")
        return text

    @field_validator("query_id")
    @classmethod
    def _normalize_query_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None

    @field_validator("identity", mode="before")
    @classmethod
    def _normalize_identity(cls, value: object) -> object:
        if value is None:
            return None
        try:
            return telemetry.normalize_identity(value)
        except telemetry.IdentityValidationError as exc:
            raise ValueError(str(exc)) from exc


class TokenCountRequest(GatewayRequestBase):
    inputs: list[InputItem]
    tools: list[ToolDefinition] = Field(default_factory=list)


class RateLimitRequest(GatewayRequestBase):
    pass


class RateLimitResponse(BaseModel):
    rate_limit: dict[str, Any] | None = None


class ModelResolveRequest(GatewayRequestBase):
    pass


class ModelResolveResponse(BaseModel):
    exists: bool
    model: str
    effective_config: dict[str, Any] | None = None
    registry_config: dict[str, Any] | None = None
    input_context_window: int | None = None


class ProviderError(BaseModel):
    type: Literal["ProviderError"] = "ProviderError"
    message: str
    provider: str | None = None
    code: str | None = None
    exception_type: str | None = None
    status_code: int | None = None


class GatewayResponse(BaseModel):
    error: ProviderError | None = None

    @model_validator(mode="after")
    def _require_success_xor_error(self) -> Self:
        payload_fields = [name for name in type(self).model_fields if name != "error"]
        has_error = self.error is not None
        has_payload = any(getattr(self, name) is not None for name in payload_fields)
        if has_error == has_payload:
            raise ValueError(
                "Gateway response must contain exactly one of error or payload"
            )
        return self


class TokenCountResponse(GatewayResponse):
    tokens: int | None = None


def query_result_response_body(
    result: QueryResult, *, signed_history: str
) -> dict[str, Any]:
    data = QueryResult.model_dump(
        result,
        mode="json",
        exclude={"history"},
        exclude_none=True,
    )
    data["signed_history"] = signed_history
    return data


class UploadFileRequest(GatewayRequestBase):
    name: str
    mime: str
    content_base64: str
    type: Literal["image", "file"] = "file"


class UploadFileResponse(GatewayResponse):
    file: FileWithId | None = None


class EmbeddingRequest(GatewayRequestBase):
    text: str
    embedding_model: str = "text-embedding-3-small"


class EmbeddingResponse(GatewayResponse):
    embedding: list[float] | None = None


class ModerationRequest(GatewayRequestBase):
    text: str


class ModerationResponse(GatewayResponse):
    response: dict[str, Any] | None = None

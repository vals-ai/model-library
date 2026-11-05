from typing import Annotated, Any, Self, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

import model_library.telemetry as telemetry
from model_gateway.egress import EgressPolicy, validate_egress_policy
from model_library.registry_utils import (
    get_model_names,
    get_registry_config,
    get_transcription_registry_config,
)

MAX_TTL_SECONDS = 7 * 24 * 60 * 60
# Tokens for `approved_identities_only` models never outlive this.
APPROVED_IDENTITY_MAX_TTL_SECONDS = 12 * 60 * 60
PASSTHROUGH_PREFIX = "/passthrough"

# Request compatibility for callers that do not yet submit attempt_id.
LEGACY_ATTEMPT_ID = ""


def accept_legacy_attempt(data: object) -> object:
    if not isinstance(data, dict):
        return data
    fields = cast(dict[str, Any], data)
    if "attempt_id" in fields:
        return fields
    return {**fields, "attempt_id": LEGACY_ATTEMPT_ID}


# The operations a token can be narrowed to, and the route each one covers.
RUN_TOKEN_OPERATION_PATHS = {
    "query": "/query",
    "tokens_count": "/tokens/count",
    "files_upload": "/files/upload",
    "embeddings": "/embeddings",
    "moderation": "/moderation",
    "audio_transcriptions": "/audio/transcriptions",
    "rate_limit": "/rate-limit",
    "passthrough": PASSTHROUGH_PREFIX,
}


def _is_model_pattern(key: str) -> bool:
    """`provider/x/*`: every id under `x/` at a known provider, pass-through only."""
    return key.endswith("/*") and bool(get_model_names(key.partition("/")[0]))


def known_models(models: list[str]) -> list[str]:
    # A typo would otherwise fail mid-task.
    unknown = [
        model
        for model in models
        if not _is_model_pattern(model)
        and get_registry_config(model) is None
        and get_transcription_registry_config(model) is None
    ]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    return models


def requires_approved_identity(model: str) -> bool:
    entry = get_registry_config(model)
    return entry is not None and entry.metadata.approved_identities_only


def known_operations(operations: list[str]) -> list[str]:
    unknown = set(operations) - RUN_TOKEN_OPERATION_PATHS.keys()
    if unknown:
        raise ValueError(f"Unknown operation(s): {', '.join(sorted(unknown))}")
    return operations


class ModelPermissions(BaseModel):
    # None allows every registry model.
    allowed_models: list[str] | None = None
    # `/embeddings` endpoint parameters, not registry keys.
    allowed_embedding_models: list[str] = []
    # None allows every operation.
    allowed_operations: list[str] | None = None


class MintRunTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed_models: list[str]
    # `/embeddings` endpoint parameters, not registry keys.
    allowed_embedding_models: list[str] = []
    # None allows every operation.
    allowed_operations: list[str] | None = None
    _accept_legacy_attempt = model_validator(mode="before")(accept_legacy_attempt)

    run_id: str
    task_id: str
    attempt_id: str

    @field_validator("allowed_models", mode="before")
    @classmethod
    def _validate_models(cls, models: object) -> object:
        # Minted tokens always name their models.
        if not isinstance(models, list):
            raise ValueError("allowed_models must be a list of model keys")
        return known_models(cast(list[str], models))

    @field_validator("allowed_operations")
    @classmethod
    def _validate_operations(cls, operations: list[str] | None) -> list[str] | None:
        return None if operations is None else known_operations(operations)

    identity: (
        Annotated[dict[str, Any], AfterValidator(telemetry.normalize_identity)] | None
    ) = None
    # The sandbox's network allowlist; provider-side fetches are held to it.
    egress: Annotated[EgressPolicy, AfterValidator(validate_egress_policy)] = "*"
    # Off by default: URL inputs (other than data:) and provider-side tools are stripped.
    allow_urls: bool = False
    allow_provider_tools: bool = False
    ttl_seconds: int = Field(gt=0, le=MAX_TTL_SECONDS)


class MintRunTokenResponse(BaseModel):
    token: str
    lease_id: str
    expires_at: float


class RevokeRunTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lease_id: str | None = None
    run_id: str | None = None

    @model_validator(mode="after")
    def _one_target(self) -> Self:
        if (self.lease_id is None) == (self.run_id is None):
            raise ValueError("Provide exactly one of lease_id or run_id")
        return self


class RevokeRunTokenResponse(BaseModel):
    revoked: int

from typing import Annotated, Any, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

import model_library.telemetry as telemetry
from model_library.registry_utils import (
    get_registry_config,
    get_transcription_registry_config,
)

MAX_TTL_SECONDS = 7 * 24 * 60 * 60

# The operations a token can be narrowed to, and the route each one covers.
RUN_TOKEN_OPERATION_PATHS = {
    "query": "/query",
    "tokens_count": "/tokens/count",
    "files_upload": "/files/upload",
    "embeddings": "/embeddings",
    "moderation": "/moderation",
    "audio_transcriptions": "/audio/transcriptions",
    "rate_limit": "/rate-limit",
}


def _known_models(models: list[str]) -> list[str]:
    # A typo would otherwise fail mid-task.
    unknown = [
        model
        for model in models
        if get_registry_config(model) is None
        and get_transcription_registry_config(model) is None
    ]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    return models


def _known_operations(operations: list[str]) -> list[str]:
    unknown = set(operations) - RUN_TOKEN_OPERATION_PATHS.keys()
    if unknown:
        raise ValueError(f"Unknown operation(s): {', '.join(sorted(unknown))}")
    return operations


class MintRunTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: str
    allowed_models: Annotated[list[str], AfterValidator(_known_models)]
    # `/embeddings` endpoint parameters, not registry keys.
    allowed_embedding_models: list[str] = []
    # None allows every operation.
    allowed_operations: (
        Annotated[list[str], AfterValidator(_known_operations)] | None
    ) = None
    identity: (
        Annotated[dict[str, Any], AfterValidator(telemetry.normalize_identity)] | None
    ) = None
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

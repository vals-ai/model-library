import hashlib
import secrets
from collections.abc import Callable
from typing import Any, TypeVar, cast
from uuid import UUID

from fastapi import Request
from pydantic import model_validator

from model_library.base import LLM
from model_library.providers.amazon.chat import AmazonModel
from model_library.providers.anthropic import AnthropicModel
from model_library.providers.google.google import GoogleModel
from model_library.register_models import RegistryEntry
from model_library.registry_utils import get_registry_config
from model_library.retriers.token import utils as token_utils

from model_gateway.egress import EgressFilter, EgressPolicy
from model_gateway.errors import RunTokenAuthorizationError
from model_gateway.observability import log_gateway_event
from model_gateway.run_token_types import (
    LEGACY_ATTEMPT_ID,
    MAX_TTL_SECONDS,
    ModelPermissions,
    accept_legacy_attempt,
    requires_approved_identity,
)
from model_gateway.types import GatewayRequestBase, QueryRequest, TokenCountRequest

_BodyT = TypeVar("_BodyT", bound=GatewayRequestBase)

RUN_TOKEN_PREFIX = "mgwt_"


class _BaseRunTokenClaims(ModelPermissions):
    run_id: str
    task_id: str
    attempt_id: str
    identity: dict[str, Any] | None = None
    egress: EgressPolicy = "*"
    allow_urls: bool = False
    allow_provider_tools: bool = False
    minted_by: str


class RunTokenClaims(_BaseRunTokenClaims):
    managed_key_id: UUID


class LegacyRunTokenClaims(_BaseRunTokenClaims):
    _accept_legacy_attempt = model_validator(mode="before")(accept_legacy_attempt)

    # Existing API keys bind tokens to their SHA-256 digest, not a managed key ID.
    minted_with: str


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _token_key(lease_id: str, *, legacy: bool) -> str:
    kind = "run_token" if legacy else "managed_run_token"
    return f"{token_utils.KEY_PREFIX}:gateway:{kind}:{lease_id}"


def _run_key(claims: RunTokenClaims | LegacyRunTokenClaims) -> str:
    if isinstance(claims, LegacyRunTokenClaims):
        return (
            f"{token_utils.KEY_PREFIX}:gateway:run_token_run:"
            f"{claims.minted_with}:{claims.run_id}"
        )
    return (
        f"{token_utils.KEY_PREFIX}:gateway:managed_run_token_run:"
        f"{claims.managed_key_id}:{claims.run_id}"
    )


_REVOKE_RUN_LUA = """
local revoked = 0
for _, lease_id in ipairs(redis.call('SMEMBERS', KEYS[1])) do
    revoked = revoked + redis.call('DEL', ARGV[1] .. lease_id)
end
redis.call('DEL', KEYS[1])
return revoked
"""


async def mint_run_token(
    claims: RunTokenClaims | LegacyRunTokenClaims, ttl_seconds: int
) -> tuple[str, str]:
    """Return the token and its lease id, the SHA-256 the claims are stored under."""
    token = f"{RUN_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    lease_id = sha256_hex(token)
    legacy = isinstance(claims, LegacyRunTokenClaims)
    pipe = token_utils.redis_client.pipeline(transaction=True)
    _ = pipe.set(
        _token_key(lease_id, legacy=legacy),
        claims.model_dump_json(),
        ex=ttl_seconds,
    )
    run_key = _run_key(claims)
    _ = pipe.sadd(run_key, lease_id)
    _ = pipe.expire(run_key, MAX_TTL_SECONDS)
    _ = await pipe.execute()
    return token, lease_id


async def get_run_token(lease_id: str) -> RunTokenClaims | LegacyRunTokenClaims | None:
    raw = await token_utils.redis_client.get(_token_key(lease_id, legacy=False))
    if raw is not None:
        return RunTokenClaims.model_validate_json(raw)
    raw = await token_utils.redis_client.get(_token_key(lease_id, legacy=True))
    return None if raw is None else LegacyRunTokenClaims.model_validate_json(raw)


async def revoke_lease(
    lease_id: str, claims: RunTokenClaims | LegacyRunTokenClaims
) -> int:
    """Revoke the lease; return 1, or 0 if a concurrent revoke got there first."""
    legacy = isinstance(claims, LegacyRunTokenClaims)
    pipe = token_utils.redis_client.pipeline(transaction=True)
    _ = pipe.delete(_token_key(lease_id, legacy=legacy))
    _ = pipe.srem(_run_key(claims), lease_id)
    deleted, _ = await pipe.execute()
    return deleted


async def revoke_run(run_id: str, key_id: UUID | str) -> int:
    """Revoke every live token the key minted for the run."""
    legacy = isinstance(key_id, str)
    kind = "run_token_run" if legacy else "managed_run_token_run"
    return await token_utils.redis_client.eval(
        _REVOKE_RUN_LUA,
        1,
        f"{token_utils.KEY_PREFIX}:gateway:{kind}:{key_id}:{run_id}",
        _token_key("", legacy=legacy),
    )


def run_token_authorized(model: type[_BodyT]) -> Callable[..., Any]:
    """Authorize the caller's models and stamp run-token identity when present."""

    async def dependency(request: Request, body: _BodyT) -> _BodyT:
        claims = cast(
            RunTokenClaims | LegacyRunTokenClaims | None,
            request.state.run_token_claims,
        )
        authority = request.state.managed_authority
        permissions = (
            claims
            if claims is not None
            else (authority.permissions if authority is not None else None)
        )
        if claims is None and requires_approved_identity(body.model):
            raise RunTokenAuthorizationError(
                f"Model {body.model!r} requires a run token for an approved identity"
            )
        if permissions is None:
            return body
        caller = "Run token" if claims is not None else "Managed key"
        if (
            permissions.allowed_models is not None
            and body.model not in permissions.allowed_models
        ):
            raise RunTokenAuthorizationError(
                f"{caller} does not authorize model {body.model!r}"
            )
        embedding_model = getattr(body, "embedding_model", None)
        if (
            embedding_model is not None
            and embedding_model not in permissions.allowed_embedding_models
        ):
            raise RunTokenAuthorizationError(
                f"{caller} does not authorize embedding model {embedding_model!r}"
            )
        if claims is None:
            return body
        # Retry params reconfigure provider budgets shared across runs.
        if getattr(body, "token_retry_params", None) is not None:
            raise RunTokenAuthorizationError("Run tokens cannot set token_retry_params")
        # A request provider_config replaces the entry's whole one, dropping a router
        # variant's YAML settings (pool, tier, preference).
        if body.config.provider_config is not None:
            entry = get_registry_config(body.model)
            if (
                entry is not None
                and entry.metadata.router
                and entry.provider_properties.model_dump(exclude_none=True)
            ):
                raise RunTokenAuthorizationError(
                    "Run tokens cannot set provider_config on router variants"
                )

        fields = type(body).model_fields
        updates: dict[str, Any] = {
            field: value
            for field, value in (
                ("run_id", claims.run_id),
                ("question_id", claims.task_id),
                ("attempt_id", claims.attempt_id),
                ("identity", claims.identity),
            )
            if field in fields
        }
        if claims.attempt_id == LEGACY_ATTEMPT_ID:
            updates.pop("attempt_id", None)
        return body.model_copy(update=updates)

    # FastAPI reads this annotation to parse `body` as the route's model.
    dependency.__annotations__["body"] = model
    return dependency


def registry_visibility(request: Request) -> Callable[[str, RegistryEntry], bool]:
    claims = cast(
        RunTokenClaims | LegacyRunTokenClaims | None, request.state.run_token_claims
    )
    if claims is None:
        return lambda _key, _config: True
    if claims.allowed_models is None:
        return lambda _key, _config: True
    keys = set(claims.allowed_models)
    full_keys = {
        entry.full_key
        for key in keys
        if (entry := get_registry_config(key)) is not None
    }
    return lambda key, config: key in keys or config.full_key in full_keys


def tool_results_are_content(llm: LLM) -> bool:
    """Whether the resolved adapter sends a list `ToolResult` result as native
    content parts (OpenAI Responses/Chat, Mistral, …). Google and Bedrock send it
    as JSON data and Anthropic as text."""
    while llm.delegate is not None:
        llm = llm.delegate
    return not isinstance(llm, (GoogleModel, AmazonModel, AnthropicModel))


def enforce_run_token_egress(
    request: Request, body: QueryRequest | TokenCountRequest, llm: LLM
) -> None:
    """Strip inputs and tools the run token does not let the provider fetch.

    Run after restoring signed history: a valid signature only proves the gateway
    produced the blob, not that this token may fetch what it contains.
    """
    claims = cast(RunTokenClaims | None, request.state.run_token_claims)
    if claims is None or (
        claims.egress == "*" and claims.allow_urls and claims.allow_provider_tools
    ):
        return
    egress = EgressFilter(
        claims.egress if claims.allow_urls else [],
        tool_results_are_content=tool_results_are_content(llm),
    )
    tools = body.tools
    if claims.egress != "*" or not claims.allow_provider_tools:
        tools = egress.filter_tools(tools)
    inputs = egress.filter_inputs(body.inputs)
    if not egress.changed:
        return
    log_gateway_event(
        "gateway.run_token.egress_blocked",
        run_id=claims.run_id,
        task_id=claims.task_id,
        model=body.model,
        removed_inputs=egress.removed_inputs,
        removed_tools=",".join(egress.removed_tools) or None,
        blocked_urls=",".join(egress.logged_urls) or None,
    )
    body.inputs = inputs
    body.tools = tools

import hashlib
import secrets
from collections.abc import Callable
from typing import Any, TypeVar, cast

from fastapi import Request
from pydantic import BaseModel

from model_library.register_models import RegistryEntry
from model_library.registry_utils import get_registry_config
from model_library.retriers.token import utils as token_utils

from model_gateway.errors import RunTokenAuthorizationError
from model_gateway.run_token_types import MAX_TTL_SECONDS
from model_gateway.types import GatewayRequestBase

_BodyT = TypeVar("_BodyT", bound=GatewayRequestBase)

RUN_TOKEN_PREFIX = "mgwt_"

# Fields that send the call to another model or endpoint with the gateway's key.
DENIED_PROVIDER_CONFIG_FIELDS = frozenset({"api_base", "fallback_models"})


class RunTokenClaims(BaseModel):
    run_id: str
    task_id: str
    allowed_models: list[str]
    allowed_embedding_models: list[str] = []
    allowed_operations: list[str] | None = None
    identity: dict[str, Any] | None = None
    minted_by: str
    # Only a gateway accepting the minting key may use this token.
    minted_with: str


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _token_key(lease_id: str) -> str:
    return f"{token_utils.KEY_PREFIX}:gateway:run_token:{lease_id}"


def _run_key(minted_with: str, run_id: str) -> str:
    return f"{token_utils.KEY_PREFIX}:gateway:run_token_run:{minted_with}:{run_id}"


_REVOKE_RUN_LUA = """
local revoked = 0
for _, lease_id in ipairs(redis.call('SMEMBERS', KEYS[1])) do
    revoked = revoked + redis.call('DEL', ARGV[1] .. lease_id)
end
redis.call('DEL', KEYS[1])
return revoked
"""


async def mint_run_token(claims: RunTokenClaims, ttl_seconds: int) -> tuple[str, str]:
    """Return the token and its lease id, the SHA-256 the claims are stored under."""
    token = f"{RUN_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    lease_id = sha256_hex(token)
    pipe = token_utils.redis_client.pipeline(transaction=True)
    _ = pipe.set(_token_key(lease_id), claims.model_dump_json(), ex=ttl_seconds)
    run_key = _run_key(claims.minted_with, claims.run_id)
    _ = pipe.sadd(run_key, lease_id)
    _ = pipe.expire(run_key, MAX_TTL_SECONDS)
    _ = await pipe.execute()
    return token, lease_id


async def get_run_token(lease_id: str) -> RunTokenClaims | None:
    raw = await token_utils.redis_client.get(_token_key(lease_id))
    return None if raw is None else RunTokenClaims.model_validate_json(raw)


async def revoke_lease(lease_id: str, claims: RunTokenClaims) -> int:
    """Revoke the lease; return 1, or 0 if a concurrent revoke got there first."""
    pipe = token_utils.redis_client.pipeline(transaction=True)
    _ = pipe.delete(_token_key(lease_id))
    _ = pipe.srem(_run_key(claims.minted_with, claims.run_id), lease_id)
    deleted, _ = await pipe.execute()
    return deleted


async def revoke_run(run_id: str, minted_with: str) -> int:
    """Revoke every live token the key minted for the run; return how many."""
    # One script, so no mint can land between reading the index and deleting it.
    return await token_utils.redis_client.eval(
        _REVOKE_RUN_LUA, 1, _run_key(minted_with, run_id), _token_key("")
    )


def run_token_authorized(model: type[_BodyT]) -> Callable[..., Any]:
    """Authorize a body against the caller's run token and stamp its identity."""

    async def dependency(request: Request, body: _BodyT) -> _BodyT:
        claims = cast(RunTokenClaims | None, request.state.run_token_claims)
        if claims is None:
            return body
        if body.model not in claims.allowed_models:
            raise RunTokenAuthorizationError(
                f"Run token does not authorize model {body.model!r}"
            )
        embedding_model = getattr(body, "embedding_model", None)
        if (
            embedding_model is not None
            and embedding_model not in claims.allowed_embedding_models
        ):
            raise RunTokenAuthorizationError(
                f"Run token does not authorize embedding model {embedding_model!r}"
            )
        # Retry params reconfigure provider budgets shared across runs.
        if getattr(body, "token_retry_params", None) is not None:
            raise RunTokenAuthorizationError("Run tokens cannot set token_retry_params")
        provider_config = body.config.provider_config
        if provider_config is not None:
            set_fields = (
                provider_config.keys()
                if isinstance(provider_config, dict)
                else provider_config.model_fields_set
            )
            denied = sorted(set_fields & DENIED_PROVIDER_CONFIG_FIELDS)
            if denied:
                raise RunTokenAuthorizationError(
                    f"Run tokens cannot set provider_config field(s): {', '.join(denied)}"
                )
            # A request provider_config replaces the entry's whole one, which would drop
            # a YAML openrouter_allowed_models pool and let the sandbox reach other models.
            entry = get_registry_config(body.model)
            if (
                entry is not None
                and getattr(
                    entry.provider_properties, "openrouter_allowed_models", None
                )
                is not None
            ):
                raise RunTokenAuthorizationError(
                    "Run tokens cannot set provider_config on models with a fixed "
                    "allowed_models pool"
                )

        fields = type(body).model_fields
        updates: dict[str, Any] = {
            field: value
            for field, value in (
                ("run_id", claims.run_id),
                ("question_id", claims.task_id),
                ("identity", claims.identity),
            )
            if field in fields
        }
        return body.model_copy(update=updates)

    # FastAPI reads this annotation to parse `body` as the route's model.
    dependency.__annotations__["body"] = model
    return dependency


def registry_visibility(request: Request) -> Callable[[str, RegistryEntry], bool]:
    claims = cast(RunTokenClaims | None, request.state.run_token_claims)
    if claims is None:
        return lambda _key, _config: True
    keys = set(claims.allowed_models)
    full_keys = {
        entry.full_key
        for key in keys
        if (entry := get_registry_config(key)) is not None
    }
    return lambda key, config: key in keys or config.full_key in full_keys

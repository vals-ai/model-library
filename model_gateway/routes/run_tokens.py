"""Run-token control routes: mint and revoke."""

import time
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from model_gateway.observability import log_gateway_event
from model_gateway.runtime_config import runtime_access
from model_gateway.run_token_types import (
    APPROVED_IDENTITY_MAX_TTL_SECONDS,
    MintRunTokenRequest,
    MintRunTokenResponse,
    RevokeRunTokenRequest,
    RevokeRunTokenResponse,
    requires_approved_identity,
)
from model_gateway.run_tokens import (
    LegacyRunTokenClaims,
    RunTokenClaims,
    get_run_token,
    mint_run_token,
    revoke_lease,
    revoke_run,
)


def register_run_token_routes(app: FastAPI) -> None:
    @app.post("/service-auth", response_model=MintRunTokenResponse)
    async def mint(
        request: Request, body: MintRunTokenRequest
    ) -> MintRunTokenResponse | JSONResponse:
        authority = request.state.managed_authority
        minted_by = cast(str, request.state.gateway_api_key_name)
        ttl_seconds = body.ttl_seconds
        if any(requires_approved_identity(model) for model in body.allowed_models):
            email = (body.identity or {}).get("email")
            if not runtime_access().admits(minted_by, email):
                log_gateway_event(
                    "gateway.run_token.approved_identity_refused",
                    run_id=body.run_id,
                    task_id=body.task_id,
                    minted_by=minted_by,
                )
                return JSONResponse(
                    status_code=403,
                    content={
                        "code": "forbidden",
                        "message": "Token names a model limited to approved identities",
                    },
                )
            ttl_seconds = min(ttl_seconds, APPROVED_IDENTITY_MAX_TTL_SECONDS)
        fields = body.model_dump(exclude={"ttl_seconds"})
        claims = (
            RunTokenClaims(
                **fields,
                minted_by=minted_by,
                managed_key_id=authority.key_id,
            )
            if authority is not None
            else LegacyRunTokenClaims(
                **fields,
                minted_by=minted_by,
                minted_with=cast(str, request.state.gateway_api_key_digest),
            )
        )
        token, lease_id = await mint_run_token(claims, ttl_seconds)
        log_gateway_event(
            "gateway.run_token.minted",
            lease_id=lease_id,
            run_id=body.run_id,
            task_id=body.task_id,
            models=",".join(body.allowed_models or []),
            ttl_seconds=ttl_seconds,
            minted_by=minted_by,
        )
        return MintRunTokenResponse(
            token=token, lease_id=lease_id, expires_at=time.time() + ttl_seconds
        )

    @app.post("/service-auth/revoke", response_model=RevokeRunTokenResponse)
    async def revoke(
        request: Request, body: RevokeRunTokenRequest
    ) -> RevokeRunTokenResponse | JSONResponse:
        authority = request.state.managed_authority
        key_id = (
            authority.key_id
            if authority is not None
            else cast(str, request.state.gateway_api_key_digest)
        )
        if body.run_id is not None:
            revoked = await revoke_run(body.run_id, key_id)
        else:
            lease_id = cast(str, body.lease_id)
            claims = await get_run_token(lease_id)
            if claims is not None and (
                (
                    isinstance(claims, LegacyRunTokenClaims)
                    and claims.minted_with != key_id
                )
                or (
                    isinstance(claims, RunTokenClaims)
                    and claims.managed_key_id != key_id
                )
            ):
                return JSONResponse(
                    status_code=403,
                    content={
                        "code": "forbidden",
                        "message": "Minted by another API key",
                    },
                )
            revoked = 0 if claims is None else await revoke_lease(lease_id, claims)
            if not revoked:
                return JSONResponse(
                    status_code=404,
                    content={
                        "code": "not_found",
                        "message": "No run token for that lease",
                    },
                )
        log_gateway_event(
            "gateway.run_token.revoked",
            lease_id=body.lease_id,
            run_id=body.run_id,
            revoked=revoked,
            minted_by=cast(str, request.state.gateway_api_key_name),
        )
        return RevokeRunTokenResponse(revoked=revoked)

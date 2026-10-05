"""Run-token control routes: mint and revoke."""

import time
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from model_gateway.observability import log_gateway_event
from model_gateway.run_token_types import (
    MintRunTokenRequest,
    MintRunTokenResponse,
    RevokeRunTokenRequest,
    RevokeRunTokenResponse,
)
from model_gateway.run_tokens import (
    RunTokenClaims,
    get_run_token,
    mint_run_token,
    revoke_lease,
    revoke_run,
)


def register_run_token_routes(app: FastAPI) -> None:
    @app.post("/service-auth", response_model=MintRunTokenResponse)
    async def mint(request: Request, body: MintRunTokenRequest) -> MintRunTokenResponse:
        minted_by = cast(str, request.state.gateway_api_key_name)
        claims = RunTokenClaims(
            **body.model_dump(exclude={"ttl_seconds"}),
            minted_by=minted_by,
            minted_with=cast(str, request.state.gateway_api_key_digest),
        )
        token, lease_id = await mint_run_token(claims, body.ttl_seconds)
        log_gateway_event(
            "gateway.run_token.minted",
            lease_id=lease_id,
            run_id=body.run_id,
            task_id=body.task_id,
            models=",".join(body.allowed_models),
            ttl_seconds=body.ttl_seconds,
            minted_by=minted_by,
        )
        return MintRunTokenResponse(
            token=token, lease_id=lease_id, expires_at=time.time() + body.ttl_seconds
        )

    @app.post("/service-auth/revoke", response_model=RevokeRunTokenResponse)
    async def revoke(
        request: Request, body: RevokeRunTokenRequest
    ) -> RevokeRunTokenResponse | JSONResponse:
        minted_with = cast(str, request.state.gateway_api_key_digest)
        if body.run_id is not None:
            # Teardown is idempotent: tokens also expire on their own.
            revoked = await revoke_run(body.run_id, minted_with)
        else:
            lease_id = cast(str, body.lease_id)
            claims = await get_run_token(lease_id)
            if claims is not None and claims.minted_with != minted_with:
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

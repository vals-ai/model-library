import logging
from dataclasses import dataclass
from typing import Protocol, cast
from uuid import UUID

from descope import AuthException, DescopeClientAsync
from descope.exceptions import ERROR_TYPE_INVALID_PUBLIC_KEY, ERROR_TYPE_INVALID_TOKEN
from fastapi import HTTPException
from jwt.exceptions import PyJWTError
from pydantic import BaseModel

from model_gateway.managed_auth.types import HumanIdentity, MachineIdentity

logger = logging.getLogger(__name__)


class SessionValidator(Protocol):
    async def validate_session(self, session_token: str) -> object: ...


class AccessKeyExchanger(Protocol):
    async def exchange_access_key(self, access_key: str) -> dict[str, object]: ...


class UserLoader(Protocol):
    async def load_by_user_id(self, user_id: str) -> object: ...


class SessionResponse(BaseModel):
    sessionToken: dict[str, object]


class UserTenant(BaseModel):
    tenantId: str


class DescopeUser(BaseModel):
    userId: str
    status: str
    userTenants: list[UserTenant]
    permissions: list[str]


class UserResponse(BaseModel):
    user: DescopeUser


@dataclass(frozen=True)
class VerifiedHuman:
    identity: HumanIdentity
    permissions: frozenset[str]


@dataclass(frozen=True)
class VerifiedMachine:
    identity: MachineIdentity
    organization_id: str
    service_id: UUID


class MachineVerifier:
    def __init__(
        self,
        project_id: str,
        expected_issuer: str,
        required_organization: str,
        registered_access_key_id: str,
        service_id: UUID,
    ) -> None:
        self._client = DescopeClientAsync(project_id=project_id)
        self._expected_issuer = expected_issuer
        self._required_organization = required_organization
        self._registered_access_key_id = registered_access_key_id
        self._service_id = service_id

    async def aclose(self) -> None:
        await self._client.aclose()

    async def verify(self, access_key: str) -> VerifiedMachine:
        try:
            exchange = await cast(AccessKeyExchanger, self._client).exchange_access_key(
                access_key
            )
        except AuthException as exc:
            if exc.status_code not in (400, 401, 403):
                raise
            raise HTTPException(status_code=401, detail="Invalid machine key") from exc
        claims = exchange.get("sessionToken")
        if (
            not isinstance(claims, dict)
            or cast(dict[str, object], claims).get("iss") != self._expected_issuer
        ):
            raise HTTPException(status_code=401, detail="Invalid machine issuer")
        if exchange.get("keyId") != self._registered_access_key_id:
            raise HTTPException(status_code=401, detail="Unregistered machine key")
        tenants = exchange.get("tenants")
        if not isinstance(tenants, dict) or tuple(cast(dict[str, object], tenants)) != (
            self._required_organization,
        ):
            raise HTTPException(status_code=403, detail="Invalid machine organization")
        return VerifiedMachine(
            identity=MachineIdentity(
                issuer=self._expected_issuer,
                subject=self._registered_access_key_id,
            ),
            organization_id=self._required_organization,
            service_id=self._service_id,
        )


class HumanVerifier:
    def __init__(
        self,
        project_id: str,
        expected_issuer: str,
        required_organization: str,
        management_credential: str,
        registered_machine_key_id: str,
    ) -> None:
        self._client = DescopeClientAsync(
            project_id=project_id, management_key=management_credential
        )
        self._expected_issuer = expected_issuer
        self._required_organization = required_organization
        self._registered_machine_key_id = registered_machine_key_id

    async def aclose(self) -> None:
        await self._client.aclose()

    async def verify(self, token: str) -> VerifiedHuman:
        validator = cast(SessionValidator, self._client)
        try:
            session = SessionResponse.model_validate(
                await validator.validate_session(token)
            )
        except AuthException as exc:
            if exc.error_type != ERROR_TYPE_INVALID_TOKEN and not (
                exc.error_type == ERROR_TYPE_INVALID_PUBLIC_KEY
                and exc.error_message
                == "Unable to validate public key. Public key not found."
            ):
                raise
            logger.warning(
                "Managed-auth human JWT rejected: %s (%s)",
                exc.error_type,
                type(exc.__context__).__name__,
            )
            raise HTTPException(
                status_code=401, detail="Invalid human session"
            ) from exc
        except PyJWTError as exc:
            logger.warning("Managed-auth human JWT rejected: %s", type(exc).__name__)
            raise HTTPException(
                status_code=401, detail="Invalid human session"
            ) from exc

        claims = session.sessionToken
        if claims.get("iss") != self._expected_issuer:
            raise HTTPException(status_code=401, detail="Invalid human session issuer")
        if (claims.get("dsub") or claims.get("sub")) == self._registered_machine_key_id:
            raise HTTPException(status_code=401, detail="Invalid human session")
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise HTTPException(status_code=401, detail="Invalid human session subject")

        try:
            user = await self._load_user(subject)
        except AuthException as exc:
            if exc.status_code != 404:
                raise
            raise HTTPException(
                status_code=401, detail="Unknown human session subject"
            ) from exc

        if user.userId != subject:
            raise HTTPException(status_code=401, detail="Invalid human session subject")
        if user.status != "enabled":
            raise HTTPException(status_code=403, detail="Human user is not enabled")
        if not any(
            tenant.tenantId == self._required_organization
            for tenant in user.userTenants
        ):
            raise HTTPException(
                status_code=403, detail="Human user is not in the required organization"
            )

        return VerifiedHuman(
            identity=HumanIdentity(issuer=self._expected_issuer, subject=subject),
            permissions=frozenset(user.permissions),
        )

    async def current_eligibility(self, identity: HumanIdentity) -> bool:
        if identity.issuer != self._expected_issuer:
            raise ValueError("Human identity issuer does not match Descope issuer")
        try:
            user = await self._load_user(identity.subject)
        except AuthException as exc:
            if exc.status_code == 404:
                return False
            raise
        if user.userId != identity.subject:
            raise ValueError("Descope returned a different human identity")
        return user.status == "enabled" and any(
            tenant.tenantId == self._required_organization
            for tenant in user.userTenants
        )

    async def _load_user(self, subject: str) -> DescopeUser:
        loader = cast(UserLoader, self._client.mgmt.user)
        response = UserResponse.model_validate(
            await loader.load_by_user_id(user_id=subject)
        )
        return response.user

import hashlib
import secrets


def managed_key_verifier(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def create_managed_key_secret() -> tuple[str, str]:
    secret = "mgwk_" + secrets.token_urlsafe(32)
    return secret, managed_key_verifier(secret)

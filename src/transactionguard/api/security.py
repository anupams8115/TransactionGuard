"""Password hashing and JWT access tokens.

* Passwords: Argon2id via ``pwdlib`` (the hasher FastAPI's docs now recommend). Argon2 is
  deliberately slow and memory-hard, so stolen hashes are expensive to brute-force.
* Tokens: short-lived HS256 JWTs signed with ``TG_JWT_SECRET``. Claims carry the user id,
  workspace id and role, but every request still re-loads the user from the database so a
  deactivated user or changed role takes effect immediately.
* Browser clients get the token in an ``HttpOnly`` + ``SameSite=Strict`` cookie (JavaScript cannot
  read it, which limits XSS token theft); API clients may send ``Authorization: Bearer``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from pwdlib import PasswordHash

from transactionguard.core.config import get_settings

COOKIE_NAME = "tg_session"
_hasher = PasswordHash.recommended()
# Hash of a random password, used to equalise timing when the email does not exist.
_DUMMY_HASH = _hasher.hash(uuid.uuid4().hex)


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    if password_hash is None:
        _hasher.verify(password, _DUMMY_HASH)  # constant-ish time: no user enumeration by timing
        return False
    return _hasher.verify(password, password_hash)


def create_access_token(*, user_id: uuid.UUID, workspace_id: uuid.UUID, role: str) -> tuple[str, int]:
    s = get_settings()
    now = datetime.now(UTC)
    ttl = timedelta(minutes=s.access_token_minutes)
    claims = {
        "sub": str(user_id),
        "ws": str(workspace_id),
        "role": role,
        "iat": now,
        "exp": now + ttl,
        "jti": uuid.uuid4().hex,
        "iss": "transactionguard",
    }
    token = jwt.encode(claims, s.jwt_secret.get_secret_value(), algorithm=s.jwt_algorithm)
    return token, int(ttl.total_seconds())


def decode_access_token(token: str) -> dict[str, Any]:
    s = get_settings()
    return jwt.decode(
        token,
        s.jwt_secret.get_secret_value(),
        algorithms=[s.jwt_algorithm],  # pinned: never accept "none" or a client-chosen algorithm
        issuer="transactionguard",
        options={"require": ["exp", "sub", "ws", "role"]},
    )

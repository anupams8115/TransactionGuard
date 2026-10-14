"""FastAPI dependencies: DB session, current user, role checks, CSRF guard."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from transactionguard.api.security import COOKIE_NAME, decode_access_token
from transactionguard.db.models import Role, User
from transactionguard.db.session import get_sessionmaker

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
CSRF_HEADER = "x-requested-with"


def get_db() -> Iterator[Session]:
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


DB = Annotated[Session, Depends(get_db)]

_UNAUTHORIZED = HTTPException(
    status.HTTP_401_UNAUTHORIZED, "Not authenticated", headers={"WWW-Authenticate": "Bearer"}
)


def get_current_user(request: Request, db: DB) -> User:
    token: str | None = None
    from_cookie = False
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    elif COOKIE_NAME in request.cookies:
        token, from_cookie = request.cookies[COOKIE_NAME], True
    if not token:
        raise _UNAUTHORIZED
    try:
        claims = decode_access_token(token)
    except jwt.PyJWTError as exc:
        raise _UNAUTHORIZED from exc
    # Cookie auth is sent automatically by browsers, so state-changing requests must also carry a
    # custom header. Cross-site forms cannot set custom headers (and CORS blocks scripted ones).
    if from_cookie and request.method not in SAFE_METHODS and not request.headers.get(CSRF_HEADER):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Missing X-Requested-With header")
    user = db.get(User, _uuid(claims["sub"]))
    if user is None or not user.is_active or str(user.workspace_id) != claims["ws"]:
        raise _UNAUTHORIZED
    return user


def _uuid(value: str):  # type: ignore[no-untyped-def]
    import uuid

    try:
        return uuid.UUID(value)
    except ValueError as exc:
        raise _UNAUTHORIZED from exc


CurrentUser = Annotated[User, Depends(get_current_user)]

_RANK = {Role.viewer: 0, Role.analyst: 1, Role.admin: 2}


def require_role(minimum: Role) -> Callable[[User], User]:
    def checker(user: CurrentUser) -> User:
        if _RANK[user.role] < _RANK[minimum]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires role '{minimum.value}' or higher")
        return user

    return checker


Analyst = Annotated[User, Depends(require_role(Role.analyst))]
Admin = Annotated[User, Depends(require_role(Role.admin))]

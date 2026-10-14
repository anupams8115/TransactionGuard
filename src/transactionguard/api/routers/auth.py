"""Authentication and user management."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from transactionguard.api.deps import DB, Admin, CurrentUser
from transactionguard.api.schemas import TokenOut, UserCreate, UserOut
from transactionguard.api.security import (
    COOKIE_NAME,
    create_access_token,
    hash_password,
    verify_password,
)
from transactionguard.core.config import get_settings
from transactionguard.core.logging import get_logger
from transactionguard.db.models import User

router = APIRouter(tags=["auth"])
log = get_logger(__name__)


def _user_out(u: User) -> UserOut:
    out = UserOut.model_validate(u)
    out.workspace_name = u.workspace.name
    return out


@router.post("/auth/login", response_model=TokenOut)
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], response: Response, db: DB) -> TokenOut:
    """OAuth2 password flow. Returns a bearer token AND sets an HttpOnly session cookie."""
    user = db.scalars(select(User).where(func.lower(User.email) == form.username.lower())).first()
    if not verify_password(form.password, user.password_hash if user else None) or not (
        user and user.is_active
    ):
        log.info("login_failed")  # never log the email/password attempted
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    token, ttl = create_access_token(user_id=user.id, workspace_id=user.workspace_id, role=user.role.value)
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=ttl,
        httponly=True,
        secure=get_settings().cookie_secure,
        samesite="strict",
        path="/",
    )
    log.info("login_succeeded", user_id=str(user.id))
    return TokenOut(access_token=token, expires_in=ttl)


@router.post("/auth/logout", status_code=204)
def logout(response: Response) -> Response:
    response.delete_cookie(COOKIE_NAME, path="/")
    response.status_code = 204
    return response


@router.get("/auth/me", response_model=UserOut)
def me(user: CurrentUser) -> UserOut:
    return _user_out(user)


@router.get("/users", response_model=list[UserOut])
def list_users(admin: Admin, db: DB) -> list[UserOut]:
    users = db.scalars(select(User).where(User.workspace_id == admin.workspace_id).order_by(User.email))
    return [_user_out(u) for u in users]


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(body: UserCreate, admin: Admin, db: DB) -> UserOut:
    """Admins create users inside their OWN workspace only."""
    user = User(
        workspace_id=admin.workspace_id,
        email=body.email.lower(),
        password_hash=hash_password(body.password),
        role=body.role,
    )
    db.add(user)
    try:
        db.flush()
    except IntegrityError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "A user with this email already exists") from exc
    db.refresh(user)
    return _user_out(user)

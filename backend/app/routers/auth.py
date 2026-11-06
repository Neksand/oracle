"""ORACLE - Auth router: login (OAuth2 password), register, me, admin check.

Security properties:
- bcrypt password hashing, JWT (HS256) access tokens
- 3 failed logins -> 5 minute lockout per (username, client IP)
- password policy enforced server-side
- audit trail for every success/failure
"""
from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.orm_models import AuditLog, User
from app.security import (
    create_access_token,
    get_current_user,
    login_lock_seconds,
    hash_password,
    register_login_failure,
    require_admin,
    reset_login_failures,
    verify_password,
)

router = APIRouter(tags=["auth"])

USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]{3,32}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

    @field_validator("username")
    @classmethod
    def _username(cls, v: str) -> str:
        if not USERNAME_RE.match(v):
            raise ValueError("username must be 3-32 chars: letters, digits, . _ -")
        return v

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        if not EMAIL_RE.match(v):
            raise ValueError("invalid email")
        return v.lower()

    @field_validator("password")
    @classmethod
    def _password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        if len(v) > 128:
            raise ValueError("password must be at most 128 characters")
        if not re.search(r"[A-Za-z]", v) or not re.search(r"\d", v):
            raise ValueError("password must contain a letter and a digit")
        return v


def _throttle_key(username: str, request: Request) -> str:
    ip = (request.client.host if request.client else "local") or "local"
    return f"{username.strip().lower()}|{ip}"


def _audit(db: Session, event: str, actor: str, detail: str, blocked: bool) -> None:
    db.add(AuditLog(event_type=event, actor=actor[:64] or "anonymous",
                    detail=detail, blocked=blocked))
    db.commit()


@router.post("/auth/register", response_model=TokenResponse, status_code=201,
             summary="Create an account and receive a JWT")
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=409, detail="Username already taken")
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        username=req.username,
        email=req.email,
        password_hash=hash_password(req.password),
        role="user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    _audit(db, "register", user.username, "Account created", blocked=False)
    token = create_access_token(user.username, user.role)
    return TokenResponse(access_token=token, role=user.role, username=user.username)


@router.post("/auth/login", response_model=TokenResponse,
             summary="OAuth2 password flow (form: username/password)")
async def login(request: Request, form: OAuth2PasswordRequestForm = Depends(),
                db: Session = Depends(get_db)):
    key = _throttle_key(form.username, request)

    locked = login_lock_seconds(key)
    if locked > 0:
        _audit(db, "login_blocked", form.username, f"Lockout active, {locked}s left", blocked=True)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed attempts. Try again in {locked} seconds.",
            headers={"Retry-After": str(locked)},
        )

    user = db.query(User).filter(User.username == form.username).first()
    if user is None or not verify_password(form.password, user.password_hash):
        remaining, lock_for = register_login_failure(key)
        _audit(db, "login_failed", form.username,
               f"Wrong password or unknown user (remaining={remaining})", blocked=True)
        if lock_for:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many failed attempts. Account locked for {lock_for} seconds.",
                headers={"Retry-After": str(lock_for)},
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Incorrect username or password. {remaining} attempt(s) left.",
        )

    if not user.is_active:
        _audit(db, "login_blocked", user.username, "Account disabled", blocked=True)
        raise HTTPException(status_code=403, detail="Account disabled")

    reset_login_failures(key)
    token = create_access_token(user.username, user.role)
    _audit(db, "login_success", user.username, "OAuth2 password flow, JWT issued", blocked=False)
    return TokenResponse(access_token=token, role=user.role, username=user.username)


@router.get("/auth/me", summary="Current user from JWT")
async def me(user: User = Depends(get_current_user)):
    return {"username": user.username, "email": user.email, "role": user.role}


@router.get("/auth/admin-only", summary="Demo of role-based access (403 for non-admins)")
async def admin_only(user: User = Depends(require_admin)):
    return {"admin_area": True, "welcome": user.username}

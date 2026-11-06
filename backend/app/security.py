"""ORACLE - Authentication & authorization (OAuth2 password flow).

- bcrypt password hashing (never store plaintext)
- JWT access tokens (HS256) with exp/iat claims
- role-based access: user / admin
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
import os
import secrets
import threading
import time
from collections.abc import Iterator

import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.orm_models import ApiKey, User

SECRET_KEY = os.environ.get("ORACLE_JWT_SECRET") or secrets.token_hex(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# ---------------------------------------------------------------------------
# Login throttling: 3 failed attempts -> 5 minute lock (per username+IP).
# In-memory is enough for the single-process demo; a shared store (Redis) is
# the upgrade path if the API is ever run with several workers.
# ---------------------------------------------------------------------------
LOGIN_MAX_ATTEMPTS = int(os.environ.get("ORACLE_LOGIN_MAX_ATTEMPTS", "3"))
LOGIN_LOCKOUT_SECONDS = int(os.environ.get("ORACLE_LOGIN_LOCKOUT_SECONDS", "300"))

_attempts: dict[str, list[float]] = {}
_attempts_lock = threading.Lock()


def login_lock_seconds(key: str) -> int:
    """Seconds left on a lockout for this key, or 0 when not locked."""
    now = time.time()
    with _attempts_lock:
        entry = _attempts.get(key)
        if not entry:
            return 0
        failures, locked_until = entry
        if locked_until and locked_until > now:
            return int(math.ceil(locked_until - now))
        if locked_until and locked_until <= now:
            _attempts.pop(key, None)
        return 0


def register_login_failure(key: str) -> tuple[int, int]:
    """Record a failure. Returns (remaining_attempts, lock_seconds)."""
    now = time.time()
    with _attempts_lock:
        failures, locked_until = _attempts.get(key, ([], 0.0))
        if locked_until and locked_until <= now:
            failures, locked_until = [], 0.0
        failures.append(now)
        # only failures inside the window count
        failures = [t for t in failures if now - t <= LOGIN_LOCKOUT_SECONDS]
        if len(failures) >= LOGIN_MAX_ATTEMPTS:
            locked_until = now + LOGIN_LOCKOUT_SECONDS
            _attempts[key] = ([], locked_until)
            return 0, LOGIN_LOCKOUT_SECONDS
        _attempts[key] = (failures, locked_until)
        return max(0, LOGIN_MAX_ATTEMPTS - len(failures)), 0


def reset_login_failures(key: str) -> None:
    with _attempts_lock:
        _attempts.pop(key, None)


# ---------------------------------------------------------------------------
# API keys: stored as SHA-256 hashes, shown in plaintext only once.
# ---------------------------------------------------------------------------
API_KEY_PREFIX = "oracle_"


def generate_api_key() -> str:
    return API_KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_user_from_api_key(db: Session, raw_key: str | None) -> User | None:
    if not raw_key:
        return None
    row = (
        db.query(ApiKey)
        .filter(ApiKey.key_hash == hash_api_key(raw_key), ApiKey.revoked.is_(False))
        .first()
    )
    if row is None:
        return None
    user = db.query(User).filter(User.id == row.user_id).first()
    if user is None or not user.is_active:
        return None
    row.last_used_at = dt.datetime.now(dt.timezone.utc)
    db.commit()
    return user


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> User:
    """Dependency for the public API: valid X-API-Key header required."""
    user = get_user_from_api_key(db, x_api_key)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return user


def hash_password(raw: str) -> str:
    import bcrypt

    return bcrypt.hashpw(raw.encode(), bcrypt.gensalt(rounds=10)).decode()


def verify_password(raw: str, hashed: str) -> bool:
    import bcrypt

    try:
        return bcrypt.checkpw(raw.encode(), hashed.encode())
    except ValueError:
        return False


def create_access_token(username: str, role: str = "user") -> str:
    now = dt.datetime.now(dt.timezone.utc)
    payload = {
        "sub": username,
        "role": role,
        "iat": now,
        "exp": now + dt.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "iss": "oracle-core",
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], issuer="oracle-core")
    except (jwt.PyJWTError, jwt.exceptions.PyJWTError):
        return None


def get_current_user(db: Session = Depends(get_db), token: str | None = Depends(oauth2_scheme)) -> User:
    """Dependency: raises 401 when the token is missing/invalid."""
    creds_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid Bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise creds_exc
    payload = decode_token(token)
    if not payload or "sub" not in payload:
        raise creds_exc
    user = db.query(User).filter(User.username == payload["sub"]).first()
    if user is None or not user.is_active:
        raise creds_exc
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return user


def optional_user(db: Session = Depends(get_db), token: str | None = Depends(oauth2_scheme)) -> Iterator[User | None]:
    """Yields the user, or None in anonymous mode (read-only public data)."""
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None
    return db.query(User).filter(User.username == payload["sub"]).first()

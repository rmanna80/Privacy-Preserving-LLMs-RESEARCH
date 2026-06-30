"""
api/security.py — token issuance / verification and the auth dependency.

Design notes (appliance model):
  - Angel runs on the firm's own hardware. The API binds to localhost.
  - Multiple users (advisors, clients, admin) share one install, so the
    real threat is one user reaching another user's data — not an
    internet attacker. Authorization between users is therefore enforced
    on every protected endpoint, server-side, from the token.
  - We issue signed JWTs (HS256). The token encodes the authenticated
    username + role; the client never gets to *assert* its own identity
    in a request body or query param.

Secret key:
  Read from ANGEL_API_SECRET. In production the app refuses to start
  without it (no silent weak fallback). A dev fallback is only used when
  ANGEL_ENV != "production".
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError

# ── Config ───────────────────────────────────────────────────────────
ALGORITHM = "HS256"
TOKEN_TTL_MINUTES = int(os.environ.get("ANGEL_TOKEN_TTL_MINUTES", "720"))  # 12h
ANGEL_ENV = os.environ.get("ANGEL_ENV", "development")

_DEV_FALLBACK_SECRET = "dev-only-insecure-secret-change-me"


def _get_secret() -> str:
    secret = os.environ.get("ANGEL_API_SECRET")
    if secret:
        return secret
    if ANGEL_ENV == "production":
        # A security product should refuse to run insecure, not run quietly insecure.
        raise RuntimeError(
            "ANGEL_API_SECRET is not set. Refusing to start in production "
            "without a strong signing secret."
        )
    return _DEV_FALLBACK_SECRET


# OAuth2 bearer scheme — tokenUrl is where clients exchange credentials.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


# ── Token issue / verify ─────────────────────────────────────────────

def create_access_token(username: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,          # subject = the authenticated username
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=TOKEN_TTL_MINUTES),
    }
    return jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)


class AuthedUser:
    """The authenticated identity, derived from the verified token —
    never from anything the client asserts in a request."""

    def __init__(self, username: str, role: str):
        self.username = username
        self.role = role

    @property
    def is_advisor(self) -> bool:
        return self.role == "advisor"

    @property
    def is_client(self) -> bool:
        return self.role == "client"

    @property
    def is_admin(self) -> bool:
        return self.role == "super_admin"


_CREDENTIALS_EXC = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(token: str = Depends(oauth2_scheme)) -> AuthedUser:
    """FastAPI dependency: decode + verify the token, return the user.

    Every protected endpoint depends on this. The identity comes only
    from the cryptographically-verified token.
    """
    try:
        payload = jwt.decode(token, _get_secret(), algorithms=[ALGORITHM])
    except JWTError:
        raise _CREDENTIALS_EXC

    username = payload.get("sub")
    role = payload.get("role")
    if not username or not role:
        raise _CREDENTIALS_EXC

    return AuthedUser(username=username, role=role)


def require_advisor(user: AuthedUser = Depends(get_current_user)) -> AuthedUser:
    """Dependency for endpoints that only advisors may call."""
    if not (user.is_advisor or user.is_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action requires an advisor account.",
        )
    return user
"""Authentication request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field

from app.schemas.user import UserOut


class RegisterRequest(BaseModel):
    """Payload for user registration."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)


class LoginRequest(BaseModel):
    """Payload for login."""

    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    """Payload for refreshing an access token."""

    refresh_token: str


class TokenResponse(BaseModel):
    """Access + refresh token pair returned on register/login/refresh."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserOut

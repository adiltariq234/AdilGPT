"""
JWT Authentication system for AdilGPT.
Provides token creation, verification, and password hashing.
"""

import os
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from database import get_user_by_username, get_user_by_id, user_exists, create_user

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production-min-32-chars-long")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# ─────────────────────────────────────────────────────────────
# Password Hashing
# ─────────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a plain text password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


# ─────────────────────────────────────────────────────────────
# JWT Token Management
# ─────────────────────────────────────────────────────────────
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)

    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access"
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


# ─────────────────────────────────────────────────────────────
# FastAPI Dependencies
# ─────────────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency to get current authenticated user.
    Returns user dict or None if not authenticated.
    """
    if not credentials:
        return None

    token = credentials.credentials
    payload = decode_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = get_user_by_id(int(user_id))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User account is deactivated")

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_admin": user.is_admin
    }


async def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Strict authentication — raises 401 if not logged in.
    Use this for protected endpoints.
    """
    user = await get_current_user(credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


async def optional_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    """
    Optional authentication — returns user if logged in, None otherwise.
    Use this for endpoints that work both ways.
    """
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


# ─────────────────────────────────────────────────────────────
# Authentication Service
# ─────────────────────────────────────────────────────────────
def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user and return user data if valid."""
    user = get_user_by_username(username)
    if not user:
        return None

    if not verify_password(password, user.hashed_password):
        return None

    if not user.is_active:
        return None

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_admin": user.is_admin
    }


def register_user(username: str, password: str, email: Optional[str] = None) -> Dict[str, Any]:
    """Register a new user."""
    if user_exists(username):
        raise ValueError(f"Username '{username}' already exists")

    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters")

    if len(username) < 3 or len(username) > 50:
        raise ValueError("Username must be 3-50 characters")

    hashed = hash_password(password)
    user = create_user(username=username, hashed_password=hashed, email=email)

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "created_at": user.created_at.isoformat() if user.created_at else None
    }

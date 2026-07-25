"""
FastAPI middleware for security, rate limiting, request tracking, and monitoring.
"""

import os
import time
import uuid
import logging
from typing import Callable, Optional
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Request ID Middleware
# ─────────────────────────────────────────────────────────────
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for tracing."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        # Set request_id on the current logging context
        old_request_id = getattr(logging.getLogger().handlers[0], '_request_id', None) if logging.getLogger().handlers else None

        start_time = time.time()

        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Log request details with explicit request_id
        duration = time.time() - start_time
        extra = {'request_id': request_id}
        logger.info(
            f"{request.method} {request.url.path} "
            f"- {response.status_code} - {duration:.3f}s "
            f"- ID: {request_id}",
            extra=extra
        )

        return response


# ─────────────────────────────────────────────────────────────
# Rate Limiting Middleware
# ─────────────────────────────────────────────────────────────
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter.
    For production, replace with Redis-backed rate limiter.
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 30,
        exempt_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.exempt_paths = set(exempt_paths or ["/health", "/ready", "/docs", "/redoc", "/openapi.json"])
        self.requests = defaultdict(list)  # ip -> [timestamps]

    async def dispatch(self, request: Request, call_next):
        # Skip exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Get client IP
        client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
        if "," in str(client_ip):
            client_ip = str(client_ip).split(",")[0].strip()

        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)

        # Clean old requests and check limit
        self.requests[client_ip] = [
            ts for ts in self.requests[client_ip]
            if ts > window_start
        ]

        if len(self.requests[client_ip]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return Response(
                content='{"error": "Rate limit exceeded. Try again later."}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"}
            )

        self.requests[client_ip].append(now)

        return await call_next(request)


# ─────────────────────────────────────────────────────────────
# Security Headers Middleware
# ─────────────────────────────────────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy (adjust as needed)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' ws: wss:;"
        )

        # HSTS (only in production)
        if os.getenv("ENV") == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


# ─────────────────────────────────────────────────────────────
# Timing Middleware
# ─────────────────────────────────────────────────────────────
class TimingMiddleware(BaseHTTPMiddleware):
    """Add response time header for monitoring."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        return response
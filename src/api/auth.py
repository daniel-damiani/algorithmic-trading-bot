"""API key authentication for mutating dashboard routes."""

from __future__ import annotations

from fastapi import Header, HTTPException

from .config import dashboard_config


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = dashboard_config.api_key
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="DASHBOARD_API_KEY is not configured on the server",
        )
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

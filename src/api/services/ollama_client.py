"""Ollama HTTP API client for dashboard assistant."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp
import structlog

from ..config import dashboard_config

logger = structlog.get_logger(__name__)

DEFAULT_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))


class OllamaClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE,
        model: str = DEFAULT_MODEL,
        timeout_seconds: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def is_available(self) -> bool:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.base_url}/api/tags") as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(f"{self.base_url}/api/tags") as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
                    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception as exc:
            logger.warning("Ollama list models failed", error=str(exc))
            return []

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Ollama error {resp.status}: {text[:500]}")
                return await resp.json()

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
    ) -> AsyncIterator[str]:
        payload = {"model": self.model, "messages": messages, "stream": True}
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Ollama error {resp.status}: {text[:500]}")
                async for line in resp.content:
                    if not line:
                        continue
                    try:
                        import json

                        chunk = json.loads(line.decode("utf-8"))
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

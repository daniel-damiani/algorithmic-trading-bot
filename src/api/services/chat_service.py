"""AI assistant: Ollama + live app/Alpaca context (Alpaca MCP-compatible tools)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import structlog

from ..config import dashboard_config
from .context_builder import APP_GUIDE, build_live_context
from .ollama_client import OllamaClient
from .status_service import status_service

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = f"""You are the QuantumSentiment trading assistant embedded in the LAN dashboard.
You help the user understand their paper trading account, bot status, positions, orders, backtests, and how the app works.

Rules:
- Answer using the LIVE CONTEXT below first. If data is missing, say what you cannot see.
- You may request fresher data via tools (same data Alpaca MCP would expose for read-only queries).
- Never place real trades unless the user explicitly asks and understands risks; default is explain-only.
- This is paper trading unless context says otherwise.
- Not financial advice — educational and operational help only.
- Be concise; use bullet points for lists.

{APP_GUIDE}
"""

# Tools mirror common Alpaca MCP read operations; implemented via our status_service + dashboard state.
ASSISTANT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "refresh_alpaca_snapshot",
            "description": "Fetch fresh Alpaca paper account, positions, and recent orders.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_bot_status",
            "description": "Get trading bot process status and last heartbeat.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class ChatService:
    def __init__(self) -> None:
        self.ollama = OllamaClient(
            base_url=dashboard_config.ollama_base_url,
            model=dashboard_config.ollama_model,
            timeout_seconds=dashboard_config.ollama_timeout_seconds,
        )

    async def status(self) -> Dict[str, Any]:
        available = await self.ollama.is_available()
        models = await self.ollama.list_models() if available else []
        return {
            "ollama_available": available,
            "ollama_base_url": self.ollama.base_url,
            "ollama_model": self.ollama.model,
            "default_model": dashboard_config.ollama_model,
            "ollama_timeout_seconds": self.ollama.timeout_seconds,
            "tools_enabled": dashboard_config.ollama_tools,
            "models": models,
            "note": "Uses live Alpaca API via dashboard (same credentials as .env). "
            "Smaller models (llama3.2:1b) reply faster on CPU. For Alpaca MCP in Cursor, see Help tab.",
        }

    async def _run_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        from ..bot_manager import bot_manager

        if name == "refresh_alpaca_snapshot":
            overview = await status_service.get_overview(
                bot_manager.get_status(), None
            )
            positions = await status_service.get_positions()
            orders = await status_service.get_orders(limit=15)
            return await build_live_context(
                overview=overview,
                positions=positions,
                orders=orders,
                include_backtests=False,
            )
        if name == "get_bot_status":
            return json.dumps(bot_manager.get_status(), indent=2, default=str)
        return json.dumps({"error": f"Unknown tool: {name}"})

    async def chat(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        refresh_context: bool = True,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        use_model = (model or "").strip() or self.ollama.model
        if not await self.ollama.is_available():
            raise RuntimeError(
                f"Ollama is not reachable at {self.ollama.base_url}. "
                "Install from https://ollama.com and run: ollama pull llama3.2:1b"
            )

        overview = None
        positions: List[Dict[str, Any]] = []
        orders: List[Dict[str, Any]] = []
        if refresh_context:
            from ..bot_manager import bot_manager
            from .models_service import latest_model_summary

            overview = await status_service.get_overview(
                bot_manager.get_status(), latest_model_summary()
            )
            positions = await status_service.get_positions()
            orders = await status_service.get_orders(limit=10)

        context = await build_live_context(
            overview=overview,
            positions=positions,
            orders=orders,
            compact=True,
        )

        system_content = f"{SYSTEM_PROMPT}\n\nLIVE CONTEXT (UTC snapshot):\n{context}"
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_content}]
        for h in history or []:
            role = h.get("role", "user")
            if role in ("user", "assistant") and h.get("content"):
                messages.append({"role": role, "content": h["content"]})
        messages.append({"role": "user", "content": user_message})

        tool_rounds = 0
        max_tool_rounds = 3
        use_tools = dashboard_config.ollama_tools
        while tool_rounds < max_tool_rounds:
            try:
                response = await self.ollama.chat(
                    messages,
                    tools=ASSISTANT_TOOLS if use_tools else None,
                    model=use_model,
                )
            except RuntimeError as exc:
                if use_tools:
                    logger.warning("Ollama tools failed, retrying without tools", error=str(exc))
                    use_tools = False
                    response = await self.ollama.chat(messages, tools=None, model=use_model)
                else:
                    raise
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise RuntimeError(
                    f"Ollama did not respond within {self.ollama.timeout_seconds}s. "
                    "Try a smaller model (e.g. llama3.2:1b in OLLAMA_MODEL) or increase OLLAMA_TIMEOUT_SECONDS."
                ) from exc
            msg = response.get("message", {})
            tool_calls = (msg.get("tool_calls") or []) if use_tools else []

            if not tool_calls:
                return {
                    "reply": msg.get("content", ""),
                    "model": use_model,
                    "tool_calls_made": tool_rounds,
                }

            messages.append(msg)
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = await self._run_tool(name, args)
                tool_msg: Dict[str, Any] = {"role": "tool", "content": result}
                if tc.get("id"):
                    tool_msg["tool_call_id"] = tc["id"]
                messages.append(tool_msg)
            tool_rounds += 1

        last = messages[-1] if messages else {}
        return {
            "reply": last.get("content", "I could not complete the request."),
            "model": use_model,
            "tool_calls_made": tool_rounds,
        }


chat_service = ChatService()

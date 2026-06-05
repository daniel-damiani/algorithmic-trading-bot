"""FastAPI dashboard server for LAN monitoring and control."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .auth import require_api_key
from .backtest_manager import backtest_manager
from .bot_manager import bot_manager
from .config import dashboard_config
from .services.models_service import latest_model_summary, list_models
from .services.status_service import status_service
from .services.chat_service import chat_service
from src.feature_flags import get_features_status, update_features
from src.risk_settings import get_risk_status, update_risk_params

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    dashboard_config.ensure_dirs()
    yield


app = FastAPI(title="QuantumSentiment Dashboard", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    @app.get("/static/app.js")
    async def dashboard_app_js():
        path = STATIC_DIR / "app.js"
        if not path.exists():
            raise HTTPException(status_code=404, detail="app.js not found")
        return FileResponse(
            path,
            media_type="application/javascript",
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class FeatureFlagsUpdateRequest(BaseModel):
    enabled: Dict[str, bool] = Field(default_factory=dict)


class RiskUpdateRequest(BaseModel):
    preset: Optional[str] = None
    params: Optional[Dict[str, float]] = None


class BotStartRequest(BaseModel):
    mode: str = "paper"
    config: str = "config/config.yaml"
    symbols: Optional[List[str]] = None


class BacktestStartRequest(BaseModel):
    symbols: List[str] = Field(default_factory=lambda: ["AAPL"])
    start_date: str
    end_date: str
    capital: float = 10000.0
    config: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None
    refresh_context: bool = True
    model: Optional[str] = None


@app.get("/api/features")
async def features_list() -> Dict[str, Any]:
    return {"features": get_features_status()}


@app.post("/api/features")
async def features_update(
    body: FeatureFlagsUpdateRequest, _: None = Depends(require_api_key)
) -> Dict[str, Any]:
    if not body.enabled:
        raise HTTPException(status_code=400, detail="No feature updates provided")
    return {"features": update_features(body.enabled)}


@app.get("/api/risk")
async def risk_get() -> Dict[str, Any]:
    return get_risk_status()


@app.post("/api/risk")
async def risk_update(
    body: RiskUpdateRequest, _: None = Depends(require_api_key)
) -> Dict[str, Any]:
    if body.preset is None and not body.params:
        raise HTTPException(status_code=400, detail="Provide preset or params")
    try:
        if body.preset is not None:
            return update_risk_params(preset=body.preset)
        return update_risk_params(params=body.params or {})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "version": "1.0.0",
        "refresh_interval_seconds": dashboard_config.refresh_interval_seconds,
        "auth_required_for_controls": bool(dashboard_config.api_key),
    }


@app.get("/api/overview")
async def overview() -> Dict[str, Any]:
    bot_status = bot_manager.get_status()
    model_summary = latest_model_summary()
    return await status_service.get_overview(bot_status, model_summary)


@app.get("/api/positions")
async def positions() -> Dict[str, Any]:
    return {"positions": await status_service.get_positions()}


@app.get("/api/orders")
async def orders() -> Dict[str, Any]:
    return {"orders": await status_service.get_orders()}


@app.get("/api/pnl/history")
async def pnl_history(period: str = "3M") -> Dict[str, Any]:
    if period not in ("1M", "3M", "6M", "1A", "all"):
        raise HTTPException(
            status_code=400,
            detail="period must be one of: 1M, 3M, 6M, 1A, all",
        )
    return await status_service.get_daily_pnl_history(period=period)


@app.get("/api/models")
async def models() -> Dict[str, Any]:
    return {"models": list_models()}


@app.get("/api/bot/status")
async def bot_status() -> Dict[str, Any]:
    return bot_manager.get_status()


@app.post("/api/bot/start")
async def bot_start(
    body: BotStartRequest, _: None = Depends(require_api_key)
) -> Dict[str, Any]:
    return bot_manager.start(
        mode=body.mode,
        config_path=body.config,
        symbols=body.symbols,
    )


@app.post("/api/bot/stop")
async def bot_stop(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    return bot_manager.stop()


@app.get("/api/backtests")
async def backtests_list() -> Dict[str, Any]:
    return {"runs": backtest_manager.list_runs()}


@app.get("/api/backtests/{run_id}")
async def backtests_get(run_id: str) -> Dict[str, Any]:
    run = backtest_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Backtest run not found")
    return run


@app.get("/api/backtests/{run_id}/status")
async def backtests_status(run_id: str) -> Dict[str, Any]:
    return backtest_manager.get_status(run_id)


@app.post("/api/backtests")
async def backtests_start(
    body: BacktestStartRequest, _: None = Depends(require_api_key)
) -> Dict[str, Any]:
    if not body.symbols:
        raise HTTPException(status_code=400, detail="At least one symbol required")
    return backtest_manager.start(
        symbols=body.symbols,
        start_date=body.start_date,
        end_date=body.end_date,
        capital=body.capital,
        config_path=body.config,
    )


@app.get("/api/assistant/status")
async def assistant_status() -> Dict[str, Any]:
    return await chat_service.status()


@app.post("/api/assistant/chat")
async def assistant_chat(body: ChatRequest) -> Dict[str, Any]:
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")
    try:
        return await chat_service.chat(
            user_message=body.message.strip(),
            history=body.history,
            refresh_context=body.refresh_context,
            model=body.model,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "Ollama request timed out. Use a smaller model (llama3.2:1b) "
                "or set OLLAMA_TIMEOUT_SECONDS in .env."
            ),
        ) from exc
    except Exception as exc:
        detail = str(exc).strip() or f"{type(exc).__name__}: request failed"
        raise HTTPException(status_code=500, detail=detail) from exc


@app.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI not found")
    return FileResponse(
        index_path,
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


def main() -> None:
    if not dashboard_config.enabled:
        print("Dashboard is disabled in config (monitoring.dashboard.enabled=false)")
        return
    port = dashboard_config.port
    print(f"Dashboard: http://localhost:{port}  (LAN: use your PC's IP, not 0.0.0.0)")
    uvicorn.run(
        "src.api.server:app",
        host=dashboard_config.host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()

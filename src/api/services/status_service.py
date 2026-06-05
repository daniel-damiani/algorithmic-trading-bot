"""Live status from Alpaca paper account (independent of running bot)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)

PNL_PERIODS = frozenset({"1M", "3M", "6M", "1A", "all"})


def _parse_daily_pnl_history(raw: Dict[str, Any], period: str) -> Dict[str, Any]:
    """Turn Alpaca portfolio history (1D bars) into daily P&L rows."""
    timestamps = raw.get("timestamp") or []
    equities = raw.get("equity") or []
    days: List[Dict[str, Any]] = []
    prev_equity: Optional[float] = None

    for ts, equity in zip(timestamps, equities):
        if equity is None:
            continue
        eq = float(equity)
        date_str = datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat()
        pnl: Optional[float] = None
        pnl_percent: Optional[float] = None
        if prev_equity is not None and prev_equity > 0:
            pnl = eq - prev_equity
            # Decimal ratio (0.005 = 0.5%) — matches backtest returns and fmtPct()
            pnl_percent = pnl / prev_equity
        days.append(
            {
                "date": date_str,
                "equity": eq,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
            }
        )
        prev_equity = eq

    with_pnl = [d for d in days if d["pnl"] is not None]
    total_pnl = sum(d["pnl"] for d in with_pnl) if with_pnl else 0.0
    avg_pnl = total_pnl / len(with_pnl) if with_pnl else 0.0
    best = max(with_pnl, key=lambda d: d["pnl"], default=None)
    worst = min(with_pnl, key=lambda d: d["pnl"], default=None)
    winning_days = sum(1 for d in with_pnl if d["pnl"] > 0)

    return {
        "period": period,
        "days": days,
        "summary": {
            "total_pnl": total_pnl,
            "avg_daily_pnl": avg_pnl,
            "trading_days": len(with_pnl),
            "winning_days": winning_days,
            "losing_days": sum(1 for d in with_pnl if d["pnl"] < 0),
            "best_day": best,
            "worst_day": worst,
            "start_equity": days[0]["equity"] if days else None,
            "end_equity": days[-1]["equity"] if days else None,
        },
    }


class StatusService:
    """Cached Alpaca status for dashboard read endpoints."""

    def __init__(self) -> None:
        self._broker = None
        self._position_tracker = None
        self._account_monitor = None
        self._connected = False
        self._lock = asyncio.Lock()
        self._last_sync: Optional[datetime] = None
        self._sync_ttl_seconds = 30

    async def _ensure_connected(self) -> bool:
        async with self._lock:
            now = datetime.now(timezone.utc)
            if (
                self._connected
                and self._last_sync
                and (now - self._last_sync).total_seconds() < self._sync_ttl_seconds
            ):
                return True

            try:
                from src.broker import (
                    AccountMonitor,
                    AccountMonitorConfig,
                    AlpacaBroker,
                    PositionTracker,
                    PositionTrackerConfig,
                )

                if self._broker is None:
                    self._position_tracker = PositionTracker(PositionTrackerConfig())
                    self._account_monitor = AccountMonitor(AccountMonitorConfig())
                    self._broker = AlpacaBroker(
                        paper_trading=True,
                        position_tracker=self._position_tracker,
                        account_monitor=self._account_monitor,
                    )

                if not self._broker.is_connected:
                    connected = await self._broker.connect()
                    if not connected:
                        self._connected = False
                        return False

                await self._broker.full_sync()
                self._connected = True
                self._last_sync = now
                return True
            except Exception as exc:
                logger.warning("Status service connection failed", error=str(exc))
                self._connected = False
                return False

    async def get_account(self) -> Dict[str, Any]:
        if not await self._ensure_connected():
            return {"error": "Unable to connect to Alpaca paper account"}

        account = await self._broker.get_account()
        equity = float(account.equity)
        last_equity = float(getattr(account, "last_equity", equity) or equity)
        daily_change = equity - last_equity
        # Decimal ratio (0.005 = +0.50%), same as Alpaca portfolio day change
        daily_pct = (daily_change / last_equity) if last_equity else 0.0

        return {
            "status": account.status,
            "equity": equity,
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "daily_pnl": daily_change,
            "daily_pnl_percent": daily_pct,
        }

    async def get_positions(self) -> List[Dict[str, Any]]:
        if not await self._ensure_connected():
            return []

        from src.broker.alpaca_positions import alpaca_position_to_dict, signed_qty_from_alpaca

        stop_loss_pct = 0.03
        take_profit_pct = 0.10
        try:
            from src.risk_settings import get_effective_params

            effective = get_effective_params()
            stop_loss_pct = float(effective.get("stop_loss_pct", stop_loss_pct))
            take_profit_pct = float(effective.get("take_profit_pct", take_profit_pct))
        except Exception:
            try:
                from src.configuration import load_config

                cfg = load_config()
                stop_loss_pct = float(getattr(cfg.risk, "stop_loss_pct", stop_loss_pct))
                take_profit_pct = float(getattr(cfg.risk, "take_profit_pct", take_profit_pct))
            except Exception:
                pass

        broker_positions = await self._broker.get_positions()
        if not broker_positions:
            return []

        active = [p for p in broker_positions if abs(signed_qty_from_alpaca(p)) >= 1e-9]
        if not active:
            return []

        symbols = [p.symbol for p in active]
        prices = await self._broker.get_latest_prices(symbols)

        rows: List[Dict[str, Any]] = []
        for broker_pos in active:
            row = alpaca_position_to_dict(
                broker_pos,
                current_price=prices.get(broker_pos.symbol),
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )
            rows.append(row)
        return rows

    async def get_orders(self, limit: int = 10) -> List[Dict[str, Any]]:
        if not await self._ensure_connected():
            return []

        orders = await self._broker.get_orders(status="all", limit=limit)
        rows: List[Dict[str, Any]] = []
        for order in orders or []:
            rows.append(
                {
                    "symbol": order.symbol,
                    "side": order.side,
                    "qty": float(order.qty),
                    "order_type": order.order_type,
                    "status": order.status,
                    "submitted_at": str(order.submitted_at),
                    "filled_qty": float(getattr(order, "filled_qty", 0) or 0),
                    "filled_avg_price": float(
                        getattr(order, "filled_avg_price", 0) or 0
                    ),
                }
            )
        return rows

    async def get_health(self) -> Dict[str, Any]:
        connected = await self._ensure_connected()
        market_open = self._broker.is_market_open() if connected else False
        broker_status = self._broker.get_broker_status() if connected else {}
        monitor_status = (
            self._account_monitor.get_current_status() if connected else {}
        )
        return {
            "broker_connected": connected,
            "market_open": market_open,
            "order_summary": broker_status.get("order_summary", {}),
            "performance_metrics": monitor_status.get("performance_metrics", {}),
            "risk_metrics": monitor_status.get("risk_metrics", {}),
        }

    async def get_overview(self, bot_status: Dict[str, Any], model_summary: Any) -> Dict[str, Any]:
        account = await self.get_account()
        health = await self.get_health()
        positions = await self.get_positions()
        portfolio_pnl = sum(p.get("total_pnl", 0) for p in positions)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "account": account,
            "health": health,
            "positions_count": len(positions),
            "portfolio_pnl": portfolio_pnl,
            "bot": bot_status,
            "model": model_summary,
        }

    async def get_daily_pnl_history(self, period: str = "3M") -> Dict[str, Any]:
        if period not in PNL_PERIODS:
            period = "3M"
        if not await self._ensure_connected():
            return {
                "error": "Unable to connect to Alpaca paper account",
                "period": period,
                "days": [],
                "summary": {},
            }
        try:
            raw = await self._broker.get_portfolio_history(
                period=period,
                timeframe="1D",
            )
            if not raw or not raw.get("timestamp"):
                return {
                    "period": period,
                    "days": [],
                    "summary": {},
                    "note": "No portfolio history returned from Alpaca.",
                }
            return _parse_daily_pnl_history(raw, period)
        except Exception as exc:
            logger.warning("Daily P&L history failed", error=str(exc))
            return {
                "error": str(exc),
                "period": period,
                "days": [],
                "summary": {},
            }


status_service = StatusService()

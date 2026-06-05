"""Helpers for mapping Alpaca position entities to internal/API formats."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .position_exits import compute_exit_levels


def signed_qty_from_alpaca(broker_pos) -> float:
    """Alpaca reports positive qty with side long/short — normalize to signed quantity."""
    qty = float(getattr(broker_pos, "qty", 0) or 0)
    side = str(getattr(broker_pos, "side", "") or "").lower()
    if side == "short" and qty > 0:
        return -qty
    return qty


def alpaca_position_to_dict(
    broker_pos,
    current_price: Optional[float] = None,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.15,
) -> Dict[str, Any]:
    """Map an Alpaca position entity to dashboard/API row format."""
    qty = signed_qty_from_alpaca(broker_pos)
    avg = float(getattr(broker_pos, "avg_entry_price", 0) or 0)
    price = float(
        current_price
        or getattr(broker_pos, "current_price", None)
        or avg
        or 0
    )
    market_value = float(getattr(broker_pos, "market_value", None) or abs(qty * price))
    unrealized = float(getattr(broker_pos, "unrealized_pl", 0) or 0)
    cost = abs(qty * avg) if avg else 0.0
    side = "short" if qty < 0 else "long"
    row = {
        "symbol": broker_pos.symbol,
        "side": side,
        "quantity": qty,
        "average_price": avg,
        "current_price": price,
        "market_value": market_value,
        "unrealized_pnl": unrealized,
        "realized_pnl": 0.0,
        "total_pnl": unrealized,
        "pnl_percent": (unrealized / cost * 100) if cost > 0 else 0.0,
    }
    row.update(
        compute_exit_levels(
            side=side,
            avg_entry=avg,
            mark=price,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )
    )
    return row

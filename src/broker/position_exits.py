"""Shared stop-loss / take-profit calculations for long and short positions."""

from __future__ import annotations

from typing import Any, Dict


def compute_exit_levels(
    *,
    side: str,
    avg_entry: float,
    mark: float,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.15,
) -> Dict[str, Any]:
    """
    Compute profit/loss percentages and trigger prices.

    For shorts: profit when mark falls below entry; stop when mark rises above entry.
    """
    if avg_entry <= 0 or mark <= 0:
        return {}

    is_short = str(side).lower() == "short"
    if is_short:
        profit_pct = (avg_entry - mark) / avg_entry
        loss_pct = (mark - avg_entry) / avg_entry
        take_profit_price = avg_entry * (1 - take_profit_pct)
        stop_loss_price = avg_entry * (1 + stop_loss_pct)
    else:
        profit_pct = (mark - avg_entry) / avg_entry
        loss_pct = (avg_entry - mark) / avg_entry
        take_profit_price = avg_entry * (1 + take_profit_pct)
        stop_loss_price = avg_entry * (1 - stop_loss_pct)

    tp_hit = profit_pct > take_profit_pct
    sl_hit = loss_pct > stop_loss_pct

    return {
        "entry_price": round(avg_entry, 4),
        "profit_pct": round(profit_pct * 100, 2),
        "take_profit_price": round(take_profit_price, 2),
        "stop_loss_price": round(stop_loss_price, 2),
        "take_profit_triggered": tp_hit,
        "stop_loss_triggered": sl_hit,
        "distance_to_take_profit_pct": round(
            max(0.0, (take_profit_pct - profit_pct) * 100), 2
        ),
    }

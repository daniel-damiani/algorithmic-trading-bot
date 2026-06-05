"""Build LLM context from dashboard state, bot heartbeat, and app knowledge."""



from __future__ import annotations



import json

from datetime import datetime, timezone

from typing import Any, Dict, List, Optional



from ..bot_manager import bot_manager

from ..backtest_manager import backtest_manager

from .models_service import latest_model_summary





APP_GUIDE = """

QuantumSentiment: algorithmic paper trading via Alpaca.

Model: MarketRegimeXGBoost (5-class XGBoost). Bot: python src/main.py --mode paper.

Dashboard tabs: Overview, P&L, Positions, Backtest, Models, Activity, Assistant, Help.

""".strip()





def _format_bot(bot_status: Dict[str, Any]) -> str:

    hb = bot_status.get("last_heartbeat") or {}

    ts = hb.get("timestamp") or "never"

    return (

        f"running={bot_status.get('running')} pid={bot_status.get('pid')} "

        f"last_heartbeat={ts}"

    )





def _format_model(model_summary: Optional[Dict[str, Any]]) -> str:

    if not model_summary:

        return "none loaded"

    name = model_summary.get("name", "?")

    version = model_summary.get("version", "?")

    acc = model_summary.get("metrics", {}).get("accuracy")

    acc_s = f" accuracy={acc}" if acc is not None else ""

    return f"{name} @ {version}{acc_s}"





def _format_account(overview: Dict[str, Any]) -> str:

    acct = overview.get("account") or {}

    health = overview.get("health") or {}

    if acct.get("error"):

        return f"error: {acct['error']}"

    pct = acct.get("daily_pnl_percent")
    pct_str = f"{pct * 100:.2f}%" if isinstance(pct, (int, float)) else "—"
    lines = [
        f"equity={acct.get('equity')} buying_power={acct.get('buying_power')}",
        f"daily_pnl={acct.get('daily_pnl')} ({pct_str})",
        f"market_open={health.get('market_open')} broker_connected={health.get('broker_connected')}",
        f"open_positions={overview.get('positions_count', 0)} portfolio_pnl={overview.get('portfolio_pnl')}",
    ]

    return "\n".join(lines)





def _format_positions(positions: List[Dict[str, Any]]) -> str:

    if not positions:

        return "none"

    lines = []

    for p in positions[:10]:

        lines.append(

            f"- {p.get('symbol')}: qty={p.get('qty')} avg={p.get('avg_entry_price')} "

            f"current={p.get('current_price')} unrealized={p.get('unrealized_pnl')}"

        )

    if len(positions) > 10:

        lines.append(f"... and {len(positions) - 10} more")

    return "\n".join(lines)





def _format_orders(orders: List[Dict[str, Any]]) -> str:

    if not orders:

        return "none"

    lines = []

    for o in orders[:8]:

        lines.append(

            f"- {o.get('submitted_at')} {o.get('symbol')} {o.get('side')} "

            f"qty={o.get('qty')} status={o.get('status')}"

        )

    return "\n".join(lines)





def _format_backtests(max_runs: int) -> str:

    runs = backtest_manager.list_runs()[:max_runs]

    if not runs:

        return "none"

    lines = []

    for r in runs:

        meta = r.get("meta") or {}

        summary = r.get("summary") or {}

        lines.append(

            f"- {r.get('id')}: {meta.get('start_date')}→{meta.get('end_date')} "

            f"return={summary.get('total_return')} trades={summary.get('total_trades')}"

        )

    return "\n".join(lines)





async def build_live_context(

    overview: Optional[Dict[str, Any]] = None,

    positions: Optional[List[Dict[str, Any]]] = None,

    orders: Optional[List[Dict[str, Any]]] = None,

    include_backtests: bool = True,

    max_backtests: int = 2,

    compact: bool = True,

) -> str:

    """Serialize current app + trading state for the assistant."""

    bot_status = bot_manager.get_status()

    model_summary = latest_model_summary()

    positions = positions if positions is not None else []

    orders = orders if orders is not None else []



    if compact:

        sections = [

            f"Snapshot (UTC): {datetime.now(timezone.utc).isoformat()}",

            APP_GUIDE,

            f"Bot: {_format_bot(bot_status)}",

            f"Model: {_format_model(model_summary)}",

        ]

        if overview:

            sections.extend(["Account:", _format_account(overview)])

        sections.extend([

            f"Positions ({len(positions)}):",

            _format_positions(positions),

            f"Recent orders ({len(orders)}):",

            _format_orders(orders),

        ])

        if include_backtests:

            sections.extend(["Recent backtests:", _format_backtests(max_backtests)])

        return "\n".join(sections)



    sections: List[str] = [

        f"Snapshot time (UTC): {datetime.now(timezone.utc).isoformat()}",

        "",

        "## App",

        APP_GUIDE,

        "",

        "## Bot process",

        json.dumps(bot_status, indent=2, default=str),

        "",

        "## Loaded model",

        json.dumps(model_summary or {"note": "no model found"}, indent=2, default=str),

    ]



    if overview:

        sections.extend(["", "## Alpaca paper account (live)", json.dumps(overview, indent=2, default=str)])



    if positions is not None:

        sections.extend([

            "",

            f"## Open positions ({len(positions)})",

            json.dumps(positions, indent=2, default=str) if positions else "No open positions.",

        ])



    if orders is not None:

        sections.extend([

            "",

            f"## Recent orders ({len(orders)})",

            json.dumps(orders[:15], indent=2, default=str) if orders else "No recent orders.",

        ])



    if include_backtests:

        runs = backtest_manager.list_runs()[:max_backtests]

        sections.extend([

            "",

            "## Recent backtest runs",

            json.dumps(runs, indent=2, default=str) if runs else "No saved backtests.",

        ])



    return "\n".join(sections)


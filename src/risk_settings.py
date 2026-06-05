"""
Runtime risk parameters for the trading bot and dashboard.

Persisted to cache/dashboard/risk_settings.json. Defaults to the low-risk preset.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = PROJECT_ROOT / "cache" / "dashboard" / "risk_settings.json"

DEFAULT_PRESET = "low"

# Keys shared by presets, bot, and UI
RISK_KEYS = (
    "stop_loss_pct",
    "take_profit_pct",
    "risk_per_trade",
    "max_drawdown",
    "daily_loss_limit",
    "max_leverage",
    "max_position_size",
)

RISK_PRESETS: Dict[str, Dict[str, Any]] = {
    "low": {
        "id": "low",
        "label": "Low risk",
        "tagline": "Conservative — capital preservation",
        "description": (
            "Tighter stops (3%), modest targets (10%), and small position sizes (10% max). "
            "Stops new trading after a 3% daily loss or 10% drawdown from equity high. "
            "Best when learning the system or protecting gains."
        ),
        "values": {
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.10,
            "risk_per_trade": 0.015,
            "max_drawdown": 0.10,
            "daily_loss_limit": 0.03,
            "max_leverage": 1.0,
            "max_position_size": 0.10,
        },
    },
    "medium": {
        "id": "medium",
        "label": "Medium risk",
        "tagline": "Balanced — moderate growth",
        "description": (
            "Moderate stops (4%) and targets (12%) with 15% max position size. "
            "Daily loss cap 5%, max drawdown 15%. A middle ground between "
            "protection and opportunity."
        ),
        "values": {
            "stop_loss_pct": 0.04,
            "take_profit_pct": 0.12,
            "risk_per_trade": 0.02,
            "max_drawdown": 0.15,
            "daily_loss_limit": 0.05,
            "max_leverage": 1.5,
            "max_position_size": 0.15,
        },
    },
    "high": {
        "id": "high",
        "label": "High risk",
        "tagline": "Aggressive — larger swings",
        "description": (
            "Wide stops (5%), larger targets (15%), up to 20% per position and 2× leverage. "
            "Allows 8% daily loss and 25% drawdown before halting new trades. "
            "Higher reward potential with higher volatility."
        ),
        "values": {
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.15,
            "risk_per_trade": 0.03,
            "max_drawdown": 0.25,
            "daily_loss_limit": 0.08,
            "max_leverage": 2.0,
            "max_position_size": 0.20,
        },
    },
}

RISK_PARAM_DEFS: List[Dict[str, Any]] = [
    {
        "key": "stop_loss_pct",
        "label": "Stop loss",
        "hint": "Close position when price moves this % against you (entry-based).",
        "unit": "percent",
        "min": 0.01,
        "max": 0.10,
        "step": 0.005,
    },
    {
        "key": "take_profit_pct",
        "label": "Take profit",
        "hint": "Close position when price moves this % in your favor (entry-based).",
        "unit": "percent",
        "min": 0.05,
        "max": 0.25,
        "step": 0.01,
    },
    {
        "key": "risk_per_trade",
        "label": "Risk per trade",
        "hint": "Target portfolio fraction allocated to each new position.",
        "unit": "percent",
        "min": 0.005,
        "max": 0.05,
        "step": 0.005,
    },
    {
        "key": "max_position_size",
        "label": "Max position size",
        "hint": "Maximum single-position weight vs account equity.",
        "unit": "percent",
        "min": 0.05,
        "max": 0.30,
        "step": 0.01,
    },
    {
        "key": "max_drawdown",
        "label": "Max drawdown",
        "hint": "Halt new trades if equity falls this % below session high.",
        "unit": "percent",
        "min": 0.05,
        "max": 0.30,
        "step": 0.01,
    },
    {
        "key": "daily_loss_limit",
        "label": "Daily loss limit",
        "hint": "Halt new trades after this % account loss in one day.",
        "unit": "percent",
        "min": 0.01,
        "max": 0.15,
        "step": 0.01,
    },
    {
        "key": "max_leverage",
        "label": "Max leverage",
        "hint": "Cap on total exposure vs equity (1.0 = no leverage).",
        "unit": "multiple",
        "min": 1.0,
        "max": 3.0,
        "step": 0.1,
    },
]


def _ensure_settings_file() -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _clamp_param(key: str, value: float) -> float:
    for defn in RISK_PARAM_DEFS:
        if defn["key"] == key:
            return float(max(defn["min"], min(defn["max"], value)))
    return float(value)


def _validate_params(params: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    defaults = RISK_PRESETS[DEFAULT_PRESET]["values"]
    for key in RISK_KEYS:
        raw = params.get(key, defaults[key])
        try:
            out[key] = _clamp_param(key, float(raw))
        except (TypeError, ValueError):
            out[key] = float(defaults[key])
    return out


def _load_raw() -> Dict[str, Any]:
    _ensure_settings_file()
    if not SETTINGS_FILE.exists():
        return {
            "preset": DEFAULT_PRESET,
            "params": deepcopy(RISK_PRESETS[DEFAULT_PRESET]["values"]),
        }
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        preset = data.get("preset") or DEFAULT_PRESET
        if preset not in RISK_PRESETS and preset != "custom":
            preset = DEFAULT_PRESET
        params = _validate_params(data.get("params") or RISK_PRESETS.get(preset, RISK_PRESETS[DEFAULT_PRESET])["values"])
        return {"preset": preset, "params": params}
    except Exception:
        return {
            "preset": DEFAULT_PRESET,
            "params": deepcopy(RISK_PRESETS[DEFAULT_PRESET]["values"]),
        }


def _save_raw(data: Dict[str, Any]) -> None:
    _ensure_settings_file()
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_effective_params() -> Dict[str, float]:
    """Current risk parameters (used by bot and position exit math)."""
    return _validate_params(_load_raw()["params"])


def get_risk_status() -> Dict[str, Any]:
    if not SETTINGS_FILE.exists():
        _save_raw(_load_raw())
    raw = _load_raw()
    presets = []
    for pid, preset in RISK_PRESETS.items():
        presets.append(
            {
                "id": pid,
                "label": preset["label"],
                "tagline": preset["tagline"],
                "description": preset["description"],
                "values": preset["values"],
            }
        )
    return {
        "preset": raw["preset"],
        "params": raw["params"],
        "effective": get_effective_params(),
        "presets": presets,
        "param_defs": RISK_PARAM_DEFS,
        "default_preset": DEFAULT_PRESET,
    }


def apply_preset(preset_id: str) -> Dict[str, Any]:
    if preset_id not in RISK_PRESETS:
        raise ValueError(f"Unknown preset: {preset_id}")
    params = deepcopy(RISK_PRESETS[preset_id]["values"])
    _save_raw({"preset": preset_id, "params": params})
    return get_risk_status()


def update_risk_params(
    params: Optional[Dict[str, Any]] = None,
    preset: Optional[str] = None,
) -> Dict[str, Any]:
    if preset is not None:
        return apply_preset(preset)

    raw = _load_raw()
    merged = {**raw["params"]}
    if params:
        for key, val in params.items():
            if key in RISK_KEYS:
                merged[key] = _clamp_param(key, float(val))
    merged = _validate_params(merged)

    # Detect if params still match a named preset
    matched = "custom"
    for pid, preset_def in RISK_PRESETS.items():
        if all(abs(merged[k] - preset_def["values"][k]) < 1e-9 for k in RISK_KEYS):
            matched = pid
            break

    _save_raw({"preset": matched, "params": merged})
    return get_risk_status()


def apply_params_to_config(config: Any) -> Dict[str, float]:
    """Patch a loaded Config object's risk/trading sections from runtime settings."""
    params = get_effective_params()
    if hasattr(config, "risk"):
        risk = config.risk
        for key in (
            "stop_loss_pct",
            "take_profit_pct",
            "risk_per_trade",
            "max_drawdown",
            "daily_loss_limit",
            "max_leverage",
        ):
            if hasattr(risk, key):
                setattr(risk, key, params[key])
    if hasattr(config, "trading") and hasattr(config.trading, "max_position_size"):
        config.trading.max_position_size = params["max_position_size"]
    return params

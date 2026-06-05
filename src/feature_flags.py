"""
Runtime feature toggles for optional data sources and integrations.

Persisted to cache/dashboard/feature_flags.json so the dashboard and bot share state.
Defaults: enabled only when required API keys are present (except always-available sources).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FLAGS_FILE = PROJECT_ROOT / "cache" / "dashboard" / "feature_flags.json"

load_dotenv(PROJECT_ROOT / ".env")

FEATURE_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "reddit",
        "label": "Reddit sentiment",
        "description": "Posts and comments from WSB, stocks, investing, etc.",
        "env_vars": ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"],
        "requires_all_env": True,
    },
    {
        "id": "newsapi",
        "label": "NewsAPI",
        "description": "Financial news headlines via newsapi.org.",
        "env_vars": ["NEWSAPI_KEY"],
        "requires_all_env": True,
    },
    {
        "id": "alpha_vantage",
        "label": "Alpha Vantage news",
        "description": "Company news feed from Alpha Vantage.",
        "env_vars": ["ALPHA_VANTAGE_API_KEY"],
        "requires_all_env": True,
    },
    {
        "id": "news_rss",
        "label": "RSS news feeds",
        "description": "Free RSS feeds (no API key). May add latency.",
        "env_vars": [],
        "requires_all_env": True,
        "default_when_available": True,
    },
    {
        "id": "dynamic_discovery",
        "label": "Dynamic symbol discovery",
        "description": "Expand watchlist from trending mentions (uses enabled sentiment sources).",
        "env_vars": [],
        "requires_all_env": True,
        "default_when_available": False,
    },
    {
        "id": "unusual_whales",
        "label": "Unusual Whales",
        "description": "Congressional trading scrape via unusualwhales.com (Playwright). Cached ~15 min per cycle.",
        "env_vars": [],
        "requires_all_env": True,
        "default_when_available": False,
    },
]


def api_key_valid(key: Optional[str]) -> bool:
    if not key or not str(key).strip():
        return False
    lowered = str(key).strip().lower()
    return not (
        lowered.startswith("your_")
        or lowered in ("changeme", "placeholder", "none", "xxx", "todo")
    )


def _feature_available(defn: Dict[str, Any]) -> tuple[bool, List[str]]:
    env_vars = defn.get("env_vars") or []
    if not env_vars:
        return True, []
    missing = [v for v in env_vars if not api_key_valid(os.getenv(v))]
    if defn.get("requires_all_env", True):
        return len(missing) == 0, missing
    return len(missing) < len(env_vars), missing


def _default_enabled(defn: Dict[str, Any], available: bool) -> bool:
    if not available:
        return False
    if "default_when_available" in defn:
        return bool(defn["default_when_available"])
    return True


def _ensure_flags_file() -> None:
    FLAGS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_saved_enabled() -> Dict[str, bool]:
    _ensure_flags_file()
    if not FLAGS_FILE.exists():
        return {}
    try:
        with open(FLAGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return dict(data.get("enabled") or {})
    except Exception:
        return {}


def _save_enabled(enabled: Dict[str, bool]) -> None:
    _ensure_flags_file()
    with open(FLAGS_FILE, "w", encoding="utf-8") as f:
        json.dump({"enabled": enabled}, f, indent=2)


def get_default_enabled_map() -> Dict[str, bool]:
    defaults: Dict[str, bool] = {}
    for defn in FEATURE_DEFINITIONS:
        available, _ = _feature_available(defn)
        defaults[defn["id"]] = _default_enabled(defn, available)
    return defaults


def get_enabled_map() -> Dict[str, bool]:
    """Merged user toggles with defaults for unset features."""
    defaults = get_default_enabled_map()
    saved = _load_saved_enabled()
    merged = {**defaults}
    for fid, val in saved.items():
        if fid in defaults:
            merged[fid] = bool(val)
    return merged


def is_active(feature_id: str) -> bool:
    """True when feature is available, user-enabled, and keys (if any) are present."""
    for defn in FEATURE_DEFINITIONS:
        if defn["id"] == feature_id:
            available, _ = _feature_available(defn)
            enabled = get_enabled_map().get(feature_id, False)
            return available and enabled
    return False


def get_features_status() -> List[Dict[str, Any]]:
    enabled_map = get_enabled_map()
    rows: List[Dict[str, Any]] = []
    for defn in FEATURE_DEFINITIONS:
        fid = defn["id"]
        available, missing = _feature_available(defn)
        enabled = enabled_map.get(fid, False)
        rows.append(
            {
                "id": fid,
                "label": defn["label"],
                "description": defn["description"],
                "available": available,
                "enabled": enabled,
                "active": available and enabled,
                "missing_env": missing,
                "env_vars": defn.get("env_vars") or [],
            }
        )
    return rows


def update_features(enabled_updates: Dict[str, bool]) -> List[Dict[str, Any]]:
    current = get_enabled_map()
    for fid, val in enabled_updates.items():
        if fid not in {d["id"] for d in FEATURE_DEFINITIONS}:
            continue
        available, _ = _feature_available(next(d for d in FEATURE_DEFINITIONS if d["id"] == fid))
        if not available and val:
            current[fid] = False
        else:
            current[fid] = bool(val)
    _save_enabled(current)
    return get_features_status()


def news_sources_active() -> set[str]:
    sources: set[str] = set()
    if is_active("alpha_vantage"):
        sources.add("alpha_vantage")
    if is_active("newsapi"):
        sources.add("newsapi")
    if is_active("news_rss"):
        sources.add("rss")
    return sources

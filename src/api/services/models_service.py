"""Scan trained model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..config import dashboard_config


def list_models(models_root: Path | None = None) -> List[Dict[str, Any]]:
    root = models_root or (dashboard_config.project_root / "models")
    if not root.exists():
        return []

    results: List[Dict[str, Any]] = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        versions = sorted(
            [
                d
                for d in model_dir.iterdir()
                if d.is_dir() and (d / "model.xgb").exists()
            ],
            key=lambda p: p.name,
            reverse=True,
        )
        if not versions:
            continue
        latest = versions[0]
        entry: Dict[str, Any] = {
            "name": model_dir.name,
            "version": latest.name,
            "path": str(latest.relative_to(dashboard_config.project_root)),
        }
        meta_path = latest / "metadata.json"
        features_path = latest / "features.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                entry["metadata"] = json.load(f)
        if features_path.exists():
            with open(features_path, encoding="utf-8") as f:
                entry["features"] = json.load(f)
        results.append(entry)
    return results


def latest_model_summary() -> Dict[str, Any] | None:
    models = list_models()
    if not models:
        return None
    preferred = next(
        (m for m in models if m["name"] == "MarketRegimeXGBoost"), models[0]
    )
    metrics = (preferred.get("metadata") or {}).get("metrics", {})
    return {
        "name": preferred["name"],
        "version": preferred["version"],
        "metrics": metrics,
    }

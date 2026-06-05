"""Shared helpers for saving training artifacts."""

from pathlib import Path
from typing import Dict, List, Optional

import xgboost as xgb

from src.models.sklearn_xgboost_predictor import save_sklearn_xgboost_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def save_massive_training_model(
    model: xgb.XGBClassifier,
    feature_names: List[str],
    metrics: Optional[Dict[str, float]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save train_simple_massive output for backtest loading."""
    out = output_dir or (PROJECT_ROOT / "models")
    return save_sklearn_xgboost_model(
        model=model,
        feature_names=feature_names,
        output_dir=out,
        metrics=metrics,
        model_name="MarketRegimeXGBoost",
    )

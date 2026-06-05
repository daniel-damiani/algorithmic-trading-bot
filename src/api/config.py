"""Dashboard configuration loaded from config.yaml and environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DASHBOARD_CACHE = PROJECT_ROOT / "cache" / "dashboard"
BACKTESTS_CACHE = DASHBOARD_CACHE / "backtests"

load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml_dashboard() -> dict:
    try:
        from src.configuration import load_config

        cfg = load_config()
        section = getattr(getattr(cfg, "monitoring", None), "dashboard", None)
        if section is None:
            return {}
        if hasattr(section, "to_dict"):
            return section.to_dict()
        return dict(section.__dict__) if hasattr(section, "__dict__") else {}
    except Exception:
        return {}


class DashboardConfig:
    def __init__(self) -> None:
        yaml_cfg = _load_yaml_dashboard()
        self.host: str = yaml_cfg.get("host", "0.0.0.0")
        self.port: int = int(yaml_cfg.get("port", 8000))
        self.enabled: bool = bool(yaml_cfg.get("enabled", True))
        self.refresh_interval_seconds: int = int(
            yaml_cfg.get("refresh_interval_seconds", 5)
        )
        api_key_env = yaml_cfg.get("api_key_env", "DASHBOARD_API_KEY")
        self.api_key: Optional[str] = os.getenv(api_key_env) or os.getenv(
            "DASHBOARD_API_KEY"
        )
        self.project_root = PROJECT_ROOT
        self.cache_dir = DASHBOARD_CACHE
        self.backtests_dir = BACKTESTS_CACHE
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        self.ollama_timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))
        self.ollama_tools = os.getenv("OLLAMA_TOOLS", "").lower() in ("1", "true", "yes")

    def ensure_dirs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backtests_dir.mkdir(parents=True, exist_ok=True)


dashboard_config = DashboardConfig()

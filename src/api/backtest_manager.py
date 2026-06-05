"""Backtest job subprocess management."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from .config import dashboard_config
from .serializers import to_jsonable

logger = structlog.get_logger(__name__)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2)


class BacktestManager:
    def list_runs(self) -> List[Dict[str, Any]]:
        dashboard_config.ensure_dirs()
        runs: List[Dict[str, Any]] = []
        if not dashboard_config.backtests_dir.exists():
            return runs
        for run_dir in sorted(dashboard_config.backtests_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            meta = _read_json(run_dir / "meta.json")
            status = _read_json(run_dir / "status.json")
            results = _read_json(run_dir / "results.json")
            runs.append(
                {
                    "id": run_dir.name,
                    "meta": meta,
                    "status": status,
                    "summary": {
                        "total_return": results.get("total_return"),
                        "sharpe_ratio": results.get("sharpe_ratio"),
                        "total_trades": results.get("total_trades"),
                    }
                    if results
                    else None,
                }
            )
        return runs

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        run_dir = dashboard_config.backtests_dir / run_id
        if not run_dir.is_dir():
            return None
        return {
            "id": run_id,
            "meta": _read_json(run_dir / "meta.json"),
            "status": _read_json(run_dir / "status.json"),
            "results": _read_json(run_dir / "results.json"),
        }

    def get_status(self, run_id: str) -> Dict[str, Any]:
        run_dir = dashboard_config.backtests_dir / run_id
        status = _read_json(run_dir / "status.json")
        if not status:
            return {"state": "unknown", "run_id": run_id}
        return status

    def start(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        capital: float = 10000.0,
        config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        dashboard_config.ensure_dirs()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = dashboard_config.backtests_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "run_id": run_id,
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "capital": capital,
            "config": config_path or "config/config.yaml",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(run_dir / "meta.json", meta)
        _write_json(
            run_dir / "status.json",
            {"state": "queued", "run_id": run_id, "updated_at": meta["created_at"]},
        )

        root = dashboard_config.project_root
        cmd = [
            sys.executable,
            str(root / "backtest.py"),
            "--symbols",
            *symbols,
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--capital",
            str(capital),
            "--output-dir",
            str(run_dir),
        ]
        if config_path:
            cmd.extend(["--config", config_path])

        log_path = run_dir / "backtest.log"
        log_file = open(log_path, "w", encoding="utf-8")
        kwargs: Dict[str, Any] = {"cwd": str(root), "stdout": log_file, "stderr": subprocess.STDOUT}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(cmd, **kwargs)
        _write_json(
            run_dir / "status.json",
            {
                "state": "running",
                "run_id": run_id,
                "pid": proc.pid,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info("Started backtest subprocess", run_id=run_id, pid=proc.pid)
        return {"ok": True, "run_id": run_id, "pid": proc.pid}


backtest_manager = BacktestManager()

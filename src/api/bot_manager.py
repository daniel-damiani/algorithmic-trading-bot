"""Paper trading bot subprocess lifecycle."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from .config import dashboard_config

logger = structlog.get_logger(__name__)

BOT_STATE_FILE = dashboard_config.cache_dir / "bot_state.json"
BOT_HEARTBEAT_FILE = dashboard_config.cache_dir / "bot_heartbeat.json"
BOT_LOG_FILE = dashboard_config.cache_dir / "bot.log"
HEARTBEAT_STALE_SECONDS = 480
STARTUP_GRACE_SECONDS = 600


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    dashboard_config.ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                check=False,
            )
            return str(pid) in result.stdout
        except Exception:
            return False
    try:
        import os

        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _heartbeat_age_seconds(heartbeat: Dict[str, Any]) -> Optional[float]:
    ts = _parse_iso(heartbeat.get("timestamp"))
    if not ts:
        return None
    return (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()


class BotManager:
    def get_status(self) -> Dict[str, Any]:
        state = _read_json(BOT_STATE_FILE)
        heartbeat = _read_json(BOT_HEARTBEAT_FILE)
        pid = int(state.get("pid", 0) or 0)
        running = _is_process_running(pid) if pid else False

        if state.get("status") == "running" and pid and not running:
            state["status"] = "stopped"
            state["stopped_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(BOT_STATE_FILE, state)

        hb_age = _heartbeat_age_seconds(heartbeat)
        started_at = _parse_iso(state.get("started_at"))
        uptime_seconds = (
            (datetime.now(timezone.utc) - started_at.astimezone(timezone.utc)).total_seconds()
            if started_at
            else None
        )

        heartbeat_stale = False
        if running:
            if hb_age is not None:
                heartbeat_stale = hb_age > HEARTBEAT_STALE_SECONDS
            elif uptime_seconds is not None and uptime_seconds > STARTUP_GRACE_SECONDS:
                heartbeat_stale = True

        healthy = running and not heartbeat_stale and bool(heartbeat.get("timestamp"))
        log_tail = self.get_log_tail(30)

        return {
            "running": running,
            "healthy": healthy,
            "heartbeat_stale": heartbeat_stale,
            "pid": pid if running else None,
            "mode": state.get("mode", "paper"),
            "started_at": state.get("started_at"),
            "uptime_seconds": uptime_seconds,
            "heartbeat_age_seconds": hb_age,
            "status": "running" if running else state.get("status", "stopped"),
            "last_heartbeat": heartbeat,
            "log_file": str(BOT_LOG_FILE),
            "log_tail": log_tail,
            "diagnosis": self._diagnose(running, healthy, heartbeat_stale, heartbeat, log_tail),
        }

    def _diagnose(
        self,
        running: bool,
        healthy: bool,
        heartbeat_stale: bool,
        heartbeat: Dict[str, Any],
        log_tail: List[str],
    ) -> Optional[str]:
        if not running:
            return None
        if healthy:
            return None
        if not heartbeat.get("timestamp"):
            if any("Failed to initialize" in line for line in log_tail):
                return "Bot process exited during startup — see log below."
            return (
                "Process is running but has not completed a trading cycle yet. "
                "First startup can take several minutes (model load). Check bot.log."
            )
        if heartbeat_stale:
            note = heartbeat.get("note") or ""
            if note.startswith("analyzing"):
                return (
                    f"Bot is working ({note}) — cycles take several minutes without API keys. "
                    "This is normal; heartbeat will refresh when the cycle completes."
                )
            return (
                f"No heartbeat in {HEARTBEAT_STALE_SECONDS // 60}+ minutes — bot may be hung. "
                "Stop and restart from Overview."
            )
        return "Bot status is degraded — check bot.log."

    def get_log_tail(self, lines: int = 50) -> List[str]:
        if not BOT_LOG_FILE.exists():
            return []
        try:
            content = BOT_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
            return content[-lines:]
        except Exception:
            return []

    def start(
        self,
        mode: str = "paper",
        config_path: str = "config/config.yaml",
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        current = self.get_status()
        if current["running"]:
            return {"ok": False, "error": "Bot is already running", "status": current}

        dashboard_config.ensure_dirs()
        root = dashboard_config.project_root
        BOT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(BOT_LOG_FILE, "a", encoding="utf-8", buffering=1)
        log_handle.write(
            f"\n--- bot start {datetime.now(timezone.utc).isoformat()} mode={mode} ---\n"
        )
        log_handle.flush()
        cmd = [
            sys.executable,
            str(root / "src" / "main.py"),
            "--mode",
            mode,
            "--config",
            config_path,
        ]
        if symbols:
            cmd.extend(["--symbols", *symbols])

        kwargs: Dict[str, Any] = {
            "cwd": str(root),
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(cmd, **kwargs)
        started_at = datetime.now(timezone.utc).isoformat()
        state = {
            "pid": proc.pid,
            "mode": mode,
            "started_at": started_at,
            "status": "running",
            "config": config_path,
            "symbols": symbols or [],
        }
        _write_json(BOT_STATE_FILE, state)
        logger.info("Started trading bot subprocess", pid=proc.pid, mode=mode)
        return {"ok": True, "pid": proc.pid, "started_at": started_at}

    def stop(self) -> Dict[str, Any]:
        state = _read_json(BOT_STATE_FILE)
        pid = int(state.get("pid", 0) or 0)
        if not pid or not _is_process_running(pid):
            state["status"] = "stopped"
            state["stopped_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(BOT_STATE_FILE, state)
            return {"ok": True, "message": "Bot was not running"}

        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            else:
                import os
                import signal

                os.killpg(os.getpgid(pid), signal.SIGTERM)
                time.sleep(2)
                if _is_process_running(pid):
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception as exc:
            logger.error("Failed to stop bot", pid=pid, error=str(exc))
            return {"ok": False, "error": str(exc)}

        state["status"] = "stopped"
        state["stopped_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(BOT_STATE_FILE, state)
        return {"ok": True, "pid": pid, "stopped_at": state["stopped_at"]}


bot_manager = BotManager()

#!/usr/bin/env python3
"""
Setup validation for quick_start.py and README.

Checks required environment variables and database connectivity
without loading the full ML stack.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    errors = []

    if not Path(".env").exists():
        errors.append("Missing .env file")

    if not os.getenv("ALPACA_API_KEY"):
        errors.append("Missing ALPACA_API_KEY in .env")
    else:
        print("OK  ALPACA_API_KEY")

    if not os.getenv("ALPACA_API_SECRET"):
        errors.append("Missing ALPACA_API_SECRET in .env")
    else:
        print("OK  ALPACA_API_SECRET")

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        errors.append("Missing DATABASE_URL in .env")
    elif db_url.startswith("postgresql"):
        try:
            from sqlalchemy import create_engine, text

            with create_engine(db_url).connect() as conn:
                conn.execute(text("SELECT 1"))
            print("OK  PostgreSQL connection")
        except Exception as e:
            errors.append(f"PostgreSQL connection failed: {e}")
    else:
        print(f"OK  Database URL configured ({db_url.split('://')[0]})")

    for config in ("config/config.yaml",):
        if not Path(config).exists():
            errors.append(f"Missing {config}")
        else:
            print(f"OK  {config}")

    if errors:
        print("\nSetup validation failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("\nSetup validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

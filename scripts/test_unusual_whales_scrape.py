#!/usr/bin/env python3
"""Smoke test Unusual Whales congressional trade scraping."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sentiment.unusual_whales_analyzer import (
    UnusualWhalesAnalyzer,
    UnusualWhalesConfig,
    to_fusion_payload,
)


def main() -> int:
    print("=== Unusual Whales scrape test ===")
    cfg = UnusualWhalesConfig(lookback_days=90, timeout=60, cache_ttl_seconds=900)
    analyzer = UnusualWhalesAnalyzer(cfg)

    print("Initializing Playwright/Selenium...")
    ok = analyzer.initialize()
    print(f"  initialize() -> {ok}")
    print(f"  scraper_ready -> {analyzer.scraper_ready}")

    if not analyzer.scraper_ready:
        print("\nFAIL: Scraper not ready. Run: playwright install chromium")
        return 1

    print("\nFetching all congressional trades (JSON API)...")
    from datetime import datetime, timedelta

    cutoff = datetime.utcnow() - timedelta(days=cfg.lookback_days)
    all_trades = analyzer._get_trades_from_json_api("", cutoff)
    print(f"  Total trades returned: {len(all_trades)}")
    if all_trades:
        sample = all_trades[0]
        print(
            f"  Sample: {sample.get('politician')} "
            f"{sample.get('trade_type')} {sample.get('symbol')} "
            f"${sample.get('value', 0):,.0f}"
        )

    for symbol in ("PLTR", "NVDA", "AAPL"):
        print(f"\nAnalyzing {symbol}...")
        result = analyzer.analyze_symbol(symbol)
        payload = to_fusion_payload(result)
        print(f"  congress_trades={result.get('total_congress_trades')}")
        print(f"  insider_trades={result.get('total_insider_trades')}")
        print(f"  political_sentiment={result.get('political_sentiment')}")
        print(f"  fusion_payload={payload}")

    analyzer.cleanup()
    print("\nPASS: Scrape test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Refresh cached SEC earnings dates and Yahoo Finance prices.

Usage:
    python scripts/refresh_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_windows.analytics import window_stats  # noqa: E402
from meta_windows.data import fetch_earnings_dates, fetch_prices  # noqa: E402
from meta_windows.windows import build_windows  # noqa: E402


def main() -> None:
    print("Refreshing earnings dates from SEC EDGAR…")
    earnings = fetch_earnings_dates(refresh=True)
    print(
        f"  {len(earnings)} quarterly 8-K Item 2.02 filings "
        f"({earnings['earnings_date'].min():%d %b %Y} → "
        f"{earnings['earnings_date'].max():%d %b %Y})"
    )

    print("Refreshing prices from Yahoo Finance…")
    prices = fetch_prices(refresh=True)
    print(
        f"  {len(prices):,} trading days "
        f"({prices.index.min():%d %b %Y} → {prices.index.max():%d %b %Y})"
    )

    windows = build_windows(earnings["earnings_date"], prices.index)
    stats = window_stats(prices, windows)
    completed = stats.loc[~stats["is_partial"]]
    print(
        f"Built {len(windows)} windows · average completed return "
        f"{completed['return_pct'].mean() * 100:+.1f}% · win rate "
        f"{(completed['return_pct'] > 0).mean() * 100:.0f}%"
    )


if __name__ == "__main__":
    main()

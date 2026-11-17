"""Data access: SEC EDGAR earnings dates and Yahoo Finance prices.

Both sources are cached under ``.cache/`` so repeated dashboard runs are fast
and resilient to temporary upstream failures.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import (
    CACHE_DIR,
    CIK,
    EARNINGS_CACHE_TTL_DAYS,
    EARNINGS_HISTORY_START,
    PRICE_CACHE_TTL_HOURS,
    PRICE_HISTORY_START,
    SEC_USER_AGENT,
    TICKER,
)

SEC_SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{CIK}.json"
SEC_ARCHIVE_URL = "https://data.sec.gov/submissions/{name}"
YAHOO_CHART_HOSTS = (
    "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
)

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def _is_fresh(path: Path, max_age: dt.timedelta) -> bool:
    if not path.exists():
        return False
    modified = dt.datetime.fromtimestamp(path.stat().st_mtime)
    return dt.datetime.now() - modified < max_age


# ---------------------------------------------------------------------------
# Earnings dates from SEC EDGAR
# ---------------------------------------------------------------------------


def _sec_get_json(url: str) -> dict:
    response = requests.get(
        url,
        headers={
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _iter_filing_rows(payload: dict):
    """Yield filing rows from either the 'recent' block or an archive block.

    Both are columnar: each key maps to a list of values indexed by filing.
    """
    forms = payload.get("form", [])
    for i in range(len(forms)):
        yield {
            "form": forms[i],
            "filing_date": payload["filingDate"][i],
            "items": payload.get("items", [None] * len(forms))[i],
            "accession_number": payload.get("accessionNumber", [None] * len(forms))[i],
            "primary_document": payload.get("primaryDocument", [None] * len(forms))[i],
        }


def _is_earnings_release(items: str | None) -> bool:
    if not items:
        return False
    return "2.02" in {item.strip() for item in items.split(",")}


def fetch_earnings_dates(refresh: bool = False) -> pd.DataFrame:
    """Return one row per quarterly earnings release, sourced from 8-K Item 2.02.

    The 8-K filing date is used as the earnings-release date. Meta files its
    earnings press release with Item 2.02 on the release day.
    """
    cache = _cache_path("earnings_dates.csv")
    ttl = dt.timedelta(days=EARNINGS_CACHE_TTL_DAYS)
    if not refresh and _is_fresh(cache, ttl):
        return pd.read_csv(cache, parse_dates=["earnings_date"])

    root = _sec_get_json(SEC_SUBMISSIONS_URL)
    filings = root.get("filings", {})
    payloads = [filings.get("recent", {})]

    for archive in filings.get("files", []):
        time.sleep(0.15)  # stay well inside SEC's 10 requests/second limit
        payloads.append(_sec_get_json(SEC_ARCHIVE_URL.format(name=archive["name"])))

    rows = []
    for payload in payloads:
        for row in _iter_filing_rows(payload):
            if row["form"] != "8-K" or not _is_earnings_release(row["items"]):
                continue
            rows.append(
                {
                    "earnings_date": pd.Timestamp(row["filing_date"]).normalize(),
                    "accession_number": row["accession_number"],
                    "primary_document": row["primary_document"],
                }
            )

    if not rows:
        raise RuntimeError("SEC EDGAR returned no earnings 8-K filings for Meta.")

    earnings = (
        pd.DataFrame(rows)
        .drop_duplicates("earnings_date")
        .sort_values("earnings_date")
        .reset_index(drop=True)
    )
    earnings = earnings[earnings["earnings_date"] >= pd.Timestamp(EARNINGS_HISTORY_START)]

    earnings.to_csv(cache, index=False)
    return earnings


# ---------------------------------------------------------------------------
# Prices from Yahoo Finance
# ---------------------------------------------------------------------------


def _yahoo_chart_payload() -> dict:
    period1 = int(pd.Timestamp(PRICE_HISTORY_START, tz="UTC").timestamp())
    period2 = int(dt.datetime.now(dt.timezone.utc).timestamp()) + 86_400
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    last_error: Exception | None = None
    for template in YAHOO_CHART_HOSTS:
        try:
            response = requests.get(
                template.format(ticker=TICKER),
                params=params,
                headers={"User-Agent": BROWSER_USER_AGENT, "Accept": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            result = (payload.get("chart") or {}).get("result") or []
            if not result:
                raise RuntimeError(
                    f"Yahoo Finance chart API returned no result: {payload}"
                )
            return result[0]
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_error = exc
    raise RuntimeError("Yahoo Finance chart API unavailable") from last_error


def _parse_yahoo_chart(result: dict) -> pd.DataFrame:
    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    if not timestamps or not quote.get("close"):
        raise RuntimeError("Yahoo Finance chart payload is missing price columns.")

    index = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert("America/New_York")
        .normalize()
        .tz_localize(None)
    )
    frame = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=index,
    )

    adjclose_block = (result.get("indicators", {}).get("adjclose") or [{}])[0]
    adjclose = pd.Series(
        adjclose_block.get("adjclose"), index=index, dtype="float64"
    )
    frame = frame.astype("float64")

    # Scale OHLC onto the total-return adjusted series so entry prices, exit
    # prices and returns all include dividends and stay internally consistent.
    factor = (adjclose / frame["close"]).replace([np.inf, -np.inf], np.nan)
    factor = factor.fillna(1.0)
    for column in ("open", "high", "low"):
        frame[column] = frame[column] * factor
    frame["close"] = adjclose.fillna(frame["close"] * factor)

    frame.index.name = "date"
    frame = frame.dropna(subset=["close"]).sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def fetch_prices(refresh: bool = False) -> pd.DataFrame:
    """Return daily OHLCV for META from the IPO to today.

    Yahoo's META history includes the period when the ticker was FB. Prices are
    dividend-adjusted (Meta has paid a quarterly dividend since 2024).
    """
    cache = _cache_path("prices.csv")
    ttl = dt.timedelta(hours=PRICE_CACHE_TTL_HOURS)
    if not refresh and _is_fresh(cache, ttl):
        return pd.read_csv(cache, index_col="date", parse_dates=["date"])

    try:
        prices = _parse_yahoo_chart(_yahoo_chart_payload())
    except Exception:
        if cache.exists():
            # Serve a stale cache rather than failing outright.
            return pd.read_csv(cache, index_col="date", parse_dates=["date"])
        raise

    if prices.empty:
        raise RuntimeError("Yahoo Finance returned no price data for META.")

    prices.to_csv(cache)
    return prices

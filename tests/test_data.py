from __future__ import annotations

import numpy as np
import pandas as pd

from meta_windows.data import _is_earnings_release, _iter_filing_rows, _parse_yahoo_chart


def test_is_earnings_release_matches_exact_item_code():
    assert _is_earnings_release("2.02,9.01")
    assert _is_earnings_release("1.01, 2.02, 8.01, 9.01")
    assert not _is_earnings_release("12.02")
    assert not _is_earnings_release("")
    assert not _is_earnings_release(None)


def test_iter_filing_rows_supports_columnar_payloads():
    payload = {
        "form": ["8-K", "10-Q"],
        "filingDate": ["2024-01-31", "2024-02-01"],
        "items": ["2.02,9.01", ""],
        "accessionNumber": ["a", "b"],
        "primaryDocument": ["x.htm", "y.htm"],
    }
    rows = list(_iter_filing_rows(payload))

    assert len(rows) == 2
    assert rows[0]["form"] == "8-K"
    assert rows[0]["filing_date"] == "2024-01-31"
    assert rows[1]["form"] == "10-Q"


def test_parse_yahoo_chart_applies_adjustment_factor():
    # Yahoo daily timestamps sit at the market open (14:30 UTC in January).
    timestamps = [
        int(pd.Timestamp("2024-01-02 14:30", tz="UTC").timestamp()),
        int(pd.Timestamp("2024-01-03 14:30", tz="UTC").timestamp()),
    ]
    result = {
        "timestamp": timestamps,
        "indicators": {
            "quote": [
                {
                    "open": [90.0, 101.0],
                    "high": [105.0, 106.0],
                    "low": [88.0, 99.0],
                    "close": [100.0, 102.0],
                    "volume": [1_000, 2_000],
                }
            ],
            "adjclose": [{"adjclose": [50.0, 51.0]}],
        },
    }

    frame = _parse_yahoo_chart(result)

    assert len(frame) == 2
    assert frame.index.min() == pd.Timestamp("2024-01-02")
    # Adjustment factor on day 1 is 0.5, so open becomes 45 and close 50.
    assert frame.loc["2024-01-02", "open"] == 45.0
    assert frame.loc["2024-01-02", "close"] == 50.0
    assert frame.loc["2024-01-02", "volume"] == 1_000
    assert np.isclose(frame.loc["2024-01-03", "close"], 51.0)

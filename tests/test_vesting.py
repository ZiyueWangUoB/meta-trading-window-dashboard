from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meta_windows.vesting import _vest_date, vesting_analysis

INDEX = pd.bdate_range("2024-01-02", "2024-03-29")


def make_prices() -> pd.DataFrame:
    closes = 100.0 + np.arange(len(INDEX), dtype=float) * 0.1
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.full(len(INDEX), 1_000_000.0),
        },
        index=INDEX,
    )


def make_windows(close_date: str = "2024-02-29") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": [1],
            "label": ["Feb 2024"],
            "year": [2024],
            "month_name": ["Feb"],
            "earnings_date": [pd.Timestamp("2024-01-31")],
            "open_date": [pd.Timestamp("2024-01-30")],
            "close_date": [pd.Timestamp(close_date)],
            "n_days": [23],
        }
    )


def test_vest_date_rolls_to_next_trading_day():
    # 15 June 2024 is a Saturday; the next business day is Monday the 17th.
    trading = pd.bdate_range("2024-06-03", "2024-06-28")
    vest = _vest_date(pd.Timestamp("2024-06-28"), trading)
    assert vest == pd.Timestamp("2024-06-17")


def test_vesting_analysis_hold_and_lag_returns():
    prices = make_prices()
    result = vesting_analysis(prices, make_windows(), n_boot=200)

    window = result["windows"].iloc[0]
    vest = pd.Timestamp("2024-02-15")
    hold = float(prices.loc["2024-02-29", "close"]) / float(prices.loc[vest, "close"]) - 1.0

    assert window["vest_date"] == vest
    assert window["hold_to_close"] == pytest.approx(hold)
    assert window["peak_date"] == pd.Timestamp("2024-02-29")
    assert window["peak_last5"]

    lag = result["lag"]
    assert lag["lag"].tolist() == list(range(0, 11))  # 11 post-vest trading days
    assert lag.loc[lag["lag"] == 0, "mean"].iloc[0] == 0.0
    assert lag.loc[lag["lag"] == 3, "n"].iloc[0] == 1
    # A positively drifting price series makes every later sale better.
    assert (lag["mean"].drop(index=0) > 0).all()

    rules = result["rules"].set_index("rule")
    assert rules.loc["vest", "mean"] == 0.0
    assert rules.loc["last", "mean"] == pytest.approx(hold)


def test_vesting_calendar_and_peak_counts():
    result = vesting_analysis(make_prices(), make_windows(), n_boot=100)

    calendar = result["calendar"].set_index("dom")
    assert calendar.loc[15, "mean"] == 0.0
    assert calendar.loc[15, "mean_percentile"] > 0
    assert calendar.loc[26, "n"] >= 1

    peaks = result["peaks"].set_index("dom")
    assert peaks["count"].sum() == 1
    assert peaks.loc[29, "count"] == 1  # data rises monotonically to month-end


def test_vesting_analysis_ignores_partial_windows():
    prices = make_prices()
    window = make_windows(close_date="2024-04-30")  # beyond the price history

    result = vesting_analysis(prices, window, n_boot=100)

    assert result["windows"].empty
    assert result["rules"].empty

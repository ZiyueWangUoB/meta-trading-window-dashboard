from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meta_windows.analytics import (
    day_of_window_profile,
    equity_curves,
    equity_summary,
    heatmap_returns,
    regime_stats,
    window_stats,
    window_trajectories,
)

INDEX = pd.bdate_range("2024-01-01", "2024-03-29")
WINDOW = pd.DataFrame(
    {
        "window_id": [1],
        "label": ["Feb 2024"],
        "year": [2024],
        "month_name": ["Feb"],
        "earnings_date": [pd.Timestamp("2024-01-24")],
        "open_date": [pd.Timestamp("2024-01-26")],
        "close_date": [pd.Timestamp("2024-02-29")],
        "n_days": [25],
    }
)


def make_prices(closes: np.ndarray | None = None) -> pd.DataFrame:
    if closes is None:
        closes = 100.0 + np.arange(len(INDEX))
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.full(len(closes), 1_000_000.0),
        },
        index=INDEX,
    )


def test_window_stats_uses_entry_open_and_exit_close():
    prices = make_prices()
    stats = window_stats(prices, WINDOW).iloc[0]

    days = prices.loc["2024-01-26":"2024-02-29"]
    expected = days["close"].iloc[-1] / days["open"].iloc[0] - 1
    assert stats["return_pct"] == pytest.approx(expected)
    assert stats["n_days"] == 25
    assert not stats["is_partial"]


def test_max_drawdown_uses_entry_price_as_initial_peak():
    closes = np.full(len(INDEX), 100.0)
    start = INDEX.get_loc(pd.Timestamp("2024-01-26"))
    closes[start] = 100.0
    closes[start + 1] = 110.0
    closes[start + 2] = 90.0
    closes[start + 3] = 100.0
    prices = make_prices(closes)

    stats = window_stats(prices, WINDOW).iloc[0]
    assert stats["max_drawdown_pct"] == pytest.approx(90.0 / 110.0 - 1.0)


def test_equity_curves_have_two_series_and_compound_windows():
    prices = make_prices()
    curves = equity_curves(prices, WINDOW)
    summary = equity_summary(curves)

    assert set(curves["series"]) == {"Open-window only", "Buy & hold"}
    assert set(summary["series"]) == {"Open-window only", "Buy & hold"}
    stats = window_stats(prices, WINDOW).iloc[0]
    strategy_final = summary.loc[
        summary["series"] == "Open-window only", "final_value"
    ].iloc[0]
    assert strategy_final == pytest.approx(100.0 * (1 + stats["return_pct"]))


def test_regime_stats_partition_days():
    prices = make_prices()
    regimes = regime_stats(prices, WINDOW)

    assert len(regimes) == 2
    counts = regimes.set_index("regime")["n_days"]
    # One return is lost to the first price observation.
    assert counts.sum() == len(prices) - 1
    assert counts["Open window"] > 0
    assert counts["Blackout"] > 0


def test_trajectories_start_at_100_and_profile_is_defined():
    prices = make_prices()
    trajectories = window_trajectories(prices, WINDOW)

    assert trajectories.groupby("window_id")["value"].first().iloc[0] == 100.0
    first_window = trajectories.loc[trajectories["window_id"] == 1]
    assert first_window["day"].iloc[0] == 0
    assert len(first_window) == WINDOW.iloc[0]["n_days"] + 1

    profile = day_of_window_profile(prices, WINDOW)
    assert profile["day"].min() == 1
    assert profile["n_windows"].max() == 1


def test_heatmap_returns_pivots_by_year_and_month():
    prices = make_prices()
    stats = window_stats(prices, WINDOW)
    pivot = heatmap_returns(stats)

    assert list(pivot.columns) == ["Feb"]
    assert list(pivot.index) == [2024]
    assert pivot.iloc[0, 0] == pytest.approx(stats["return_pct"].iloc[0])


def test_partial_window_flagged_when_close_is_beyond_prices():
    prices = make_prices()
    window = WINDOW.copy()
    window["close_date"] = pd.Timestamp("2024-04-30")

    stats = window_stats(prices, window).iloc[0]
    assert stats["is_partial"]

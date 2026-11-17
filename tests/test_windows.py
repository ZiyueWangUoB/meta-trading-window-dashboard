from __future__ import annotations

import pandas as pd

from meta_windows.windows import (
    assign_window_days,
    build_windows,
    windows_overlap,
)


def trading_days(start: str = "2024-01-01", end: str = "2024-12-31") -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def test_window_opens_on_second_trading_day_after_earnings():
    windows = build_windows(pd.to_datetime(["2024-01-24"]), trading_days())

    assert len(windows) == 1
    window = windows.iloc[0]
    assert window["open_date"] == pd.Timestamp("2024-01-26")
    assert window["close_date"] == pd.Timestamp("2024-02-29")
    assert window["label"] == "Feb 2024"
    assert window["month_name"] == "Feb"
    assert window["n_days"] == 25


def test_window_label_uses_middle_month_even_when_it_opens_in_prior_month():
    # Q2 2024 earnings were released on 31 July; the window opens in July but
    # runs through the August middle-month close.
    windows = build_windows(pd.to_datetime(["2024-07-31"]), trading_days())

    window = windows.iloc[0]
    assert window["open_date"] == pd.Timestamp("2024-08-02")
    assert window["close_date"] == pd.Timestamp("2024-08-30")  # Aug 31 is Saturday
    assert window["label"] == "Aug 2024"
    assert window["month_name"] == "Aug"


def test_earnings_before_price_history_are_skipped():
    windows = build_windows(
        pd.to_datetime(["2011-10-26", "2024-01-24"]),
        trading_days(),
    )
    assert len(windows) == 1
    assert windows.iloc[0]["label"] == "Feb 2024"


def test_window_keeps_policy_end_when_end_is_in_the_future():
    trading = pd.bdate_range("2024-01-01", "2024-11-15")
    windows = build_windows(pd.to_datetime(["2024-10-23"]), trading)

    window = windows.iloc[0]
    assert window["open_date"] == pd.Timestamp("2024-10-25")
    assert window["close_date"] == pd.Timestamp("2024-11-30")
    # n_days counts only observed trading days.
    assert window["n_days"] == 16


def test_quarters_do_not_overlap():
    earnings = pd.to_datetime(
        ["2024-01-24", "2024-04-24", "2024-07-31", "2024-10-23"]
    )
    windows = build_windows(earnings, trading_days())
    assert len(windows) == 4
    assert not windows_overlap(windows)


def test_assign_window_days_numbers_trading_days():
    prices = pd.DataFrame(
        {"close": range(len(trading_days())), "volume": 1},
        index=trading_days(),
    )
    windows = build_windows(pd.to_datetime(["2024-01-24"]), prices.index)
    annotated = assign_window_days(prices, windows)

    in_window = annotated.loc[annotated["in_window"]]
    assert in_window["day_in_window"].tolist() == list(
        range(1, windows.iloc[0]["n_days"] + 1)
    )
    assert in_window.index.min() == pd.Timestamp("2024-01-26")
    assert not annotated.loc["2024-03-01", "in_window"]
    assert pd.isna(annotated.loc["2024-03-01", "window_id"])

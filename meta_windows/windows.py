"""Derive Meta's employee open-trading windows from earnings dates.

Policy encoded here (from Meta's 2024 Form 10-K):

* The closed trading period begins at market close on the last day of the
  middle month of each quarter — February, May, August and November.
* The window reopens on the second full trading day after the quarterly
  earnings release, and stays open until the next middle-month close.

The result is one open window per quarter, roughly four to six weeks long.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MIDDLE_MONTH_NAMES, MIDDLE_MONTHS, WINDOW_OPEN_TRADING_DAY_OFFSET

WINDOW_COLUMNS = [
    "window_id",
    "label",
    "year",
    "month_name",
    "earnings_date",
    "open_date",
    "close_date",
    "n_days",
]


def _next_middle_month_end(day: pd.Timestamp) -> pd.Timestamp:
    """Return the first Feb/May/Aug/Nov month-end on or after ``day``."""
    day = pd.Timestamp(day).normalize()
    year, month = day.year, day.month
    for _ in range(24):
        if month in MIDDLE_MONTHS:
            month_end = (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)).normalize()
            if month_end >= day:
                return month_end
        month += 1
        if month > 12:
            month = 1
            year += 1
    raise ValueError(f"Could not find a middle-month end after {day.date()}")


def build_windows(earnings_dates, trading_days) -> pd.DataFrame:
    """Build one open-trading-window row per earnings release.

    Parameters
    ----------
    earnings_dates:
        Iterable of earnings release dates (8-K Item 2.02 filing dates).
    trading_days:
        DatetimeIndex-like collection of exchange trading days; the price
        history index is the natural input.
    """
    trading = pd.DatetimeIndex(pd.to_datetime(list(trading_days))).normalize()
    trading = pd.DatetimeIndex(sorted(set(trading)))

    earnings = (
        pd.Series(pd.to_datetime(list(earnings_dates)))
        .dt.normalize()
        .sort_values()
        .unique()
        .tolist()
    )

    records = []
    for earnings_date in earnings:
        earnings_date = pd.Timestamp(earnings_date)
        if len(trading) == 0 or earnings_date < trading[0]:
            # No price history covers this release (e.g. before the IPO).
            continue
        after = trading[trading > earnings_date]
        if len(after) < WINDOW_OPEN_TRADING_DAY_OFFSET:
            continue

        open_date = pd.Timestamp(after[WINDOW_OPEN_TRADING_DAY_OFFSET - 1])
        target_end = _next_middle_month_end(open_date)
        if target_end > trading[-1]:
            # The window is still open (or its end is beyond the last quote).
            # Keep the policy end date so the dashboard can mark it partial.
            close_date = target_end
        else:
            on_or_before = trading[trading <= target_end]
            if len(on_or_before) == 0:
                continue
            close_date = pd.Timestamp(on_or_before[-1])
        if close_date < open_date:
            continue

        records.append(
            {
                "earnings_date": earnings_date,
                "open_date": open_date,
                "close_date": close_date,
            }
        )

    if not records:
        return pd.DataFrame(columns=WINDOW_COLUMNS)

    windows = pd.DataFrame(records).drop_duplicates(subset=["open_date", "close_date"])
    windows = windows.sort_values("open_date").reset_index(drop=True)
    windows["window_id"] = np.arange(1, len(windows) + 1)
    # Label by the middle month the window runs through - e.g. earnings in
    # late July open the window in July but it is "Aug" for policy purposes.
    windows["label"] = windows["close_date"].dt.strftime("%b %Y")
    windows["year"] = windows["close_date"].dt.year
    windows["month_name"] = windows["close_date"].dt.month.map(MIDDLE_MONTH_NAMES)
    windows["n_days"] = [
        int(((trading >= open_date) & (trading <= close_date)).sum())
        for open_date, close_date in zip(windows["open_date"], windows["close_date"])
    ]

    return windows[WINDOW_COLUMNS]


def assign_window_days(prices: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Annotate each price row with its window id and day number (1-based)."""
    frame = prices.copy()
    frame["window_id"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["day_in_window"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    index = frame.index.normalize()
    for _, window in windows.iterrows():
        mask = (index >= window["open_date"]) & (index <= window["close_date"])
        n_matching = int(mask.sum())
        if n_matching == 0:
            continue
        frame.loc[mask, "window_id"] = int(window["window_id"])
        frame.loc[mask, "day_in_window"] = np.arange(1, n_matching + 1)

    frame["in_window"] = frame["window_id"].notna()
    return frame


def windows_overlap(windows: pd.DataFrame) -> bool:
    """True if any two windows share at least one calendar day."""
    ordered = windows.sort_values("open_date").reset_index(drop=True)
    for i in range(1, len(ordered)):
        if ordered.loc[i, "open_date"] <= ordered.loc[i - 1, "close_date"]:
            return True
    return False

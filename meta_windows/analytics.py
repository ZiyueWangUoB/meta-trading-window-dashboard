"""Metrics for Meta's price action inside and outside trading windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .windows import assign_window_days

TRADING_DAYS_PER_YEAR = 252
MONTH_ORDER = ["Feb", "May", "Aug", "Nov"]


def window_stats(prices: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Per-window performance and risk metrics.

    ``return_pct`` is the tradeable window return: buy at the first window
    day's open, sell at the last window day's close. Volatility and drawdown
    are computed from daily closes inside the window.
    """
    rows: list[dict] = []
    last_price_date = prices.index.max()

    for window in windows.itertuples(index=False):
        mask = (prices.index >= window.open_date) & (prices.index <= window.close_date)
        days = prices.loc[mask]
        if days.empty:
            continue

        entry_price = float(days["open"].iloc[0])
        exit_price = float(days["close"].iloc[-1])
        window_return = exit_price / entry_price - 1.0

        daily = prices["close"].pct_change().loc[days.index].dropna()
        volatility = (
            float(daily.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
            if len(daily) > 1
            else np.nan
        )

        closes = days["close"].to_numpy(dtype=float)
        path = np.concatenate([[entry_price], closes])
        running_peak = np.maximum.accumulate(path)
        max_drawdown = float((path / running_peak - 1.0).min())

        highs = days["high"].to_numpy(dtype=float)
        lows = days["low"].to_numpy(dtype=float)
        price_range = float(highs.max() / lows.min() - 1.0) if len(lows) else np.nan

        volume = days["volume"].astype(float)
        prior_volume = prices.loc[prices.index < window.open_date, "volume"].astype(float).tail(60)
        volume_ratio = (
            float(volume.mean() / prior_volume.mean())
            if len(prior_volume) and float(prior_volume.mean()) > 0
            else np.nan
        )

        rows.append(
            {
                "window_id": int(window.window_id),
                "label": window.label,
                "year": int(window.year),
                "month_name": window.month_name,
                "earnings_date": window.earnings_date,
                "open_date": window.open_date,
                "close_date": window.close_date,
                "n_days": int(len(days)),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return_pct": window_return,
                "max_drawdown_pct": max_drawdown,
                "volatility_pct": volatility,
                "range_pct": price_range,
                "up_day_share": float((daily > 0).mean()) if len(daily) else np.nan,
                "best_day_pct": float(daily.max()) if len(daily) else np.nan,
                "worst_day_pct": float(daily.min()) if len(daily) else np.nan,
                "avg_volume": float(volume.mean()),
                "volume_ratio": volume_ratio,
                "is_partial": bool(
                    window.open_date <= last_price_date < window.close_date
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("open_date").reset_index(drop=True)
    return frame


def regime_stats(prices: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Daily return statistics for open-window days versus blackout days."""
    annotated = assign_window_days(prices, windows)
    returns = prices["close"].pct_change()
    frame = pd.DataFrame(
        {"ret": returns, "in_window": annotated["in_window"]}
    ).dropna()

    rows = []
    for flag, label in ((True, "Open window"), (False, "Blackout")):
        values = frame.loc[frame["in_window"] == flag, "ret"]
        n_days = int(len(values))
        if n_days == 0:
            continue
        total_return = float((1.0 + values).prod() - 1.0)
        annualized_return = (1.0 + total_return) ** (
            TRADING_DAYS_PER_YEAR / n_days
        ) - 1.0
        annualized_vol = float(values.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        rows.append(
            {
                "regime": label,
                "n_days": n_days,
                "share_of_days": n_days / len(frame),
                "total_return": total_return,
                "annualized_return": float(annualized_return),
                "annualized_vol": annualized_vol,
                "sharpe": (
                    float(values.mean()) * TRADING_DAYS_PER_YEAR / annualized_vol
                    if annualized_vol > 0
                    else np.nan
                ),
                "avg_daily_return": float(values.mean()),
                "hit_rate": float((values > 0).mean()),
                "best_day": float(values.max()),
                "worst_day": float(values.min()),
            }
        )

    return pd.DataFrame(rows)


def equity_curves(prices: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Two equity series indexed to 100 at the first window's open.

    * ``Open-window only`` compounds each window's tradeable return and holds
      cash between windows.
    * ``Buy & hold`` tracks META over the same calendar span.
    """
    stats = window_stats(prices, windows)
    stats = stats.dropna(subset=["return_pct"])
    if stats.empty:
        return pd.DataFrame(columns=["date", "value", "series"])

    last_available = pd.Timestamp(prices.index.max())
    strategy_points: list[tuple[pd.Timestamp, float]] = []
    level = 100.0
    for row in stats.itertuples(index=False):
        open_date = pd.Timestamp(row.open_date)
        exit_date = min(pd.Timestamp(row.close_date), last_available)
        if exit_date < open_date:
            continue
        strategy_points.append((open_date, level))
        level *= 1.0 + float(row.return_pct)
        strategy_points.append((exit_date, level))

    if not strategy_points:
        return pd.DataFrame(columns=["date", "value", "series"])

    start = strategy_points[0][0]
    end = strategy_points[-1][0]
    strategy = pd.DataFrame(strategy_points, columns=["date", "value"])
    strategy["series"] = "Open-window only"

    span = prices.loc[(prices.index >= start) & (prices.index <= end)]
    if span.empty:
        return strategy

    base_price = float(span["open"].iloc[0])
    buy_hold = pd.DataFrame(
        {
            "date": span.index,
            "value": span["close"].to_numpy(dtype=float) / base_price * 100.0,
        }
    )
    buy_hold["series"] = "Buy & hold"

    return pd.concat([strategy, buy_hold], ignore_index=True)


def equity_summary(curves: pd.DataFrame) -> pd.DataFrame:
    """CAGR and max drawdown for each equity series."""
    rows = []
    for series, frame in curves.groupby("series", sort=False):
        frame = frame.sort_values("date")
        if frame.empty:
            continue
        start = pd.Timestamp(frame["date"].iloc[0])
        end = pd.Timestamp(frame["date"].iloc[-1])
        years = max((end - start).days / 365.25, 1e-9)
        final_value = float(frame["value"].iloc[-1])
        values = frame["value"].to_numpy(dtype=float)
        running_peak = np.maximum.accumulate(values)
        max_drawdown = float((values / running_peak - 1.0).min())
        rows.append(
            {
                "series": series,
                "start": start,
                "end": end,
                "final_value": final_value,
                "cagr": (final_value / 100.0) ** (1.0 / years) - 1.0,
                "max_drawdown": max_drawdown,
            }
        )
    return pd.DataFrame(rows)


def window_trajectories(prices: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Price path of every window, indexed to 100 at the entry open.

    Day 0 is the entry at the first open; day ``k`` is the close of the k-th
    trading day in the window.
    """
    records = []
    for window in windows.itertuples(index=False):
        mask = (prices.index >= window.open_date) & (prices.index <= window.close_date)
        days = prices.loc[mask]
        if days.empty:
            continue
        entry = float(days["open"].iloc[0])
        values = np.concatenate([[100.0], days["close"].to_numpy(dtype=float) / entry * 100.0])
        records.append(
            pd.DataFrame(
                {
                    "window_id": int(window.window_id),
                    "label": window.label,
                    "year": int(window.year),
                    "month_name": window.month_name,
                    "day": np.arange(len(values)),
                    "value": values,
                }
            )
        )
    if not records:
        return pd.DataFrame(
            columns=["window_id", "label", "year", "month_name", "day", "value"]
        )
    return pd.concat(records, ignore_index=True)


def day_of_window_profile(prices: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Average daily return by position within the window."""
    trajectories = window_trajectories(prices, windows)
    if trajectories.empty:
        return pd.DataFrame(
            columns=["day", "avg_return", "median_return", "hit_rate", "n_windows"]
        )

    trajectories = trajectories.sort_values(["window_id", "day"]).copy()
    trajectories["prev_value"] = trajectories.groupby("window_id")["value"].shift(1)
    daily = trajectories.dropna(subset=["prev_value"]).copy()
    daily["return"] = daily["value"] / daily["prev_value"] - 1.0

    profile = (
        daily.groupby("day")
        .agg(
            avg_return=("return", "mean"),
            median_return=("return", "median"),
            hit_rate=("return", lambda values: float((values > 0).mean())),
            n_windows=("return", "size"),
        )
        .reset_index()
    )
    return profile


def heatmap_returns(stats: pd.DataFrame) -> pd.DataFrame:
    """Year x window-month matrix of window returns."""
    if stats.empty:
        return pd.DataFrame()
    pivot = stats.pivot_table(
        index="year", columns="month_name", values="return_pct", aggfunc="mean"
    )
    columns = [month for month in MONTH_ORDER if month in pivot.columns]
    return pivot.reindex(columns=columns).sort_index()

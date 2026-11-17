"""Vesting-date analysis: when, after the 15th, has selling been best?

Meta RSUs vest on the 15th of the trade-window month (or the next trading day
when the 15th is a weekend/holiday). Employees can only sell inside the open
window, so the decision is effectively: sell on the vest date, wait a few
days, or wait until the window closes at month-end.

All returns here are close-to-close and measured from the vest-day close.
Positive means the later sale price was higher than selling on the vest date.
Only completed windows are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analytics import window_stats

RULE_LABELS = {
    "vest": "Sell on vest day",
    "plus_3": "Vest + 3 days",
    "plus_5": "Vest + 5 days",
    "plus_7": "Vest + 7 days",
    "plus_10": "Vest + 10 days",
    "dom_20": "20th of month",
    "dom_25": "25th of month",
    "dom_26": "First day ≥ 26th",
    "last": "Window close",
}
RULE_ORDER = list(RULE_LABELS)


def _vest_date(close_date: pd.Timestamp, trading: pd.DatetimeIndex) -> pd.Timestamp | None:
    target = pd.Timestamp(year=close_date.year, month=close_date.month, day=15)
    candidates = trading[(trading >= target) & (trading <= close_date)]
    return pd.Timestamp(candidates[0]) if len(candidates) else None


def _bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _empty_result() -> dict[str, pd.DataFrame]:
    return {
        "windows": pd.DataFrame(),
        "lag": pd.DataFrame(),
        "calendar": pd.DataFrame(),
        "peaks": pd.DataFrame(),
        "rules": pd.DataFrame(),
        "months": pd.DataFrame(),
    }


def vesting_analysis(
    prices: pd.DataFrame,
    windows: pd.DataFrame,
    n_boot: int = 5_000,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Run the vest-date analysis.

    Returns a dict of DataFrames:

    - ``windows``: one row per completed window with vest date, peak date and
      hold-to-close return.
    - ``lag``: average return by number of trading days after vest.
    - ``calendar``: average return by calendar day of the month after vest.
    - ``peaks``: where in the month the post-vest high occurred.
    - ``rules``: summary of fixed sale rules.
    - ``months``: sale-rule results split by Feb/May/Aug/Nov.
    """
    if prices.empty or windows.empty:
        return _empty_result()

    stats = window_stats(prices, windows)
    completed = stats.loc[~stats["is_partial"]].copy()
    if completed.empty:
        return _empty_result()

    trading = prices.index
    rng = np.random.default_rng(seed)

    window_rows: list[dict] = []
    calendar_rows: list[dict] = []
    rule_rows: list[dict] = []

    for window in completed.itertuples(index=False):
        vest = _vest_date(pd.Timestamp(window.close_date), trading)
        if vest is None:
            continue

        post = trading[(trading >= vest) & (trading <= window.close_date)]
        window_days = trading[
            (trading >= window.open_date) & (trading <= window.close_date)
        ]
        if len(post) == 0 or len(window_days) == 0:
            continue

        entry_price = float(prices.loc[window.open_date, "open"])
        vest_close = float(prices.loc[vest, "close"])
        close_price = float(prices.loc[window.close_date, "close"])
        closes = prices.loc[window_days, "close"].astype(float)
        percentiles = closes.rank(pct=True) * 100.0
        post_prices = prices.loc[post, "close"].astype(float)
        peak_date = pd.Timestamp(post_prices.idxmax())
        peak_offset = int(post.get_loc(peak_date))

        window_rows.append(
            {
                "window_id": int(window.window_id),
                "label": window.label,
                "year": int(window.year),
                "month": window.month_name,
                "open_date": pd.Timestamp(window.open_date),
                "vest_date": vest,
                "close_date": pd.Timestamp(window.close_date),
                "entry_price": entry_price,
                "vest_close": vest_close,
                "close_price": close_price,
                "entry_to_vest": vest_close / entry_price - 1.0,
                "hold_to_close": close_price / vest_close - 1.0,
                "peak_return": float(post_prices.max()) / vest_close - 1.0,
                "peak_date": peak_date,
                "peak_dom": int(peak_date.day),
                "peak_offset": peak_offset,
                "n_post": int(len(post)),
                "peak_last5": peak_offset >= len(post) - 5,
            }
        )

        for date in post:
            calendar_rows.append(
                {
                    "label": window.label,
                    "month": window.month_name,
                    "dom": int(date.day),
                    "rel_vest": float(prices.loc[date, "close"]) / vest_close - 1.0,
                    "percentile": float(percentiles.loc[date]),
                }
            )

        rule_dates: dict[str, pd.Timestamp] = {"vest": vest, "last": pd.Timestamp(window.close_date)}
        for lag, rule in ((3, "plus_3"), (5, "plus_5"), (7, "plus_7"), (10, "plus_10")):
            if lag < len(post):
                rule_dates[rule] = pd.Timestamp(post[lag])
        for day in (20, 25, 26):
            later = post[post.day >= day]
            if len(later):
                rule_dates[f"dom_{day}"] = pd.Timestamp(later[0])

        for rule, date in rule_dates.items():
            rule_rows.append(
                {
                    "label": window.label,
                    "month": window.month_name,
                    "rule": rule,
                    "date": date,
                    "ret": float(prices.loc[date, "close"]) / vest_close - 1.0,
                }
            )

    if not window_rows:
        return _empty_result()

    window_table = pd.DataFrame(window_rows).sort_values("open_date").reset_index(drop=True)
    calendar = pd.DataFrame(calendar_rows)
    rule_returns = pd.DataFrame(rule_rows)

    # --- lag profile -------------------------------------------------------
    lag_rows = []
    for lag in range(0, 13):
        values = []
        for window in window_table.itertuples(index=False):
            post = trading[
                (trading >= window.vest_date) & (trading <= window.close_date)
            ]
            if lag < len(post):
                values.append(
                    float(prices.loc[post[lag], "close"]) / window.vest_close - 1.0
                )
        if not values:
            continue
        values_array = np.asarray(values)
        ci_low, ci_high = _bootstrap_ci(values_array, n_boot, rng)
        lag_rows.append(
            {
                "lag": lag,
                "n": len(values_array),
                "mean": float(values_array.mean()),
                "median": float(np.median(values_array)),
                "win_rate": float((values_array > 0).mean()),
                "std": float(values_array.std(ddof=1)) if len(values_array) > 1 else np.nan,
                "p10": float(np.percentile(values_array, 10)),
                "p90": float(np.percentile(values_array, 90)),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    lag = pd.DataFrame(lag_rows)

    # --- calendar profile --------------------------------------------------
    calendar_agg = (
        calendar.groupby("dom")
        .agg(
            n=("rel_vest", "size"),
            mean=("rel_vest", "mean"),
            median=("rel_vest", "median"),
            win_rate=("rel_vest", lambda values: float((values > 0).mean())),
            mean_percentile=("percentile", "mean"),
        )
        .reindex(range(15, 32))
    )

    # --- peak timing -------------------------------------------------------
    peak_counts = (
        window_table["peak_dom"]
        .value_counts()
        .reindex(range(15, 32), fill_value=0)
        .rename("count")
        .to_frame()
    )
    peak_counts.index.name = "dom"
    peak_counts["share"] = peak_counts["count"] / len(window_table)

    # --- fixed sale rules --------------------------------------------------
    pivot = rule_returns.pivot_table(
        index="label", columns="rule", values="ret", aggfunc="first"
    )
    baseline = pivot["vest"]
    rule_summary_rows = []
    for rule in RULE_ORDER:
        if rule not in pivot.columns:
            continue
        values = pivot[rule].dropna()
        if values.empty:
            continue
        diffs = (pivot[rule] - baseline).dropna()
        ci_low, ci_high = _bootstrap_ci(values.to_numpy(), n_boot, rng)
        diff_low, diff_high = _bootstrap_ci(diffs.to_numpy(), n_boot, rng)
        rule_summary_rows.append(
            {
                "rule": rule,
                "label": RULE_LABELS[rule],
                "n": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "win_rate": float((values > 0).mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "diff_mean": float(diffs.mean()) if len(diffs) else np.nan,
                "diff_ci_low": diff_low,
                "diff_ci_high": diff_high,
                "beat_vest": float((diffs > 0).mean()) if len(diffs) else np.nan,
            }
        )
    rules = pd.DataFrame(rule_summary_rows)

    # --- by month ----------------------------------------------------------
    month_frames = []
    for month in ("Feb", "May", "Aug", "Nov"):
        subset = rule_returns.loc[rule_returns["month"] == month]
        row = {"month": month, "windows": int(subset["label"].nunique())}
        for rule in ("vest", "plus_5", "dom_26", "last"):
            values = subset.loc[subset["rule"] == rule, "ret"]
            if values.empty:
                row[f"{rule}_mean"] = np.nan
                row[f"{rule}_win"] = np.nan
            else:
                row[f"{rule}_mean"] = float(values.mean())
                row[f"{rule}_win"] = float((values > 0).mean())
        month_frames.append(row)
    months = pd.DataFrame(month_frames)

    return {
        "windows": window_table,
        "lag": lag,
        "calendar": calendar_agg.reset_index(),
        "peaks": peak_counts.reset_index(),
        "rules": rules,
        "months": months,
    }

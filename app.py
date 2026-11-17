"""Streamlit dashboard: Meta stock action inside employee trading windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from meta_windows.analytics import (
    day_of_window_profile,
    equity_curves,
    equity_summary,
    heatmap_returns,
    regime_stats,
    window_stats,
    window_trajectories,
)
from meta_windows.charts import (
    day_profile_chart,
    equity_chart,
    price_window_chart,
    return_distribution_chart,
    trajectory_chart,
    volume_ratio_chart,
    window_heatmap_chart,
    window_returns_chart,
)
from meta_windows.data import fetch_earnings_dates, fetch_prices
from meta_windows.windows import assign_window_days, build_windows

st.set_page_config(
    page_title="Meta employee trading windows",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    :root {
        --surface: #fcfcfb;
        --page: #f9f9f7;
        --ink: #0b0b0b;
        --secondary: #52514e;
        --muted: #898781;
        --hairline: rgba(11, 11, 11, 0.10);
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--page);
        color: var(--ink);
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    .block-container {
        padding-top: 3.9rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }
    h1, h2, h3 {
        letter-spacing: -0.02em;
        color: var(--ink);
    }
    .eyebrow {
        font-size: 0.76rem;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .lede {
        color: var(--secondary);
        font-size: 1.02rem;
        max-width: 72ch;
        margin-bottom: 0.4rem;
    }
    .kpi-card {
        background: var(--surface);
        border: 1px solid var(--hairline);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        min-height: 7.1rem;
    }
    .kpi-label {
        color: var(--secondary);
        font-size: 0.82rem;
        line-height: 1.25;
    }
    .kpi-value {
        color: var(--ink);
        font-size: 1.75rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin: 0.3rem 0 0.2rem;
    }
    .kpi-note {
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.3;
    }
    .kpi-note.neg { color: #d03b3b; }
    .kpi-note.pos { color: #006300; }
    [data-testid="stSidebar"] {
        background: #f3f2ee;
        border-right: 1px solid var(--hairline);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.2rem;
        border-bottom: 1px solid var(--hairline);
    }
    .stTabs [data-baseweb="tab"] {
        padding-bottom: 0.6rem;
        color: var(--secondary);
    }
    .stTabs [aria-selected="true"] {
        color: var(--ink);
        font-weight: 600;
    }
    .caption-note {
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: -0.3rem;
        margin-bottom: 0.8rem;
    }
    footer { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner="Loading Meta prices and SEC earnings dates…")
def load_data(refresh_token: int = 0):
    refresh = refresh_token > 0
    earnings = fetch_earnings_dates(refresh=refresh)
    prices = fetch_prices(refresh=refresh)
    windows = build_windows(earnings["earnings_date"], prices.index)
    return earnings, prices, windows


def pct(value: float, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def money(value: float) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"${float(value):,.2f}"


def _series_value(summary: pd.DataFrame, series: str, metric: str) -> float:
    row = summary.loc[summary["series"] == series]
    if row.empty:
        return np.nan
    return float(row.iloc[0][metric])


def kpi_card(label: str, value: str, note: str = "", note_class: str = "") -> None:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note {note_class}">{note}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

st.session_state.setdefault("refresh_token", 0)

with st.sidebar:
    st.markdown("### Meta trading windows")
    st.caption("Employee open-trading windows, from the 2012 IPO to today.")
    if st.button("Refresh data", width="stretch"):
        st.session_state["refresh_token"] += 1
        st.cache_data.clear()

try:
    earnings, prices, all_windows = load_data(st.session_state["refresh_token"])
except Exception as exc:  # noqa: BLE001 - surface data failures in the UI
    st.error(
        "Could not load price or filing data. Check the network connection and "
        f"try Refresh data. Details: {exc}"
    )
    st.stop()

if all_windows.empty:
    st.error("No trading windows could be derived from the loaded data.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("#### Filters")
    year_min, year_max = int(all_windows["year"].min()), int(all_windows["year"].max())
    selected_years = st.slider(
        "Window year (month-end)",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1,
    )
    month_options = ["Feb", "May", "Aug", "Nov"]
    selected_months = st.multiselect(
        "Window month",
        options=month_options,
        default=month_options,
        help="Meta's open window runs through the end of Feb, May, Aug or Nov.",
    )
    log_scale = st.toggle("Log price scale", value=False)
    st.caption(
        "Filters scope every chart. Days outside the selected windows are "
        "counted as blackout."
    )
    st.markdown("---")
    st.caption("Prices: Yahoo Finance · Earnings dates: SEC EDGAR 8-K Item 2.02")

selected = all_windows.loc[
    all_windows["year"].between(*selected_years)
    & all_windows["month_name"].isin(selected_months)
].copy()

if selected.empty:
    st.warning("No windows match those filters. Widen the year or month selection.")
    st.stop()

selected = selected.sort_values("open_date").reset_index(drop=True)
selected["window_id"] = np.arange(1, len(selected) + 1)

last_price_date = prices.index.max()
span_start = pd.Timestamp(selected["open_date"].min())
span_end = min(pd.Timestamp(selected["close_date"].max()), last_price_date)
price_span = prices.loc[
    (prices.index >= span_start) & (prices.index <= span_end)
].copy()

if price_span.empty:
    st.warning("No price data covers the selected windows.")
    st.stop()

annotated = assign_window_days(price_span, selected)
stats = window_stats(prices, selected)
regimes = regime_stats(price_span, selected)
curves = equity_curves(prices, selected)
summary = equity_summary(curves)
trajectories = window_trajectories(prices, selected)
profile = day_of_window_profile(prices, selected)
heatmap = heatmap_returns(stats)

completed = stats.loc[~stats["is_partial"]] if not stats.empty else stats
n_open_now = int(stats["is_partial"].sum()) if not stats.empty else 0
n_completed = int(len(completed))

open_cagr = _series_value(summary, "Open-window only", "cagr")
buy_hold_cagr = _series_value(summary, "Buy & hold", "cagr")
open_dd = _series_value(summary, "Open-window only", "max_drawdown")
buy_hold_dd = _series_value(summary, "Buy & hold", "max_drawdown")
share_open_days = (
    float(regimes.loc[regimes["regime"] == "Open window", "share_of_days"].iloc[0])
    if not regimes.empty
    else np.nan
)

# ---------------------------------------------------------------------------
# Header and KPIs
# ---------------------------------------------------------------------------

st.markdown(
    f'<div class="eyebrow">{prices.index.min():%b %Y} — {prices.index.max():%b %Y} '
    f'· {len(stats)} quarterly windows</div>',
    unsafe_allow_html=True,
)
st.title("Meta's employee trading windows")
st.markdown(
    '<p class="lede">Meta employees can only trade company stock during quarterly '
    "open windows: from the second full trading day after earnings until the end of "
    "Feb, May, Aug or Nov. This dashboard measures how META behaves inside those "
    "windows versus the blackout periods between them.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="caption-note">{pct(share_open_days)} of trading days fall inside '
    "an open window. Window returns are entry at the first open, exit at the "
    "last close.</p>",
    unsafe_allow_html=True,
)

cagr_gap = open_cagr - buy_hold_cagr
cagr_note = (
    f"{cagr_gap * 100:+.1f} pp vs buy & hold"
    if not (pd.isna(open_cagr) or pd.isna(buy_hold_cagr))
    else "selected span"
)
cagr_class = "neg" if (not pd.isna(cagr_gap) and cagr_gap < 0) else "pos"
open_note = f"{n_completed} completed" + (
    f" · {n_open_now} open now" if n_open_now else ""
)

k1, k2, k3, k4, k5 = st.columns(5, gap="medium")
with k1:
    kpi_card("Windows analysed", f"{len(stats)}", open_note)
with k2:
    kpi_card(
        "Avg window return",
        pct(completed["return_pct"].mean()) if n_completed else "—",
        "completed windows",
    )
with k3:
    kpi_card(
        "Window win rate",
        pct((completed["return_pct"] > 0).mean()) if n_completed else "—",
        "positive windows",
    )
with k4:
    kpi_card("Window-only CAGR", pct(open_cagr), cagr_note, cagr_class)
with k5:
    kpi_card("Buy & hold CAGR", pct(buy_hold_cagr), "same calendar span")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_price, tab_score, tab_regime, tab_inside, tab_method = st.tabs(
    ["Price & windows", "Window scoreboard", "Open vs blackout", "Inside the window", "Methodology"]
)

with tab_price:
    st.subheader("Price action, with every open window shaded")
    st.markdown(
        '<p class="caption-note">Shaded bands are employee open-trading windows. '
        "The lower panel splits daily volume into open-window days (blue) and "
        "blackout days (gray).</p>",
        unsafe_allow_html=True,
    )
    fig = price_window_chart(price_span, annotated, selected, log_scale=log_scale)
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

    with st.expander("Window calendar and data"):
        calendar = selected.assign(
            Earnings=selected["earnings_date"].dt.strftime("%d %b %Y"),
            Opens=selected["open_date"].dt.strftime("%d %b %Y"),
            Closes=selected["close_date"].dt.strftime("%d %b %Y"),
        )[["label", "Earnings", "Opens", "Closes", "n_days"]]
        calendar.columns = ["Window", "Earnings release", "Window opens", "Window closes", "Trading days"]
        st.dataframe(calendar, hide_index=True, width="stretch")
        st.download_button(
            "Download window calendar (CSV)",
            data=selected.to_csv(index=False),
            file_name="meta_trading_windows.csv",
            mime="text/csv",
        )

with tab_score:
    st.subheader("Every window's return")
    st.markdown(
        '<p class="caption-note">Blue bars are gains, red bars are losses. Faded '
        "bars are still open. Hover for dates and length.</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        window_returns_chart(stats),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

    st.markdown("#### When do windows pay?")
    heat_col, month_col = st.columns([1.6, 1], gap="large")
    with heat_col:
        st.plotly_chart(
            window_heatmap_chart(heatmap),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
    with month_col:
        st.markdown("**Return by window month**")
        month_summary = (
            stats.groupby("month_name")
            .agg(
                Windows=("return_pct", "size"),
                Avg_return=("return_pct", "mean"),
                Win_rate=("return_pct", lambda values: float((values > 0).mean())),
            )
            .reindex(["Feb", "May", "Aug", "Nov"])
            .dropna(how="all")
            .reset_index()
            .rename(
                columns={
                    "month_name": "Month",
                    "Avg_return": "Avg return",
                    "Win_rate": "Win rate",
                }
            )
        )
        month_summary["Avg return"] *= 100
        month_summary["Win rate"] *= 100
        st.dataframe(
            month_summary,
            hide_index=True,
            width="stretch",
            column_config={
                "Avg return": st.column_config.NumberColumn(format="%+.1f%%"),
                "Win rate": st.column_config.NumberColumn(format="%.0f%%"),
            },
        )
        st.caption(
            "One window per quarter, so month-level averages rest on small "
            "samples. The full table below has every window."
        )

    st.markdown("#### Window detail")
    display = stats.copy()
    display["Status"] = np.where(display["is_partial"], "Open now", "Closed")
    display = display.assign(
        Window=display["label"],
        Earnings=display["earnings_date"].dt.strftime("%d %b %Y"),
        Opens=display["open_date"].dt.strftime("%d %b %Y"),
        Closes=display["close_date"].dt.strftime("%d %b %Y"),
        Return=display["return_pct"] * 100,
        Drawdown=display["max_drawdown_pct"] * 100,
        Volatility=display["volatility_pct"] * 100,
        Range=display["range_pct"] * 100,
        Up_days=display["up_day_share"] * 100,
        Volume_ratio=display["volume_ratio"],
    )
    table = display[
        [
            "Window",
            "Status",
            "Earnings",
            "Opens",
            "Closes",
            "n_days",
            "Return",
            "Drawdown",
            "Volatility",
            "Range",
            "Up_days",
            "Volume_ratio",
        ]
    ].rename(
        columns={
            "n_days": "Days",
            "Return": "Return",
            "Drawdown": "Max drawdown",
            "Volatility": "Ann. volatility",
            "Range": "High-low range",
            "Up_days": "Up days",
            "Volume_ratio": "Vol vs prior 60d",
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Return": st.column_config.NumberColumn(format="%+.1f%%"),
            "Max drawdown": st.column_config.NumberColumn(format="%.1f%%"),
            "Ann. volatility": st.column_config.NumberColumn(format="%.1f%%"),
            "High-low range": st.column_config.NumberColumn(format="%.1f%%"),
            "Up days": st.column_config.NumberColumn(format="%.0f%%"),
            "Vol vs prior 60d": st.column_config.NumberColumn(format="%.2f×"),
        },
    )
    st.download_button(
        "Download window statistics (CSV)",
        data=stats.to_csv(index=False),
        file_name="meta_window_stats.csv",
        mime="text/csv",
    )

with tab_regime:
    st.subheader("Open windows vs blackout periods")
    st.markdown(
        '<p class="caption-note">The blue line compounds each window\'s tradeable '
        "return and sits in cash between windows. The orange line holds META "
        "throughout. Both are indexed to 100 at the first selected window's open. "
        "Selected-window shading marks days the strategy is invested.</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        equity_chart(curves, selected),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

    eq1, eq2 = st.columns([1, 1], gap="large")
    with eq1:
        st.markdown("**Strategy comparison**")
        eq_display = summary.assign(
            CAGR=summary["cagr"] * 100,
            Drawdown=summary["max_drawdown"] * 100,
            Final=summary["final_value"],
        )[["series", "CAGR", "Drawdown", "Final"]].rename(
            columns={
                "series": "Strategy",
                "CAGR": "CAGR",
                "Drawdown": "Max drawdown",
                "Final": "Final index",
            }
        )
        st.dataframe(
            eq_display,
            hide_index=True,
            width="stretch",
            column_config={
                "CAGR": st.column_config.NumberColumn(format="%.1f%%"),
                "Max drawdown": st.column_config.NumberColumn(format="%.1f%%"),
                "Final index": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        st.caption(
            f"Open-window max drawdown {pct(open_dd)} · buy & hold {pct(buy_hold_dd)}."
        )
    with eq2:
        st.markdown("**Daily return distribution**")
        st.plotly_chart(
            return_distribution_chart(prices, annotated),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )
        st.caption("Histogram clipped to ±10% daily moves; axis is share of days.")

    st.markdown("#### Daily behaviour by regime")
    regime_display = regimes.assign(
        Share=regimes["share_of_days"] * 100,
        Total=regimes["total_return"] * 100,
        Annual_return=regimes["annualized_return"] * 100,
        Annual_vol=regimes["annualized_vol"] * 100,
        Hit=regimes["hit_rate"] * 100,
        Avg_day=regimes["avg_daily_return"] * 100,
        Best=regimes["best_day"] * 100,
        Worst=regimes["worst_day"] * 100,
    )[
        [
            "regime",
            "n_days",
            "Share",
            "Total",
            "Annual_return",
            "Annual_vol",
            "sharpe",
            "Hit",
            "Avg_day",
            "Best",
            "Worst",
        ]
    ].rename(
        columns={
            "regime": "Regime",
            "n_days": "Trading days",
            "Share": "Share of days",
            "Total": "Total return",
            "Annual_return": "Annualised return",
            "Annual_vol": "Annualised volatility",
            "sharpe": "Sharpe",
            "Hit": "Hit rate",
            "Avg_day": "Average day",
            "Best": "Best day",
            "Worst": "Worst day",
        }
    )
    st.dataframe(
        regime_display,
        hide_index=True,
        width="stretch",
        column_config={
            "Share of days": st.column_config.NumberColumn(format="%.1f%%"),
            "Total return": st.column_config.NumberColumn(format="%+.1f%%"),
            "Annualised return": st.column_config.NumberColumn(format="%+.1f%%"),
            "Annualised volatility": st.column_config.NumberColumn(format="%.1f%%"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Hit rate": st.column_config.NumberColumn(format="%.1f%%"),
            "Average day": st.column_config.NumberColumn(format="%+.2f%%"),
            "Best day": st.column_config.NumberColumn(format="%+.1f%%"),
            "Worst day": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )
    st.caption(
        "Daily returns are close-to-close. Days outside the selected windows are "
        "treated as blackout, including any windows removed by the month filter."
    )

with tab_inside:
    st.subheader("What a window looks like from the inside")
    min_windows = max(5, round(len(selected) * 0.25))
    st.markdown(
        '<p class="caption-note">Each gray line is one window, indexed to 100 at '
        "the entry open. Blue is the median path; orange is the latest window. "
        f"Median and day-profile charts only include days observed by at least "
        f"{min_windows} windows.</p>",
        unsafe_allow_html=True,
    )
    latest_id = int(selected["window_id"].iloc[-1]) if not selected.empty else None
    st.plotly_chart(
        trajectory_chart(trajectories, latest_id, min_windows=min_windows),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

    st.markdown("**Average return by day in the window**")
    st.plotly_chart(
        day_profile_chart(profile, min_windows=min_windows),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )
    st.caption(
        "Blue bars are positive average returns, red negative. Only days "
        f"with at least {min_windows} windows are shown."
    )

    st.markdown("**Volume inside windows vs the prior 60 days**")
    st.plotly_chart(
        volume_ratio_chart(stats),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )
    st.caption(
        "Ratio of average daily volume during each window to the 60 trading "
        "days before it. Baseline is 1.0×."
    )

with tab_method:
    st.subheader("Methodology and caveats")
    st.markdown(
        """
### How a window is defined

Meta's 2024 Form 10-K describes a **closed trading period** that begins at
market close on the last day of the **middle month of each quarter**
(February, May, August, November). The window reopens on the **second full
trading day after the quarterly earnings release** and stays open until the
next middle-month close.

For each earnings release, the dashboard therefore:

1. Takes the earnings release date from the company's **SEC Form 8-K with Item 2.02**.
2. Finds the second trading day strictly after that date — the window open.
3. Runs the window through the last trading day on or before the next
   middle-month end. If that end is in the future, the policy date is kept and
   the window is marked **open now / partial**.

Source: [Meta 2024 Form 10-K](https://d18rn0p25nwr6d.cloudfront.net/CIK-0001326801/5e0f2f01-cd4e-4252-afe5-4052117d5c93.html#28),
[Meta Code of Conduct](https://www.meta.com/sv-se/file/1979986042551159/2025-Code-of-Conduct---02.02.26---zhCN.pdf/#4#4#4).

### Data

- **Prices:** Yahoo Finance daily chart API, META (the ticker was FB before
  October 2021), adjusted for dividends. Full history starts at the
  **18 May 2012 IPO**.
- **Earnings dates:** [SEC EDGAR submissions API](https://data.sec.gov/submissions/CIK0001326801.json),
  filtering Form 8-K filings whose items include `2.02` (results of operations).
  The 8-K filing date is used as the release date; Meta files the earnings
  release on the announcement day.
- Both sources are cached in `.cache/` for seven days (earnings) and twelve
  hours (prices). Use **Refresh data** in the sidebar to bypass the cache.

### Metrics

- **Window return:** first window day's open → last window day's close. This is
  the tradeable return available to an employee who can only transact in the
  window.
- **Volatility:** standard deviation of daily close-to-close returns inside the
  window, annualised by √252.
- **Max drawdown:** largest peak-to-trough decline of daily closes, including
  the entry price.
- **Window-only strategy:** compounds each selected window's return and earns
  nothing between windows. **Buy & hold** holds META over the same calendar span.
- **Blackout regime:** all trading days in the selected span that are not inside
  a selected open window.

### Caveats

- The window is a **policy calendar, not a filing**. Meta can close the window
  early for material non-public information, and the People Portal calendar is
  authoritative for employees.
- This measures **price behaviour during windows**, not what employees actually
  traded. Section 16 insiders may trade through pre-cleared 10b5-1 plans outside
  the window, and Form 4 filings are not a census of employee trades.
- The 8-K date is a close proxy for the release date, but a release after the
  close plus a next-day filing can shift the calculated open by one day.
- Adjusted prices include dividend reinvestment; actual employee trade prices
  are unadjusted. Meta's dividend is small relative to daily moves.
- The latest window is partial and its return is marked accordingly.
- Nothing here is investment advice.
        """
    )
    st.caption(
        f"Data loaded {pd.Timestamp.now():%d %b %Y %H:%M} · "
        f"{len(prices):,} trading days · {len(all_windows)} total windows in history."
    )

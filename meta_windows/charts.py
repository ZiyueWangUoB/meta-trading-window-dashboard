"""Plotly chart builders using one shared, accessible light theme.

Palette: categorical slots 1-2 from the validated reference palette
(#2a78d6 blue, #eb6834 orange; adjacent CVD ΔE 24.7). Positive/negative
returns use the documented blue <-> red diverging pair with a neutral gray
zero baseline. One y-axis per plot; price and volume are stacked panels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
NEUTRAL = "#f0efec"
CONTEXT_GRAY = "#c3c2b7"
WINDOW_FILL = "rgba(42, 120, 214, 0.10)"

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _base_layout(
    fig: go.Figure,
    height: int = 420,
    hovermode: str = "x unified",
    showlegend: bool = True,
    right_margin: int = 24,
) -> go.Figure:
    fig.update_layout(
        template="none",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_FAMILY, color=INK, size=13),
        height=height,
        margin=dict(l=64, r=right_margin, t=56, b=40),
        hovermode=hovermode,
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=AXIS,
            font=dict(color=INK, size=12),
        ),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=SECONDARY, size=12),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=AXIS,
        linewidth=1,
        tickfont=dict(color=MUTED, size=11),
        zeroline=False,
        title_font=dict(color=SECONDARY, size=12),
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        linecolor=AXIS,
        linewidth=1,
        tickfont=dict(color=MUTED, size=11),
        zeroline=False,
        title_font=dict(color=SECONDARY, size=12),
        automargin=True,
    )
    return fig


def _add_window_bands(fig: go.Figure, windows: pd.DataFrame, **kwargs) -> None:
    for window in windows.itertuples(index=False):
        fig.add_vrect(
            x0=window.open_date,
            x1=window.close_date,
            fillcolor=WINDOW_FILL,
            line_width=0,
            layer="below",
            **kwargs,
        )


def price_window_chart(
    prices: pd.DataFrame,
    annotated: pd.DataFrame,
    windows: pd.DataFrame,
    log_scale: bool = False,
) -> go.Figure:
    """Price with shaded open windows and a separate daily-volume panel."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.74, 0.26],
    )

    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices["close"],
            mode="lines",
            name="META price",
            line=dict(color=BLUE, width=2),
            hovertemplate="$%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    blackout = annotated.loc[~annotated["in_window"]]
    open_days = annotated.loc[annotated["in_window"]]
    fig.add_trace(
        go.Bar(
            x=blackout.index,
            y=blackout["volume"],
            name="Blackout days",
            marker=dict(color=CONTEXT_GRAY, line_width=0),
            hovertemplate="%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=open_days.index,
            y=open_days["volume"],
            name="Open-window days",
            marker=dict(color=BLUE, line_width=0),
            hovertemplate="%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    _add_window_bands(fig, windows, row=1, col=1)
    fig.update_layout(barmode="overlay", bargap=0.15)

    last = prices.iloc[-1]
    fig.add_annotation(
        x=prices.index[-1],
        y=float(last["close"]),
        text=f"${float(last['close']):,.0f}",
        xanchor="left",
        xshift=6,
        yanchor="middle",
        showarrow=False,
        font=dict(color=INK, size=11),
        row=1,
        col=1,
    )

    if log_scale:
        fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(tickprefix="$", row=1, col=1)
    fig.update_yaxes(showgrid=False, rangemode="tozero", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(showgrid=False, tickformat="%Y", row=1, col=1)
    fig.update_xaxes(showgrid=False, tickformat="%Y", row=2, col=1)

    return _base_layout(fig, height=580, right_margin=64)


def window_returns_chart(stats: pd.DataFrame) -> go.Figure:
    """Per-window return bars, blue above zero and red below.

    The x-axis is categorical (one slot per quarter) rather than a date axis,
    so bars stay readable across a 14-year history.
    """
    data = stats.sort_values("open_date").reset_index(drop=True)
    colors = np.where(data["return_pct"] >= 0, BLUE, RED)
    opacities = np.where(data["is_partial"], 0.55, 1.0)
    customdata = np.column_stack(
        [
            data["label"],
            data["open_date"].dt.strftime("%d %b %Y"),
            data["close_date"].dt.strftime("%d %b %Y"),
            data["n_days"],
        ]
    )

    fig = go.Figure(
        go.Bar(
            x=data["label"],
            y=data["return_pct"] * 100,
            marker=dict(
                color=colors,
                opacity=opacities,
                line=dict(color=SURFACE, width=1),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Window %{customdata[1]} → %{customdata[2]} "
                "(%{customdata[3]} trading days)<br>"
                "Return %{y:+.1f}%<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_color=AXIS, line_width=1)

    tickvals: list[str] = []
    ticktext: list[str] = []
    previous_year = None
    for label, year in zip(data["label"], data["year"]):
        if year != previous_year:
            tickvals.append(str(label))
            ticktext.append(str(year))
            previous_year = year

    for row in (
        data.loc[data["return_pct"].idxmax()],
        data.loc[data["return_pct"].idxmin()],
    ):
        positive = row["return_pct"] >= 0
        fig.add_annotation(
            x=row["label"],
            y=row["return_pct"] * 100,
            text=f"{row['return_pct'] * 100:+.1f}%",
            showarrow=False,
            yshift=16 if positive else -16,
            font=dict(color=INK, size=11),
        )

    fig.update_yaxes(ticksuffix="%", zeroline=False)
    fig.update_xaxes(
        type="category",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
    )
    return _base_layout(fig, height=400, hovermode="closest", showlegend=False)


def window_heatmap_chart(pivot: pd.DataFrame) -> go.Figure:
    """Year x middle-month matrix of window returns (diverging blue/red)."""
    if pivot.empty:
        return _base_layout(go.Figure(), height=280, showlegend=False)

    z = pivot.to_numpy(dtype=float) * 100
    finite = z[np.isfinite(z)]
    limit = max(abs(finite).max(), 1.0) if finite.size else 10.0

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(pivot.columns),
            y=[str(year) for year in pivot.index],
            colorscale=[[0.0, RED], [0.5, NEUTRAL], [1.0, BLUE]],
            zmid=0,
            zmin=-limit,
            zmax=limit,
            xgap=2,
            ygap=2,
            hovertemplate="<b>%{y} · %{x}</b><br>Return %{z:+.1f}%<extra></extra>",
            colorbar=dict(
                title=dict(text="Return", side="top", font=dict(color=SECONDARY, size=11)),
                thickness=10,
                len=0.75,
                outlinewidth=0,
                tickfont=dict(color=MUTED, size=10),
                ticksuffix="%",
            ),
        )
    )
    year_labels = [str(year) for year in pivot.index]
    for row_index, year_label in enumerate(year_labels):
        for column_index, month_label in enumerate(pivot.columns):
            value = z[row_index, column_index]
            if not np.isfinite(value):
                continue
            fig.add_annotation(
                x=month_label,
                y=year_label,
                text=f"{value:+.1f}%",
                showarrow=False,
                font=dict(
                    size=10,
                    color="#ffffff" if abs(value) > limit * 0.6 else INK,
                ),
            )

    fig.update_xaxes(side="top", showgrid=False)
    fig.update_yaxes(
        showgrid=False,
        tickmode="array",
        tickvals=year_labels,
        ticktext=year_labels,
    )
    height = max(300, 30 * len(pivot.index) + 110)
    return _base_layout(fig, height=height, hovermode="closest", showlegend=False)


def equity_chart(curves: pd.DataFrame, windows: pd.DataFrame) -> go.Figure:
    """Open-window-only vs buy-and-hold, both indexed to 100."""
    fig = go.Figure()
    _add_window_bands(fig, windows)

    series_colors = {"Open-window only": BLUE, "Buy & hold": ORANGE}
    for series, color in series_colors.items():
        data = curves.loc[curves["series"] == series].sort_values("date")
        if data.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["value"],
                mode="lines",
                name=series,
                line=dict(color=color, width=2),
                hovertemplate="%{y:.1f}<extra></extra>",
            )
        )

    fig.add_hline(y=100, line_color=AXIS, line_width=1)

    for series, color in series_colors.items():
        data = curves.loc[curves["series"] == series].sort_values("date")
        if data.empty:
            continue
        last = data.iloc[-1]
        fig.add_annotation(
            x=last["date"],
            y=float(last["value"]),
            text=f"{series} · {float(last['value']):,.0f}",
            xanchor="left",
            xshift=6,
            yanchor="middle",
            showarrow=False,
            font=dict(color=INK, size=11),
        )

    start = curves["date"].min()
    end = curves["date"].max()
    fig.update_xaxes(range=[start - pd.Timedelta(days=10), end + pd.Timedelta(days=40)])
    fig.update_yaxes(title_text="Index (first window open = 100)", rangemode="tozero")
    return _base_layout(fig, height=470, right_margin=170)


def return_distribution_chart(prices: pd.DataFrame, annotated: pd.DataFrame) -> go.Figure:
    """Daily close-to-close returns inside vs outside open windows."""
    returns = prices["close"].pct_change()
    frame = pd.DataFrame(
        {"ret": returns, "in_window": annotated["in_window"]}
    ).dropna()

    fig = go.Figure()
    for flag, label, color in (
        (True, "Open window", BLUE),
        (False, "Blackout", ORANGE),
    ):
        values = frame.loc[frame["in_window"] == flag, "ret"] * 100
        fig.add_trace(
            go.Histogram(
                x=values,
                name=label,
                histnorm="percent",
                xbins=dict(start=-10, end=10, size=0.5),
                marker=dict(
                    color=color,
                    line=dict(color=color, width=1),
                ),
                opacity=0.35,
                hovertemplate="%{y:.1f}% of days<extra>" + label + "</extra>",
            )
        )

    fig.add_vline(x=0, line_color=AXIS, line_width=1)
    fig.update_layout(barmode="overlay", bargap=0.03)
    fig.update_xaxes(range=[-10, 10], ticksuffix="%")
    fig.update_yaxes(ticksuffix="%", title_text="Share of days")
    return _base_layout(fig, height=400, hovermode="closest")


def trajectory_chart(
    trajectories: pd.DataFrame,
    latest_window_id: int | None,
    min_windows: int = 10,
) -> go.Figure:
    """Every window's path, with median and latest window emphasised.

    The median line only covers days observed by at least ``min_windows``
    windows, so the tail is not driven by one or two long windows.
    """
    fig = go.Figure()
    first = True
    for _, group in trajectories.groupby("window_id", sort=True):
        label = str(group["label"].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=group["day"],
                y=group["value"],
                mode="lines",
                name="Individual windows",
                legendgroup="individual",
                showlegend=first,
                line=dict(color=CONTEXT_GRAY, width=1),
                opacity=0.65,
                hovertemplate=f"<b>{label}</b><br>Day %{{x}}: %{{y:.1f}}<extra></extra>",
            )
        )
        first = False

    median_by_day = trajectories.groupby("day")["value"].agg(["median", "size"])
    median = median_by_day.loc[median_by_day["size"] >= min_windows, "median"]
    if not median.empty:
        fig.add_trace(
            go.Scatter(
                x=median.index,
                y=median.to_numpy(),
                mode="lines",
                name="Median window",
                line=dict(color=BLUE, width=2.5),
                hovertemplate="Median · day %{x}: %{y:.1f}<extra></extra>",
            )
        )

    if latest_window_id is not None:
        latest = trajectories.loc[trajectories["window_id"] == latest_window_id]
        if not latest.empty:
            latest_label = str(latest["label"].iloc[0])
            fig.add_trace(
                go.Scatter(
                    x=latest["day"],
                    y=latest["value"],
                    mode="lines",
                    name=f"Latest window ({latest_label})",
                    line=dict(color=ORANGE, width=2.5),
                    hovertemplate=(
                        f"<b>Latest · {latest_label}</b><br>"
                        "Day %{x}: %{y:.1f}<extra></extra>"
                    ),
                )
            )
            fig.add_annotation(
                x=latest["day"].iloc[-1],
                y=float(latest["value"].iloc[-1]),
                text=f"Latest · {float(latest['value'].iloc[-1]):.0f}",
                xanchor="left",
                xshift=6,
                yanchor="middle",
                showarrow=False,
                font=dict(color=INK, size=11),
            )

    if not median.empty:
        fig.add_annotation(
            x=median.index[-1],
            y=float(median.iloc[-1]),
            text=f"Median · {float(median.iloc[-1]):.0f}",
            xanchor="left",
            xshift=6,
            yanchor="middle",
            showarrow=False,
            font=dict(color=INK, size=11),
        )

    fig.update_xaxes(title_text="Trading day in window (0 = entry at open)")
    fig.update_yaxes(title_text="Price index (entry open = 100)")
    return _base_layout(fig, height=470, hovermode="closest", right_margin=110)


def day_profile_chart(
    profile: pd.DataFrame, max_day: int = 30, min_windows: int = 10
) -> go.Figure:
    """Average daily return by position in the window."""
    data = profile.loc[
        (profile["day"] <= max_day) & (profile["n_windows"] >= min_windows)
    ].copy()
    colors = np.where(data["avg_return"] >= 0, BLUE, RED)
    customdata = np.column_stack(
        [data["hit_rate"] * 100, data["n_windows"], data["median_return"] * 100]
    )

    fig = go.Figure(
        go.Bar(
            x=data["day"],
            y=data["avg_return"] * 100,
            marker=dict(color=colors, line=dict(color=SURFACE, width=1)),
            customdata=customdata,
            hovertemplate=(
                "Day %{x}<br>"
                "Average return %{y:+.2f}%<br>"
                "Median return %{customdata[2]:+.2f}%<br>"
                "Hit rate %{customdata[0]:.0f}%<br>"
                "Windows %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_color=AXIS, line_width=1)
    fig.update_xaxes(title_text="Trading day in window", dtick=2)
    fig.update_yaxes(ticksuffix="%")
    return _base_layout(fig, height=380, hovermode="closest", showlegend=False)


def vesting_lag_chart(lag: pd.DataFrame) -> go.Figure:
    """Average return vs the vest-day close, by trading days after vest."""
    data = lag.loc[lag["lag"] <= 10].copy()
    x = data["lag"].to_numpy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=data["ci_high"] * 100,
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=data["ci_low"] * 100,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(42, 120, 214, 0.12)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=data["mean"] * 100,
            mode="lines+markers",
            name="Average return",
            line=dict(color=BLUE, width=2.5),
            marker=dict(
                color=BLUE,
                size=8,
                line=dict(color=SURFACE, width=2),
            ),
            customdata=np.column_stack(
                [
                    data["n"],
                    data["win_rate"] * 100,
                    data["ci_low"] * 100,
                    data["ci_high"] * 100,
                ]
            ),
            hovertemplate=(
                "Day %{x} after vest<br>"
                "Average %{y:+.2f}%<br>"
                "95% CI %{customdata[2]:+.1f}% to %{customdata[3]:+.1f}%<br>"
                "Higher than vest day %{customdata[1]:.0f}%<br>"
                "Windows %{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_color=AXIS, line_width=1)

    trough = data.loc[data["mean"].idxmin()]
    last = data.iloc[-1]
    for row, shift in ((trough, -18), (last, 16)):
        fig.add_annotation(
            x=row["lag"],
            y=row["mean"] * 100,
            text=f"{row['mean'] * 100:+.1f}%",
            showarrow=False,
            yshift=shift,
            font=dict(color=INK, size=11),
        )

    fig.update_xaxes(title_text="Trading days after vest (0 = sell on vest day)", dtick=1)
    fig.update_yaxes(title_text="Average return vs vest-day close", ticksuffix="%")
    return _base_layout(fig, height=420, hovermode="closest", showlegend=False)


def vesting_calendar_chart(calendar: pd.DataFrame) -> go.Figure:
    """Average return vs vest-day close for each calendar day after vest."""
    data = calendar.loc[np.isfinite(calendar["mean"])].copy()
    colors = np.where(data["mean"] >= 0, BLUE, RED)

    fig = go.Figure(
        go.Bar(
            x=data["dom"],
            y=data["mean"] * 100,
            marker=dict(color=colors, line=dict(color=SURFACE, width=1)),
            customdata=np.column_stack(
                [data["n"], data["win_rate"] * 100, data["mean_percentile"]]
            ),
            hovertemplate=(
                "Day %{x} of the month<br>"
                "Average vs vest %{y:+.2f}%<br>"
                "Higher than vest day %{customdata[1]:.0f}%<br>"
                "Average price percentile in window %{customdata[2]:.0f}<br>"
                "Windows %{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=15, line_color=AXIS, line_width=1)
    fig.add_annotation(
        x=15,
        y=1.03,
        yref="paper",
        text="vest date",
        showarrow=False,
        font=dict(color=MUTED, size=10),
    )
    fig.add_hline(y=0, line_color=AXIS, line_width=1)

    best = data.loc[data["mean"].idxmax()]
    fig.add_annotation(
        x=best["dom"],
        y=best["mean"] * 100,
        text=f"Day {int(best['dom'])}: {best['mean'] * 100:+.1f}%",
        showarrow=False,
        yshift=16,
        font=dict(color=INK, size=11),
    )

    fig.update_xaxes(title_text="Calendar day of the window month", dtick=1)
    fig.update_yaxes(title_text="Average return vs vest-day close", ticksuffix="%")
    return _base_layout(fig, height=400, hovermode="closest", showlegend=False)


def vesting_peak_chart(peaks: pd.DataFrame) -> go.Figure:
    """Count of windows whose post-vest high occurred on each calendar day."""
    data = peaks.copy()
    fig = go.Figure(
        go.Bar(
            x=data["dom"],
            y=data["count"],
            marker=dict(color=BLUE, line=dict(color=SURFACE, width=1)),
            customdata=data["share"] * 100,
            hovertemplate=(
                "Day %{x}<br>"
                "%{y} windows<br>"
                "%{customdata:.0f}% of windows<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=15, line_color=AXIS, line_width=1)
    fig.add_annotation(
        x=15,
        y=1.03,
        yref="paper",
        text="vest date",
        showarrow=False,
        font=dict(color=MUTED, size=10),
    )

    top = data.nlargest(2, "count")
    for row in top.itertuples(index=False):
        fig.add_annotation(
            x=row.dom,
            y=row.count,
            text=f"{row.count}",
            showarrow=False,
            yshift=14,
            font=dict(color=INK, size=11),
        )

    fig.update_xaxes(title_text="Calendar day of the window month", dtick=1)
    fig.update_yaxes(title_text="Windows peaking on this date", rangemode="tozero")
    return _base_layout(fig, height=400, hovermode="closest", showlegend=False)


def volume_ratio_chart(stats: pd.DataFrame) -> go.Figure:
    """Window average volume relative to the prior 60 trading days."""
    data = stats.sort_values("open_date")
    fig = go.Figure(
        go.Bar(
            x=data["label"],
            y=data["volume_ratio"],
            name="Volume ratio",
            marker=dict(color=BLUE, line=dict(color=SURFACE, width=1)),
            customdata=np.column_stack(
                [
                    data["open_date"].dt.strftime("%d %b %Y"),
                    data["avg_volume"] / 1_000_000,
                ]
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Window opens %{customdata[0]}<br>"
                "Volume vs prior 60 days %{y:.2f}×<br>"
                "Average daily volume %{customdata[1]:,.1f}M<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=1, line_color=AXIS, line_width=1)
    fig.update_yaxes(title_text="Average volume ÷ prior 60-day average")
    fig.update_xaxes(tickangle=-45, nticks=12)
    return _base_layout(fig, height=380, hovermode="closest", showlegend=False)

"""
visualize.py
============
Production-quality chart library for the Corporate Bond Maturity Wall strategy.

All charts are saved to ``outputs/`` at 300 DPI in seaborn-whitegrid style.
Chart functions are designed to be called from Jupyter notebooks and produce
both screen-display and file-save outputs.

Chart inventory
---------------
1. ``plot_maturity_wall_waterfall``     — per-issuer 8-quarter debt maturity bar
2. ``plot_universe_maturity_heatmap``   — cross-universe maturity concentration heatmap
3. ``plot_signal_decay_function``       — decay weight vs months-to-maturity
4. ``plot_stress_score_timeseries``     — single-company stress score + equity price
5. ``plot_lead_time_distribution``      — histogram of signal lead times
6. ``plot_ic_by_horizon_and_regime``    — grouped bar: IC by horizon × regime
7. ``plot_cumulative_pnl_with_carry_bleed`` — PnL with event shading
8. ``plot_xbrl_coverage_quality_report`` — stacked bar: data quality by sector
9. ``plot_rdm_distribution_by_rate_regime`` — violin: RDM by rate era
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI environments

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {
    "CRITICAL": "#C0392B",
    "ELEVATED": "#E67E22",
    "WATCH":    "#F39C12",
    "NORMAL":   "#27AE60",
    "peak":     "#E74C3C",
    "neutral":  "#2C3E50",
    "carry_bleed": "#E74C3C",
    "payoff":   "#27AE60",
}
DPI = 300
FIG_STANDARD = (12, 6)
FIG_WIDE     = (16, 6)
FIG_TALL     = (10, 8)


def _save(fig: plt.Figure, filename: str) -> Path:
    """Save figure to outputs/ and return the path."""
    path = config.OUTPUTS_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    logger.info("Chart saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# 1. Maturity wall waterfall
# ---------------------------------------------------------------------------

def plot_maturity_wall_waterfall(
    ticker: str,
    quarterly_schedule_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> plt.Figure:
    """
    Waterfall-style bar chart showing the debt maturity calendar for one issuer.

    The peak quarter is highlighted in red.  A horizontal dashed line marks
    the MCS threshold (35% of total forward debt).  This is **Figure 1** —
    the most intuitive chart for a non-quant portfolio manager.

    Parameters
    ----------
    ticker : str
        Company ticker.
    quarterly_schedule_df : pd.DataFrame
        Quarterly maturity schedule (all companies).
    as_of_date : pd.Timestamp
        Snapshot date; only future quarters shown.

    Returns
    -------
    plt.Figure
    """
    as_of = pd.Timestamp(as_of_date)
    co_data = quarterly_schedule_df[quarterly_schedule_df["ticker"] == ticker].copy()
    co_data = co_data[co_data["quarter_end_date"] >= as_of].copy()
    co_data = co_data.sort_values("quarter_end_date").head(config.LOOKFORWARD_QUARTERS)

    if co_data.empty:
        logger.warning("No data for %s waterfall chart", ticker)
        fig, ax = plt.subplots(figsize=FIG_STANDARD)
        ax.text(0.5, 0.5, f"No data for {ticker}", ha="center", va="center")
        return fig

    qsum = co_data.groupby("quarter_label")["debt_maturing_mm"].sum().reset_index()
    total = qsum["debt_maturing_mm"].sum()
    peak_idx = qsum["debt_maturing_mm"].idxmax()
    peak_q = qsum.loc[peak_idx, "quarter_label"]
    peak_debt = qsum.loc[peak_idx, "debt_maturing_mm"]
    mcs = peak_debt / total if total > 0 else 0.0
    threshold_line = total * config.MCS_HIGH_THRESHOLD

    colors = [
        PALETTE["peak"] if q == peak_q else "#3498DB"
        for q in qsum["quarter_label"]
    ]

    fig, ax = plt.subplots(figsize=FIG_STANDARD)
    bars = ax.bar(
        qsum["quarter_label"],
        qsum["debt_maturing_mm"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )

    ax.axhline(
        threshold_line,
        linestyle="--",
        color="#E67E22",
        linewidth=1.5,
        label=f"MCS threshold (35% × total = ${threshold_line:,.0f}MM)",
    )

    # Annotate peak bar
    ax.annotate(
        f"Peak: {peak_q}\n${peak_debt:,.0f}MM\nMCS={mcs:.0%}",
        xy=(peak_idx, peak_debt),
        xytext=(peak_idx + 0.3, peak_debt * 1.05),
        fontsize=9,
        color=PALETTE["peak"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PALETTE["peak"]),
    )

    ax.set_title(
        f"{ticker} — Debt Maturity Wall as of {as_of_date.strftime('%Y-%m-%d')}",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Quarter", fontsize=11)
    ax.set_ylabel("Debt Maturing ($MM)", fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    _save(fig, f"{ticker}_maturity_wall_{as_of_date.strftime('%Y%m%d')}.png")
    return fig


# ---------------------------------------------------------------------------
# 2. Universe maturity heatmap
# ---------------------------------------------------------------------------

def plot_universe_maturity_heatmap(
    rolling_signals_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    quarterly_schedule_df: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """
    Cross-universe heatmap: x=forward quarters, y=companies, color=debt_maturing.

    Companies are sorted by MCS (highest at top).  Red clusters in a single
    column indicate systemic maturity-wall risk in that quarter.

    Parameters
    ----------
    rolling_signals_df : pd.DataFrame
        Rolling signal panel (used for sorting by MCS).
    as_of_date : pd.Timestamp
        Snapshot date.
    quarterly_schedule_df : pd.DataFrame, optional
        If provided, uses actual maturity amounts; otherwise shows MCS ranking.

    Returns
    -------
    plt.Figure
    """
    as_of = pd.Timestamp(as_of_date)
    snap = rolling_signals_df[rolling_signals_df["as_of_date"] == as_of].copy()
    snap = snap.sort_values("MCS", ascending=False).reset_index(drop=True)

    if quarterly_schedule_df is not None and not quarterly_schedule_df.empty:
        # Build heatmap matrix from actual schedule
        future_qs = pd.date_range(start=as_of, periods=config.LOOKFORWARD_QUARTERS + 1, freq="QE")
        from src.utils import get_quarter_label
        q_labels = [get_quarter_label(d) for d in future_qs[1:]]

        matrix_rows = []
        for ticker in snap["ticker"]:
            co = quarterly_schedule_df[quarterly_schedule_df["ticker"] == ticker].copy()
            co = co[co["quarter_end_date"] >= as_of]
            qsum = co.groupby("quarter_label")["debt_maturing_mm"].sum()
            row = {q: qsum.get(q, 0.0) for q in q_labels}
            row["ticker"] = ticker
            matrix_rows.append(row)

        matrix_df = pd.DataFrame(matrix_rows).set_index("ticker")
        matrix_df = matrix_df.reindex(snap["ticker"])
        matrix_df = matrix_df.fillna(0)
    else:
        # Fallback: use MCS as single-column heatmap
        matrix_df = snap.set_index("ticker")[["MCS"]].fillna(0)

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    sns.heatmap(
        matrix_df,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Debt Maturing ($MM)"},
        fmt=".0f",
        annot=len(matrix_df) <= 25,  # only annotate if few companies
        annot_kws={"fontsize": 7},
    )
    ax.set_title(
        f"Universe Debt Maturity Heatmap — as of {as_of_date.strftime('%Y-%m-%d')}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Quarter", fontsize=11)
    ax.set_ylabel("Company (sorted by MCS)", fontsize=11)
    plt.tight_layout()
    _save(fig, f"universe_maturity_heatmap_{as_of_date.strftime('%Y%m%d')}.png")
    return fig


# ---------------------------------------------------------------------------
# 3. Signal decay function
# ---------------------------------------------------------------------------

def plot_signal_decay_function() -> plt.Figure:
    """
    Visualise the exponential decay weight as a function of months-to-maturity.

    Shows three half-life assumptions (6, 12, 18 months) on the same axes.
    Vertical lines mark the NEAR and MEDIUM proximity thresholds.

    This chart is the direct mathematical response to the *carry-bleed*
    problem: it shows how the position weight evolves as the maturity
    date approaches, and why the decay function produces lower drawdown
    than the raw ``1/months`` formula.

    Returns
    -------
    plt.Figure
    """
    from src.features import compute_signal_decay_weight

    months = np.linspace(0, 48, 300)
    halflives = [6, 12, 18]
    colors = ["#E74C3C", "#3498DB", "#27AE60"]

    fig, ax = plt.subplots(figsize=FIG_STANDARD)

    for hl, color in zip(halflives, colors):
        weights = [compute_signal_decay_weight(m, halflife=float(hl)) for m in months]
        ax.plot(months, weights, color=color, linewidth=2,
                label=f"Half-life = {hl} months")

    ax.axvline(config.MONTHS_NEAR_THRESHOLD,   color="red",    linestyle=":",
               linewidth=1.5, label=f"NEAR threshold ({config.MONTHS_NEAR_THRESHOLD}m)")
    ax.axvline(config.MONTHS_MEDIUM_THRESHOLD, color="orange", linestyle=":",
               linewidth=1.5, label=f"MEDIUM threshold ({config.MONTHS_MEDIUM_THRESHOLD}m)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Neutral weight (1.0)")

    ax.set_title("Signal Decay Function: Position Weight vs Months to Peak Maturity",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Months to Peak Maturity Quarter", fontsize=11)
    ax.set_ylabel("Decay Weight", fontsize=11)
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 5)
    ax.legend(fontsize=9)

    # Annotate key values
    ax.annotate("Weight ≈ 4× at maturity\n(maximum stress amplification)",
                xy=(1, 4.0), xytext=(8, 4.3), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("Weight ≈ 0.25 at 36m\n(distant — low allocation)",
                xy=(36, 0.25), xytext=(28, 0.8), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray"))

    plt.tight_layout()
    _save(fig, "signal_decay_function.png")
    return fig


# ---------------------------------------------------------------------------
# 4. Stress score time series
# ---------------------------------------------------------------------------

def plot_stress_score_timeseries(
    ticker: str,
    rolling_signals_df: pd.DataFrame,
    equity_prices_df: Optional[pd.DataFrame] = None,
    stress_events_df: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """
    Time series of stress_score for one company, overlaid with equity price.

    Vertical lines mark known stress events.  A horizontal dashed line marks
    the CRITICAL tier threshold (approximated as the 90th percentile).

    Parameters
    ----------
    ticker : str
    rolling_signals_df : pd.DataFrame
    equity_prices_df : pd.DataFrame, optional
    stress_events_df : pd.DataFrame, optional
        If provided, draws vertical lines at known stress event dates.

    Returns
    -------
    plt.Figure
    """
    co_signals = rolling_signals_df[rolling_signals_df["ticker"] == ticker].copy()
    co_signals["as_of_date"] = pd.to_datetime(co_signals["as_of_date"])
    co_signals = co_signals.sort_values("as_of_date")

    if co_signals.empty:
        logger.warning("No signal data for %s", ticker)
        fig, ax = plt.subplots(figsize=FIG_STANDARD)
        ax.text(0.5, 0.5, f"No data for {ticker}", ha="center", va="center")
        return fig

    fig, ax1 = plt.subplots(figsize=FIG_WIDE)

    # Stress score (primary y-axis)
    ax1.fill_between(co_signals["as_of_date"], co_signals["stress_score"],
                     alpha=0.25, color="#E74C3C")
    ax1.plot(co_signals["as_of_date"], co_signals["stress_score"],
             color="#C0392B", linewidth=2, label="Stress Score")

    # CRITICAL threshold line
    critical_thresh = co_signals["stress_score"].quantile(0.90)
    ax1.axhline(critical_thresh, color="#E74C3C", linestyle="--", linewidth=1,
                alpha=0.7, label="CRITICAL threshold (~90th pct)")

    ax1.set_ylabel("Stress Score", fontsize=11, color="#C0392B")
    ax1.tick_params(axis="y", colors="#C0392B")

    # Equity price (secondary y-axis)
    if equity_prices_df is not None and ticker in equity_prices_df.columns:
        ax2 = ax1.twinx()
        eq = equity_prices_df[[ticker]].dropna()
        ax2.plot(eq.index, eq[ticker], color="#2C3E50", linewidth=1.5,
                 alpha=0.7, label=f"{ticker} Price", linestyle="-")
        ax2.set_ylabel(f"{ticker} Equity Price ($)", fontsize=11, color="#2C3E50")
        ax2.tick_params(axis="y", colors="#2C3E50")
        ax2.legend(loc="upper right", fontsize=9)

    # Known stress events
    if stress_events_df is not None and not stress_events_df.empty:
        co_events = stress_events_df[stress_events_df["ticker"] == ticker]
        for _, ev in co_events.iterrows():
            try:
                ev_date = pd.Timestamp(ev["stress_start_date"])
                ax1.axvline(ev_date, color="darkred", linestyle=":", linewidth=2)
                ax1.text(ev_date, ax1.get_ylim()[1] * 0.95, ev.get("event_type", "Event"),
                         rotation=90, fontsize=8, color="darkred", va="top")
            except Exception:
                pass

    ax1.set_title(f"{ticker} — Stress Score Over Time", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Date", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    _save(fig, f"{ticker}_stress_timeseries.png")
    return fig


# ---------------------------------------------------------------------------
# 5. Lead-time distribution
# ---------------------------------------------------------------------------

def plot_lead_time_distribution(lead_time_df: pd.DataFrame) -> plt.Figure:
    """
    Histogram of signal lead times across all validated stress events.

    Parameters
    ----------
    lead_time_df : pd.DataFrame
        Output of ``signal.compute_signal_lead_time_analysis``.

    Returns
    -------
    plt.Figure
    """
    triggered = lead_time_df[lead_time_df.get("signal_triggered", True) == True].copy()
    lead_times = triggered["lead_time_months"].dropna()

    if lead_times.empty:
        fig, ax = plt.subplots(figsize=FIG_STANDARD)
        ax.text(0.5, 0.5, "No lead time data available", ha="center", va="center")
        return fig

    median_lt = lead_times.median()

    fig, ax = plt.subplots(figsize=FIG_STANDARD)
    ax.hist(lead_times, bins=min(15, len(lead_times)), color="#3498DB",
            edgecolor="white", linewidth=0.8, alpha=0.85)

    ax.axvline(3,  color="orange", linestyle="--", linewidth=2,
               label="Min actionable (3m)")
    ax.axvline(18, color="orange", linestyle="--", linewidth=2,
               label="Max practical (18m)")
    ax.axvline(median_lt, color="red", linestyle="-", linewidth=2,
               label=f"Median lead time: {median_lt:.1f}m")

    ax.set_title("Signal Lead Time Distribution — Historical Stress Events",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Lead Time (months before stress event)", fontsize=11)
    ax.set_ylabel("Count of Events", fontsize=11)
    ax.legend(fontsize=9)

    # Shaded actionable zone
    ax.axvspan(3, 18, alpha=0.08, color="green", label="Actionable window")
    ax.text(10.5, ax.get_ylim()[1] * 0.85, "Actionable\nWindow",
            ha="center", fontsize=9, color="green", alpha=0.7)

    plt.tight_layout()
    _save(fig, "signal_lead_time_distribution.png")
    return fig


# ---------------------------------------------------------------------------
# 6. IC by horizon and regime
# ---------------------------------------------------------------------------

def plot_ic_by_horizon_and_regime(ic_results_dict: dict) -> plt.Figure:
    """
    Grouped bar chart: Spearman IC by return horizon, coloured by signal variant.

    Parameters
    ----------
    ic_results_dict : dict
        Output of ``signal.compute_information_coefficient``.

    Returns
    -------
    plt.Figure
    """
    horizons  = ["30d", "60d", "90d", "180d", "252d"]
    variants  = ["raw_stress_IC", "decay_stress_IC", "trace_enriched_IC", "xbrl_only_IC"]
    labels    = ["Raw Stress", "Decay-Adjusted", "TRACE-Enriched", "XBRL-Only"]
    colors    = ["#E74C3C", "#3498DB", "#27AE60", "#F39C12"]

    x = np.arange(len(horizons))
    width = 0.2

    fig, ax = plt.subplots(figsize=FIG_WIDE)

    for i, (variant, label, color) in enumerate(zip(variants, labels, colors)):
        ic_vals = [
            ic_results_dict.get(h, {}).get(variant, np.nan)
            for h in horizons
        ]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, ic_vals, width, label=label,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Spearman IC by Return Horizon and Signal Variant",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Return Horizon", fontsize=11)
    ax.set_ylabel("Spearman IC (positive = signal correct)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.3, 0.5)

    plt.tight_layout()
    _save(fig, "ic_by_horizon_and_variant.png")
    return fig


# ---------------------------------------------------------------------------
# 7. Cumulative PnL with carry-bleed shading
# ---------------------------------------------------------------------------

def plot_cumulative_pnl_with_carry_bleed(
    backtest_df: pd.DataFrame,
    event_overlay_df: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """
    Cumulative PnL with event windows shaded to show carry bleed vs payoff.

    Red shading  = carry-bleed period (signal active, equity still rising).
    Green shading = payoff period (equity falls, short profits).

    This is the **most honest chart in the repository** — it shows both the
    profitability AND the pain of being early.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of ``backtest.run_long_short_backtest``.
    event_overlay_df : pd.DataFrame, optional
        Output of ``backtest.run_stress_event_overlay``.

    Returns
    -------
    plt.Figure
    """
    if backtest_df is None or backtest_df.empty:
        fig, ax = plt.subplots(figsize=FIG_WIDE)
        ax.text(0.5, 0.5, "No backtest data available", ha="center", va="center")
        return fig

    bt = backtest_df.copy()
    bt["date"] = pd.to_datetime(bt["date"])
    bt = bt.sort_values("date")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]})

    # Top: cumulative PnL
    if "cumulative_pnl" not in bt.columns:
        bt["cumulative_pnl"] = (1 + bt["net_return"]).cumprod()
    ax1.plot(bt["date"], bt["cumulative_pnl"] - 1, color="#2C3E50",
             linewidth=2, label="Cumulative Net Return")
    ax1.fill_between(bt["date"], bt["cumulative_pnl"] - 1, 0,
                     alpha=0.15, color="#2C3E50")

    # Event shading
    if event_overlay_df is not None and not event_overlay_df.empty:
        for _, ev in event_overlay_df.iterrows():
            signal_d     = pd.Timestamp(ev.get("signal_date", pd.NaT))
            stress_start = pd.Timestamp(ev.get("stress_start_date", pd.NaT))
            peak_d       = pd.Timestamp(ev.get("peak_stress_date", pd.NaT))

            if pd.notna(signal_d) and pd.notna(stress_start):
                ax1.axvspan(signal_d, stress_start, alpha=0.15,
                            color=PALETTE["carry_bleed"], label="_carry_bleed")
            if pd.notna(stress_start) and pd.notna(peak_d):
                ax1.axvspan(stress_start, peak_d, alpha=0.15,
                            color=PALETTE["payoff"], label="_payoff")

    # Legend for shading
    carry_patch  = mpatches.Patch(color=PALETTE["carry_bleed"], alpha=0.5,
                                   label="Carry-bleed period (before event)")
    payoff_patch = mpatches.Patch(color=PALETTE["payoff"], alpha=0.5,
                                   label="Payoff period (event crystallises)")
    ax1.legend(handles=[carry_patch, payoff_patch] +
               [plt.Line2D([0], [0], color="#2C3E50", linewidth=2, label="Cumulative Return")],
               fontsize=9)

    ax1.set_title("Cumulative PnL: Long-Short Maturity Wall Strategy\n"
                  "(Red=carry bleed, Green=payoff)", fontsize=13, fontweight="bold")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # Bottom: drawdown
    if "drawdown" not in bt.columns:
        roll_max = bt["cumulative_pnl"].cummax()
        bt["drawdown"] = (bt["cumulative_pnl"] - roll_max) / roll_max
    ax2.fill_between(bt["date"], bt["drawdown"], 0,
                     alpha=0.7, color="#E74C3C")
    ax2.set_ylabel("Drawdown", fontsize=10)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    plt.tight_layout()
    _save(fig, "cumulative_pnl_carry_bleed.png")
    return fig


# ---------------------------------------------------------------------------
# 8. XBRL coverage quality report
# ---------------------------------------------------------------------------

def plot_xbrl_coverage_quality_report(
    xbrl_quality_report_df: pd.DataFrame,
    universe_df: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """
    Stacked bar chart: XBRL data quality by sector.

    x = sector, y = number of companies, colors = coverage_flag distribution
    (COMPLETE / PARTIAL / MINIMAL / EXCLUDED).

    Parameters
    ----------
    xbrl_quality_report_df : pd.DataFrame
        Output of ``xbrl_parser.build_universe_maturity_schedules`` quality report.
    universe_df : pd.DataFrame, optional
        Used to join sector information.

    Returns
    -------
    plt.Figure
    """
    if xbrl_quality_report_df is None or xbrl_quality_report_df.empty:
        fig, ax = plt.subplots(figsize=FIG_STANDARD)
        ax.text(0.5, 0.5, "No quality report data", ha="center", va="center")
        return fig

    qdf = xbrl_quality_report_df.copy()
    if universe_df is not None and "sector" in universe_df.columns:
        qdf = qdf.merge(universe_df[["ticker", "sector"]], on="ticker", how="left")
    else:
        qdf["sector"] = "Unknown"

    # Determine dominant coverage flag per company
    def dominant_flag(row: pd.Series) -> str:
        if row.get("recommendation") == "EXCLUDE":
            return "EXCLUDED"
        cf = max(
            ("COMPLETE", row.get("complete_filings", 0)),
            ("PARTIAL",  row.get("partial_filings", 0)),
            ("MINIMAL",  row.get("minimal_filings", 0)),
            key=lambda x: x[1],
        )
        return cf[0]

    qdf["dominant_flag"] = qdf.apply(dominant_flag, axis=1)

    flag_order  = ["COMPLETE", "PARTIAL", "MINIMAL", "EXCLUDED"]
    flag_colors = ["#27AE60", "#F39C12", "#E67E22", "#C0392B"]

    sector_summary = (
        qdf.groupby(["sector", "dominant_flag"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=flag_order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=FIG_STANDARD)
    sector_summary.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=flag_colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_title("XBRL Data Quality by Sector — Maturity Schedule Coverage",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Sector", fontsize=11)
    ax.set_ylabel("Number of Companies", fontsize=11)
    ax.legend(title="Coverage Flag", loc="upper right", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save(fig, "xbrl_coverage_quality_report.png")
    return fig


# ---------------------------------------------------------------------------
# 9. RDM distribution by rate regime
# ---------------------------------------------------------------------------

def plot_rdm_distribution_by_rate_regime(
    signal_df: pd.DataFrame,
    macro_rates_df: Optional[pd.DataFrame] = None,
) -> plt.Figure:
    """
    Violin plot: RDM distribution split by rate environment era.

    Compares the 2014–2018 low-rate era vs the 2022–2023 high-rate era.
    Expected finding: RDM distribution shifts right in 2022–2023, validating
    that the multiplier is capturing genuine refinancing difficulty.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Rolling signal panel (contains RDM and as_of_date).
    macro_rates_df : pd.DataFrame, optional
        Not used directly; era classification is based on as_of_date year.

    Returns
    -------
    plt.Figure
    """
    df = signal_df.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["year"] = df["as_of_date"].dt.year

    def rate_era(year: int) -> str:
        if year <= 2018:
            return "2014–2018\n(Low-rate era)"
        if year >= 2022:
            return "2022–2023\n(High-rate era)"
        return "2019–2021\n(Transition)"

    df["era"] = df["year"].apply(rate_era)
    era_order = ["2014–2018\n(Low-rate era)", "2019–2021\n(Transition)", "2022–2023\n(High-rate era)"]
    df["era"] = pd.Categorical(df["era"], categories=era_order, ordered=True)

    rdm_clean = df[df["RDM"].between(0.1, 5.0)].copy()  # clip outliers for visualisation

    fig, ax = plt.subplots(figsize=FIG_STANDARD)
    sns.violinplot(
        data=rdm_clean,
        x="era",
        y="RDM",
        order=era_order,
        palette=["#3498DB", "#F39C12", "#E74C3C"],
        inner="quartile",
        ax=ax,
    )

    ax.axhline(config.RDM_STRESS_THRESHOLD, color="red", linestyle="--",
               linewidth=1.5, label=f"Stress threshold (RDM={config.RDM_STRESS_THRESHOLD})")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="RDM = 1.0 (neutral)")

    ax.set_title("RDM Distribution by Rate Environment Era",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Rate Environment Era", fontsize=11)
    ax.set_ylabel("Refinancing Difficulty Multiplier (RDM)", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save(fig, "rdm_distribution_by_rate_regime.png")
    return fig

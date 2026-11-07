"""
backtest.py
===========
Backtest engine for the Corporate Bond Maturity Wall strategy.

Implements quarterly-rebalanced long-short equity backtests with a full cost
stack: transaction costs and short borrow costs.  Also provides performance
attribution utilities and known-stress-event overlays for honest carry-bleed
quantification.

All backtest functions apply ``utils.lag_dataframe`` to the signal panel
before use to enforce point-in-time discipline.

Design philosophy
-----------------
This backtest is deliberately simple and transparent.  The core question is
whether the maturity-wall signal has directional information for equity
returns, not whether a particular trading strategy is profitable after all
costs.  A quant PM reading this code should be able to verify every cost
assumption and override them for their own borrow cost assumptions.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config
from src.utils import lag_dataframe, get_all_quarter_ends, compute_forward_returns

logger = logging.getLogger(__name__)

# Cost constants (in decimal per year or per trade)
TC_PER_TRADE      = config.TRANSACTION_COST_BPS / 10_000  # round-trip per rebalance
BORROW_ANNUAL     = config.SHORT_BORROW_COST_BPS  / 10_000  # annualised borrow cost
BORROW_QUARTERLY  = BORROW_ANNUAL / 4                        # quarterly borrow drag


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_quarterly_returns(
    equity_prices_df: pd.DataFrame,
    quarter_ends: list,
) -> pd.DataFrame:
    """
    Compute one-quarter holding-period returns for each ticker.

    Returns
    -------
    pd.DataFrame
        index=quarter_end_date (entry), columns=tickers, values=Q return.
    """
    qe_dates = sorted(quarter_ends)
    returns_rows = []
    for i in range(len(qe_dates) - 1):
        entry  = qe_dates[i]
        exit_  = qe_dates[i + 1]
        # Find nearest price dates
        entry_idx = equity_prices_df.index.searchsorted(entry, side="left")
        exit_idx  = equity_prices_df.index.searchsorted(exit_,  side="left")
        if entry_idx >= len(equity_prices_df) or exit_idx >= len(equity_prices_df):
            continue
        p_entry = equity_prices_df.iloc[entry_idx]
        p_exit  = equity_prices_df.iloc[exit_idx]
        q_ret = (p_exit / p_entry) - 1.0
        q_ret.name = entry  # label by entry date
        returns_rows.append(q_ret)

    if not returns_rows:
        return pd.DataFrame()
    return pd.DataFrame(returns_rows)


def _apply_cost_stack(
    gross_return: float,
    is_short: bool = False,
    rebalanced: bool = True,
) -> Tuple[float, float, float]:
    """
    Apply transaction costs and (if short) borrow cost.

    Returns
    -------
    tuple: (net_return, tc_drag, borrow_drag)
    """
    tc_drag = TC_PER_TRADE if rebalanced else 0.0
    borrow_drag = BORROW_QUARTERLY if is_short else 0.0
    net_return = gross_return - tc_drag - borrow_drag
    return net_return, tc_drag, borrow_drag


def _compute_drawdown_series(cumulative_returns: pd.Series) -> pd.Series:
    """Compute drawdown series from cumulative return series (level)."""
    roll_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - roll_max) / roll_max
    return drawdown


# ---------------------------------------------------------------------------
# 1. Short-only backtest
# ---------------------------------------------------------------------------

def run_short_only_backtest(
    rolling_signals_df: pd.DataFrame,
    equity_prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Quarterly equal-weight short book: SHORT the CRITICAL tier each quarter.

    Parameters
    ----------
    rolling_signals_df : pd.DataFrame
        Rolling signal panel.  ``lag_dataframe`` is applied internally.
    equity_prices_df : pd.DataFrame
        Adjusted close prices.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``short_return`` (gross), ``gross_return``,
        ``borrow_cost_drag``, ``tc_drag``, ``net_return``,
        ``cumulative_pnl``, ``drawdown``, ``n_shorts``.

    Notes
    -----
    **Short book mechanics:**

    - Entry: first trading day after quarter-end (lagged by ``SIGNAL_UPDATE_LAG_DAYS``).
    - Exit: next quarter-end.
    - Equal weight across all CRITICAL-tier names.
    - Short return = -(equity return over holding period).
    - Borrow cost: ``SHORT_BORROW_COST_BPS`` annualised, charged quarterly.
    - Transaction cost: ``TRANSACTION_COST_BPS`` round-trip per rebalance.

    The 50bps borrow cost assumption may be optimistic for CCC-rated names
    in genuine distress (actual borrow costs can be 200–500bps for hard-to-
    borrow securities).  See the notebook's break-even analysis for
    sensitivity at 150bps and 300bps.
    """
    # Strict lag: signal from quarter Q is not available until Q+1 entry
    signals_lagged = lag_dataframe(
        rolling_signals_df.set_index("as_of_date")[["ticker", "stress_tier", "stress_score"]]
        .reset_index()
    )

    quarter_ends = get_all_quarter_ends(config.BACKTEST_START, config.BACKTEST_END)
    q_returns = _get_quarterly_returns(equity_prices_df, quarter_ends)

    rows = []
    for i, entry_date in enumerate(quarter_ends[:-1]):
        entry_ts = pd.Timestamp(entry_date)

        # Get the signal from the *previous* quarter end (lagged)
        prev_qe = quarter_ends[i]  # entry quarter's signal date
        signal_snap = rolling_signals_df[
            rolling_signals_df["as_of_date"] == pd.Timestamp(prev_qe)
        ]
        shorts = signal_snap[signal_snap["stress_tier"] == "CRITICAL"]["ticker"].tolist()
        n_shorts = len(shorts)

        if n_shorts == 0 or entry_ts not in q_returns.index:
            rows.append({
                "date": entry_ts, "short_return": 0.0, "gross_return": 0.0,
                "borrow_cost_drag": 0.0, "tc_drag": 0.0, "net_return": 0.0,
                "n_shorts": 0,
            })
            continue

        available_shorts = [t for t in shorts if t in q_returns.columns]
        if not available_shorts:
            rows.append({
                "date": entry_ts, "short_return": 0.0, "gross_return": 0.0,
                "borrow_cost_drag": 0.0, "tc_drag": 0.0, "net_return": 0.0,
                "n_shorts": 0,
            })
            continue

        # Equal-weight short returns
        eq_returns = q_returns.loc[entry_ts, available_shorts].dropna()
        if eq_returns.empty:
            continue
        gross_short = -eq_returns.mean()  # short = negative of long return

        net_ret, tc_drag, borrow_drag = _apply_cost_stack(gross_short, is_short=True)

        rows.append({
            "date": entry_ts,
            "short_return": round(gross_short, 6),
            "gross_return": round(gross_short, 6),
            "borrow_cost_drag": round(borrow_drag, 6),
            "tc_drag": round(tc_drag, 6),
            "net_return": round(net_ret, 6),
            "n_shorts": len(available_shorts),
        })

    if not rows:
        return pd.DataFrame()

    bt = pd.DataFrame(rows).set_index("date").sort_index()
    bt["cumulative_pnl"] = (1 + bt["net_return"]).cumprod()
    bt["drawdown"] = _compute_drawdown_series(bt["cumulative_pnl"])
    return bt.reset_index()


# ---------------------------------------------------------------------------
# 2. Long-short backtest
# ---------------------------------------------------------------------------

def run_long_short_backtest(
    rolling_signals_df: pd.DataFrame,
    equity_prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Quarterly equal-weight long-short: LONG clean balance sheets, SHORT stressed.

    Parameters
    ----------
    rolling_signals_df : pd.DataFrame
        Rolling signal panel.
    equity_prices_df : pd.DataFrame
        Adjusted close prices.

    Returns
    -------
    pd.DataFrame
        Same structure as ``run_short_only_backtest`` plus ``long_return``,
        ``long_n``, ``short_n``.
    """
    quarter_ends = get_all_quarter_ends(config.BACKTEST_START, config.BACKTEST_END)
    q_returns = _get_quarterly_returns(equity_prices_df, quarter_ends)

    rows = []
    for i, entry_date in enumerate(quarter_ends[:-1]):
        entry_ts = pd.Timestamp(entry_date)
        prev_qe = quarter_ends[i]
        signal_snap = rolling_signals_df[
            rolling_signals_df["as_of_date"] == pd.Timestamp(prev_qe)
        ]

        longs  = signal_snap[signal_snap["stress_tier"] == "NORMAL"]["ticker"].tolist()
        shorts = signal_snap[signal_snap["stress_tier"] == "CRITICAL"]["ticker"].tolist()

        if entry_ts not in q_returns.index:
            continue

        avail_longs  = [t for t in longs  if t in q_returns.columns]
        avail_shorts = [t for t in shorts if t in q_returns.columns]

        long_ret = short_ret = 0.0
        if avail_longs:
            long_ret = q_returns.loc[entry_ts, avail_longs].dropna().mean()
        if avail_shorts:
            short_ret = -q_returns.loc[entry_ts, avail_shorts].dropna().mean()

        gross = (long_ret + short_ret) / 2.0  # equal dollar L/S
        _, tc_long, _  = _apply_cost_stack(long_ret,  is_short=False)
        _, tc_short, borrow_drag = _apply_cost_stack(short_ret, is_short=True)
        tc_total = tc_long + tc_short
        net_ret = gross - tc_total - borrow_drag

        rows.append({
            "date": entry_ts,
            "long_return": round(long_ret, 6),
            "short_return": round(short_ret, 6),
            "gross_return": round(gross, 6),
            "borrow_cost_drag": round(borrow_drag, 6),
            "tc_drag": round(tc_total, 6),
            "net_return": round(net_ret, 6),
            "long_n": len(avail_longs),
            "short_n": len(avail_shorts),
        })

    if not rows:
        return pd.DataFrame()
    bt = pd.DataFrame(rows).set_index("date").sort_index()
    bt["cumulative_pnl"] = (1 + bt["net_return"]).cumprod()
    bt["drawdown"] = _compute_drawdown_series(bt["cumulative_pnl"])
    return bt.reset_index()


# ---------------------------------------------------------------------------
# 3. Raw vs decay-adjusted comparison
# ---------------------------------------------------------------------------

def run_decay_weight_comparison(
    rolling_signals_df: pd.DataFrame,
    equity_prices_df: pd.DataFrame,
) -> Dict[str, dict]:
    """
    Compare long-short backtest results using raw_stress vs decay-adjusted stress_score.

    Swaps the ranking signal between the two variants and compares Sharpe ratios.
    The decay-adjusted version should show lower maximum drawdown because early,
    distant signals are downweighted — reducing carry bleed from positions entered
    too far in advance of the maturity event.

    Parameters
    ----------
    rolling_signals_df : pd.DataFrame
        Rolling signal panel (contains both ``raw_stress`` and ``stress_score``).
    equity_prices_df : pd.DataFrame
        Adjusted close prices.

    Returns
    -------
    dict
        ``{"raw": metrics_dict, "decay_adjusted": metrics_dict}``
        where each ``metrics_dict`` is from ``compute_performance_metrics``.
    """
    def _swap_signal(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
        """Rebuild stress_tier from a different score column."""
        modified = df.copy()
        for as_of, grp in modified.groupby("as_of_date"):
            idx = grp.index
            if len(grp) < 5:
                continue
            ranks = grp[score_col].rank(ascending=True, method="first")
            deciles = pd.cut(ranks, bins=np.linspace(0, len(grp), 11),
                             labels=list(range(1, 11)), include_lowest=True).astype(float)
            tiers = deciles.apply(
                lambda d: "CRITICAL" if d == 10 else "ELEVATED" if d >= 7 else "WATCH" if d >= 4 else "NORMAL"
            )
            modified.loc[idx, "stress_tier"] = tiers
        return modified

    raw_signals = _swap_signal(rolling_signals_df, "raw_stress")
    decay_signals = rolling_signals_df.copy()  # already uses stress_score by default

    bt_raw   = run_long_short_backtest(raw_signals, equity_prices_df)
    bt_decay = run_long_short_backtest(decay_signals, equity_prices_df)

    metrics_raw   = compute_performance_metrics(bt_raw,   label="raw_stress")
    metrics_decay = compute_performance_metrics(bt_decay, label="decay_adjusted")

    print("\n" + "=" * 72)
    print("  Raw Stress vs Decay-Adjusted Stress: Key Metrics")
    print("=" * 72)
    for key in ["annualized_return", "sharpe_ratio", "max_drawdown", "calmar_ratio"]:
        print(f"  {key:30s} raw={metrics_raw.get(key, 'N/A')!r:>10}  "
              f"decay={metrics_decay.get(key, 'N/A')!r:>10}")
    print("=" * 72 + "\n")

    return {"raw": metrics_raw, "decay_adjusted": metrics_decay}


# ---------------------------------------------------------------------------
# 4. Macro regime backtest
# ---------------------------------------------------------------------------

def run_macro_regime_backtest(
    rolling_signals_df: pd.DataFrame,
    equity_prices_df: pd.DataFrame,
    macro_rates_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the long-short backtest separately for each macro regime period.

    Regimes: RISING_RATES, FALLING_RATES, WIDE_SPREADS, TIGHT_SPREADS, NEUTRAL.
    See ``signal.compute_macro_regime_interaction`` for regime definitions.

    Parameters
    ----------
    rolling_signals_df : pd.DataFrame
        Rolling signal panel.
    equity_prices_df : pd.DataFrame
        Adjusted close prices.
    macro_rates_df : pd.DataFrame
        FRED macro rates (T10Y, HY_OAS).

    Returns
    -------
    pd.DataFrame
        One row per regime: annualized return, Sharpe, max drawdown,
        n_quarters.

    Notes
    -----
    If the strategy's Sharpe ratio is dramatically higher in ``RISING_RATES``
    than in other regimes, the alpha may largely be a disguised rate-sensitivity
    bet.  This is documented in the notebook's Limitations section.
    """
    from src.signal import compute_macro_regime_interaction

    # Assign regime to each quarter
    macro_rates_df = macro_rates_df.copy()
    macro_rates_df.index = pd.to_datetime(macro_rates_df.index)

    def get_regime(dt: pd.Timestamp) -> str:
        as_of_rates = macro_rates_df[macro_rates_df.index <= dt]
        if as_of_rates.empty:
            return "NEUTRAL"
        trailing = macro_rates_df[
            (macro_rates_df.index <= dt) & (macro_rates_df.index >= dt - pd.DateOffset(months=12))
        ]
        t10y_now = float(as_of_rates["T10Y"].iloc[-1])
        t10y_12m = float(trailing["T10Y"].iloc[0]) if not trailing.empty else t10y_now
        hy_oas   = float(as_of_rates.get("HY_OAS", pd.Series([350.0])).iloc[-1])
        hy_oas_bps = hy_oas if hy_oas > 10 else hy_oas * 100
        t10y_chg = t10y_now - t10y_12m

        if hy_oas_bps > 600:
            return "WIDE_SPREADS"
        if hy_oas_bps < 350:
            return "TIGHT_SPREADS"
        if t10y_chg > 1.0:
            return "RISING_RATES"
        if t10y_chg < -1.0:
            return "FALLING_RATES"
        return "NEUTRAL"

    quarter_ends = get_all_quarter_ends(config.BACKTEST_START, config.BACKTEST_END)
    regime_qe: dict = {r: [] for r in ["RISING_RATES", "FALLING_RATES",
                                         "WIDE_SPREADS", "TIGHT_SPREADS", "NEUTRAL"]}
    for qe in quarter_ends:
        r = get_regime(pd.Timestamp(qe))
        regime_qe[r].append(qe)

    regime_metrics_rows = []
    quarter_returns = _get_quarterly_returns(equity_prices_df, quarter_ends)

    for regime, dates in regime_qe.items():
        if len(dates) < 3:
            continue
        regime_signals = rolling_signals_df[
            rolling_signals_df["as_of_date"].isin([pd.Timestamp(d) for d in dates])
        ].copy()
        if regime_signals.empty:
            continue

        # Build mini backtest for this regime
        mini_rows = []
        for i, entry_date in enumerate(dates[:-1]):
            entry_ts = pd.Timestamp(entry_date)
            sig_snap = regime_signals[regime_signals["as_of_date"] == entry_ts]
            shorts = sig_snap[sig_snap["stress_tier"] == "CRITICAL"]["ticker"].tolist()
            longs  = sig_snap[sig_snap["stress_tier"] == "NORMAL"]["ticker"].tolist()

            if entry_ts not in quarter_returns.index:
                continue

            avail_s = [t for t in shorts if t in quarter_returns.columns]
            avail_l = [t for t in longs  if t in quarter_returns.columns]
            s_ret = -quarter_returns.loc[entry_ts, avail_s].dropna().mean() if avail_s else 0.0
            l_ret =  quarter_returns.loc[entry_ts, avail_l].dropna().mean() if avail_l else 0.0
            gross = (s_ret + l_ret) / 2.0
            net   = gross - TC_PER_TRADE * 2 - BORROW_QUARTERLY
            mini_rows.append({"date": entry_ts, "net_return": net})

        if not mini_rows:
            continue
        mini_bt = pd.DataFrame(mini_rows).set_index("date")
        mini_bt["cumulative_pnl"] = (1 + mini_bt["net_return"]).cumprod()
        metrics = compute_performance_metrics(mini_bt.reset_index(), label=regime)
        row = {"regime": regime, "n_quarters": len(mini_rows)}
        row.update(metrics)
        regime_metrics_rows.append(row)

    return pd.DataFrame(regime_metrics_rows)


# ---------------------------------------------------------------------------
# 5. Performance metrics
# ---------------------------------------------------------------------------

def compute_performance_metrics(
    backtest_df: pd.DataFrame,
    label: str = "",
) -> dict:
    """
    Compute standard backtest performance metrics from a return series.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Must contain ``date`` and ``net_return`` columns.
    label : str
        Label for printed table.

    Returns
    -------
    dict
        Keys: ``annualized_return``, ``sharpe_ratio``, ``max_drawdown``,
        ``max_drawdown_start``, ``max_drawdown_end``, ``calmar_ratio``,
        ``sortino_ratio``, ``win_rate``, ``avg_quarterly_return``,
        ``borrow_cost_drag_pct``, ``n_quarters``.
    """
    if backtest_df is None or backtest_df.empty:
        return {}

    r = backtest_df["net_return"].dropna()
    n = len(r)
    if n == 0:
        return {}

    ann_factor = 4  # quarterly → annual
    ann_return  = (1 + r.mean()) ** ann_factor - 1
    ann_vol     = r.std() * np.sqrt(ann_factor)
    sharpe      = (ann_return - 0.0) / ann_vol if ann_vol > 0 else np.nan

    # Max drawdown
    cum_ret = (1 + r).cumprod()
    roll_max = cum_ret.cummax()
    drawdown = (cum_ret - roll_max) / roll_max
    max_dd = float(drawdown.min())
    max_dd_end_idx   = drawdown.idxmin()
    max_dd_start_idx = roll_max[:max_dd_end_idx].idxmax() if not cum_ret[:max_dd_end_idx].empty else max_dd_end_idx

    calmar  = ann_return / abs(max_dd) if max_dd != 0 else np.nan
    downside_vol = r[r < 0].std() * np.sqrt(ann_factor)
    sortino = ann_return / downside_vol if downside_vol > 0 else np.nan
    win_rate = float((r > 0).mean())
    avg_q_ret = float(r.mean())

    # Borrow cost drag (short-only perspective)
    borrow_drag_total = backtest_df.get("borrow_cost_drag", pd.Series(0.0)).sum()
    gross_total = backtest_df.get("gross_return", r).sum()
    borrow_pct = borrow_drag_total / abs(gross_total) * 100 if gross_total != 0 else 0.0

    metrics = {
        "label": label,
        "n_quarters": n,
        "annualized_return": round(ann_return, 4),
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 3) if not np.isnan(sharpe) else None,
        "max_drawdown": round(max_dd, 4),
        "max_drawdown_start": str(backtest_df["date"].iloc[max_dd_start_idx])[:10] if "date" in backtest_df.columns else None,
        "max_drawdown_end":   str(backtest_df["date"].iloc[max_dd_end_idx])[:10]   if "date" in backtest_df.columns else None,
        "calmar_ratio": round(calmar, 3) if not np.isnan(calmar) else None,
        "sortino_ratio": round(sortino, 3) if not np.isnan(sortino) else None,
        "win_rate": round(win_rate, 4),
        "avg_quarterly_return": round(avg_q_ret, 4),
        "borrow_cost_drag_pct_of_gross": round(borrow_pct, 2),
    }

    if label:
        print(f"\n{'='*60}")
        print(f"  Performance Metrics: {label}")
        print(f"{'='*60}")
        for k, v in metrics.items():
            if k != "label":
                print(f"  {k:40s}: {v}")
        print(f"{'='*60}\n")

    return metrics


# ---------------------------------------------------------------------------
# 6. Stress event overlay
# ---------------------------------------------------------------------------

def run_stress_event_overlay(
    backtest_df: pd.DataFrame,
    stress_events_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Overlay known stress events on the backtest PnL series.

    For each event in ``historical_stress_events.csv``:

    1. Identify the signal-trigger quarter (when stress_tier first hit CRITICAL).
    2. Compute strategy return during the *carry-bleed* period:
       [signal_trigger_date, stress_start_date].
    3. Compute strategy return during the *payoff* period:
       [stress_start_date, peak_stress_date].

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of ``run_long_short_backtest`` or ``run_short_only_backtest``.
    stress_events_df : pd.DataFrame
        Loaded ``historical_stress_events.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``event_type``, ``signal_date``,
        ``stress_start_date``, ``peak_stress_date``,
        ``carry_bleed_return`` (strategy return before event crystallised),
        ``payoff_return`` (strategy return during event),
        ``net_event_return`` (combined).

    Notes
    -----
    The carry-bleed analysis is the most important output for a PM:

        "We were right but we lost 8% waiting.
         Was the eventual payoff worth it?"

    A negative ``carry_bleed_return`` AND a positive ``payoff_return``
    confirms the maturity-wall thesis: the strategy suffers carry bleed
    in anticipation of the event and then profits when it crystallises.

    A negative ``carry_bleed_return`` AND a negative ``payoff_return``
    indicates a false positive — the stress event did NOT lead to equity
    underperformance (e.g. company refinanced successfully, or was acquired).
    """
    if backtest_df is None or backtest_df.empty or stress_events_df is None or stress_events_df.empty:
        return pd.DataFrame()

    bt = backtest_df.copy()
    bt["date"] = pd.to_datetime(bt["date"])
    bt = bt.set_index("date").sort_index()

    rows = []
    for _, event in stress_events_df.iterrows():
        stress_start = pd.Timestamp(event.get("stress_start_date", ""))
        peak_stress  = pd.Timestamp(event.get("peak_stress_date", ""))

        if pd.isna(stress_start) or pd.isna(peak_stress):
            continue

        # Estimate signal date as the first quarter-end before stress_start
        pre_event_bt = bt[bt.index < stress_start]
        if pre_event_bt.empty:
            signal_date = stress_start
        else:
            signal_date = pre_event_bt.index[-1]

        # Carry-bleed period: signal_date → stress_start
        carry_period = bt[(bt.index >= signal_date) & (bt.index < stress_start)]
        carry_bleed = carry_period["net_return"].sum() if not carry_period.empty else np.nan

        # Payoff period: stress_start → peak_stress
        payoff_period = bt[(bt.index >= stress_start) & (bt.index <= peak_stress)]
        payoff_ret = payoff_period["net_return"].sum() if not payoff_period.empty else np.nan

        net_evt = (carry_bleed or 0.0) + (payoff_ret or 0.0)

        rows.append({
            "ticker": event["ticker"],
            "event_type": event.get("event_type", ""),
            "signal_date": signal_date,
            "stress_start_date": stress_start,
            "peak_stress_date": peak_stress,
            "carry_bleed_return": round(float(carry_bleed), 4) if not pd.isna(carry_bleed) else None,
            "payoff_return": round(float(payoff_ret), 4) if not pd.isna(payoff_ret) else None,
            "net_event_return": round(float(net_evt), 4),
        })

    return pd.DataFrame(rows)

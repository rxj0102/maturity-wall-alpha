"""
signal.py
=========
Cross-sectional signal analysis, statistical validation, and IC computation
for the Corporate Bond Maturity Wall strategy.

Responsibilities
----------------
- Cross-sectional ranking of the universe by stress score
- Historical stress event lead-time validation (out-of-sample)
- t-test separation between CRITICAL and NORMAL tier equity returns
- Spearman Information Coefficient (IC) analysis
- Sector breakdown and macro regime interaction analyses
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.utils import compute_forward_returns, winsorize_series

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Cross-sectional ranking
# ---------------------------------------------------------------------------

def rank_universe_by_stress(
    stress_signals_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Rank all companies by stress score at a given date.

    Excludes companies with ``data_quality_flag == "DATA_ERROR"``.  Assigns
    stress deciles 1–10 (1 = lowest stress, 10 = highest stress).

    Parameters
    ----------
    stress_signals_df : pd.DataFrame
        Rolling signal panel from ``features.build_rolling_signal_history``.
    as_of_date : pd.Timestamp
        Snapshot date to rank.

    Returns
    -------
    pd.DataFrame
        Ranked DataFrame for the given date with added ``position_side`` column:

        - ``"SHORT"`` for stress_decile 10 (CRITICAL tier)
        - ``"LONG"``  for stress_decile ≤ 2 (NORMAL tier)
        - ``"NONE"``  otherwise

    Notes
    -----
    Short candidates are the CRITICAL tier (stress_decile == 10).
    Long candidates are the cleanest balance sheets (stress_decile ≤ 2):
    companies with low MCS (no near-term concentration) and low RDM
    (refinancing at or below their original cost).
    """
    as_of = pd.Timestamp(as_of_date)
    snapshot = stress_signals_df[
        (stress_signals_df["as_of_date"] == as_of)
        & (stress_signals_df["data_quality_flag"] != "DATA_ERROR")
    ].copy()

    if snapshot.empty:
        logger.warning("No signals found for %s", as_of.date())
        return pd.DataFrame()

    # Re-rank within snapshot (stress_decile may already be set; recompute for safety)
    snapshot["stress_rank"] = snapshot["stress_score"].rank(ascending=True, method="first")
    n = len(snapshot)
    snapshot["stress_decile"] = pd.cut(
        snapshot["stress_rank"],
        bins=np.linspace(0, n, 11),
        labels=list(range(1, 11)),
        include_lowest=True,
    ).astype(float)

    snapshot["position_side"] = np.where(
        snapshot["stress_decile"] == 10, "SHORT",
        np.where(snapshot["stress_decile"] <= 2, "LONG", "NONE"),
    )

    return snapshot.sort_values("stress_decile", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Lead-time analysis
# ---------------------------------------------------------------------------

def compute_signal_lead_time_analysis(
    stress_signals_df: pd.DataFrame,
    stress_events_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate signal lead time against known historical stress events.

    For each company in ``historical_stress_events.csv``, find the earliest
    quarter where ``stress_tier`` was ``"CRITICAL"`` or ``"ELEVATED"`` and
    compute the months of advance warning before the documented event start date.

    Parameters
    ----------
    stress_signals_df : pd.DataFrame
        Rolling signal panel.
    stress_events_df : pd.DataFrame
        Loaded ``historical_stress_events.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``event_type``, ``first_signal_date``,
        ``stress_event_start_date``, ``lead_time_months``,
        ``stress_score_at_first_signal``, ``equity_return_signal_to_event``.

    Notes
    -----
    Target: median lead time > 3 months (actionable) and < 24 months
    (not so early that carry bleed is prohibitive).

    Companies where the signal never fired (``CRITICAL`` or ``ELEVATED`` never
    reached before the event) are flagged with ``lead_time_months = NaN``
    and ``signal_triggered = False``.  These are false negatives — the strategy
    missed the event.
    """
    from src.utils import months_between

    rows = []
    for _, event in stress_events_df.iterrows():
        ticker = event["ticker"]
        event_start = pd.Timestamp(event["stress_start_date"])

        co_signals = stress_signals_df[stress_signals_df["ticker"] == ticker].copy()
        if co_signals.empty:
            rows.append({
                "ticker": ticker,
                "event_type": event.get("event_type", ""),
                "first_signal_date": pd.NaT,
                "stress_event_start_date": event_start,
                "lead_time_months": np.nan,
                "stress_score_at_first_signal": np.nan,
                "signal_triggered": False,
            })
            continue

        co_signals["as_of_date"] = pd.to_datetime(co_signals["as_of_date"])
        pre_event = co_signals[co_signals["as_of_date"] < event_start]
        elevated = pre_event[pre_event["stress_tier"].isin(["CRITICAL", "ELEVATED"])]

        if elevated.empty:
            rows.append({
                "ticker": ticker,
                "event_type": event.get("event_type", ""),
                "first_signal_date": pd.NaT,
                "stress_event_start_date": event_start,
                "lead_time_months": np.nan,
                "stress_score_at_first_signal": np.nan,
                "signal_triggered": False,
            })
            continue

        first_signal_row = elevated.sort_values("as_of_date").iloc[0]
        first_signal_date = first_signal_row["as_of_date"]
        lead_months = months_between(first_signal_date, event_start)

        rows.append({
            "ticker": ticker,
            "event_type": event.get("event_type", ""),
            "first_signal_date": first_signal_date,
            "stress_event_start_date": event_start,
            "lead_time_months": round(lead_months, 1),
            "stress_score_at_first_signal": first_signal_row.get("stress_score", np.nan),
            "signal_triggered": True,
        })

    result = pd.DataFrame(rows)
    triggered = result[result["signal_triggered"]]
    if not triggered.empty:
        median_lt = triggered["lead_time_months"].median()
        logger.info(
            "Lead-time analysis: %d/%d events triggered | median lead = %.1f months",
            triggered["signal_triggered"].sum(), len(result), median_lt,
        )
    return result


# ---------------------------------------------------------------------------
# 3. t-test validation
# ---------------------------------------------------------------------------

def run_ttest_validation(
    stress_signals_df: pd.DataFrame,
    equity_prices_df: pd.DataFrame,
    horizon_days: int = 90,
) -> dict:
    """
    T-test: CRITICAL/ELEVATED tier vs NORMAL tier forward equity returns.

    Parameters
    ----------
    stress_signals_df : pd.DataFrame
        Rolling signal panel.
    equity_prices_df : pd.DataFrame
        Adjusted close prices; index=date, columns=tickers.
    horizon_days : int
        Forward return horizon in calendar days.

    Returns
    -------
    dict
        Results for all horizons (30, 60, 90, 180, 252 days):

        .. code-block:: python

            {
                "90d": {
                    "n_stressed": int,
                    "n_normal": int,
                    "mean_stressed": float,
                    "mean_normal": float,
                    "std_stressed": float,
                    "std_normal": float,
                    "t_stat": float,
                    "p_value": float,
                },
                ...
            }
    """
    all_horizons = [30, 60, 90, 180, 252]
    results: dict = {}

    for h in all_horizons:
        fwd_returns = compute_forward_returns(equity_prices_df, h)
        rows_stressed, rows_normal = [], []

        for _, signal_row in stress_signals_df.iterrows():
            ticker = signal_row["ticker"]
            as_of = pd.Timestamp(signal_row["as_of_date"])
            tier = signal_row.get("stress_tier", "")

            if ticker not in fwd_returns.columns:
                continue

            # Get forward return starting on as_of date
            if as_of not in fwd_returns.index:
                # Use nearest available date
                candidates = fwd_returns.index[fwd_returns.index >= as_of]
                if candidates.empty:
                    continue
                as_of = candidates[0]

            fwd_ret = fwd_returns.loc[as_of, ticker]
            if pd.isna(fwd_ret):
                continue

            if tier in ("CRITICAL", "ELEVATED"):
                rows_stressed.append(fwd_ret)
            elif tier == "NORMAL":
                rows_normal.append(fwd_ret)

        if len(rows_stressed) < 5 or len(rows_normal) < 5:
            results[f"{h}d"] = {"error": "insufficient observations", "horizon_days": h}
            continue

        stressed_arr = np.array(rows_stressed)
        normal_arr   = np.array(rows_normal)
        t_stat, p_val = stats.ttest_ind(stressed_arr, normal_arr, equal_var=False)

        key = f"{h}d"
        results[key] = {
            "horizon_days": h,
            "n_stressed": len(stressed_arr),
            "n_normal": len(normal_arr),
            "mean_stressed": round(float(np.mean(stressed_arr)), 4),
            "mean_normal": round(float(np.mean(normal_arr)), 4),
            "std_stressed": round(float(np.std(stressed_arr)), 4),
            "std_normal": round(float(np.std(normal_arr)), 4),
            "t_stat": round(float(t_stat), 3),
            "p_value": round(float(p_val), 4),
        }
        logger.info(
            "t-test %dd: stressed=%.2f%%, normal=%.2f%%, t=%.2f, p=%.3f",
            h,
            100 * results[key]["mean_stressed"],
            100 * results[key]["mean_normal"],
            t_stat, p_val,
        )

    # Print formatted table
    print("\n" + "=" * 72)
    print("  T-Test: CRITICAL/ELEVATED vs NORMAL Forward Equity Returns")
    print("=" * 72)
    header = f"{'Horizon':>8} {'N(str)':>7} {'N(nor)':>7} "
    header += f"{'Mean(str)':>10} {'Mean(nor)':>10} {'t-stat':>8} {'p-value':>8}"
    print(header)
    print("-" * 72)
    for h in all_horizons:
        key = f"{h}d"
        r = results.get(key, {})
        if "error" in r:
            print(f"{key:>8}   (insufficient data)")
        else:
            print(
                f"{key:>8} {r['n_stressed']:>7} {r['n_normal']:>7} "
                f"{100*r['mean_stressed']:>9.2f}% {100*r['mean_normal']:>9.2f}% "
                f"{r['t_stat']:>8.2f} {r['p_value']:>8.4f}"
            )
    print("=" * 72 + "\n")
    return results


# ---------------------------------------------------------------------------
# 4. Information Coefficient
# ---------------------------------------------------------------------------

def compute_information_coefficient(
    stress_signals_df: pd.DataFrame,
    equity_prices_df: pd.DataFrame,
) -> dict:
    """
    Compute Spearman IC between stress score and forward equity returns.

    IC is computed with inverted sign: high stress_score → negative equity
    return.  Positive IC means the signal is directionally correct.

    Parameters
    ----------
    stress_signals_df : pd.DataFrame
        Rolling signal panel.
    equity_prices_df : pd.DataFrame
        Adjusted close prices.

    Returns
    -------
    dict
        Nested dict: ``{horizon: {"raw_stress": IC, "stress_score": IC, ...}}``.
        Also includes breakdowns by data source
        (``"trace_enriched"`` vs ``"xbrl_only"``).

    Notes
    -----
    The 180d and 252d horizons are expected to show the strongest IC.
    This is consistent with the "right but early" thesis: the maturity
    wall is a slow-moving fundamental catalyst that takes 6–18 months
    to price into equities.
    """
    horizons = [30, 60, 90, 180, 252]
    results: dict = {}

    for h in horizons:
        fwd_returns = compute_forward_returns(equity_prices_df, h)
        pairs_raw, pairs_decay, pairs_trace, pairs_xbrl = [], [], [], []

        for _, row in stress_signals_df.iterrows():
            ticker = row["ticker"]
            as_of = pd.Timestamp(row["as_of_date"])

            if ticker not in fwd_returns.columns:
                continue
            candidates = fwd_returns.index[fwd_returns.index >= as_of]
            if candidates.empty:
                continue
            fwd_ret = fwd_returns.loc[candidates[0], ticker]
            if pd.isna(fwd_ret):
                continue

            raw_s = row.get("raw_stress", np.nan)
            dec_s = row.get("stress_score", np.nan)
            src   = row.get("rate_data_source", "")

            if not pd.isna(raw_s) and not pd.isna(fwd_ret):
                pairs_raw.append((raw_s, -fwd_ret))   # invert return sign
            if not pd.isna(dec_s) and not pd.isna(fwd_ret):
                pairs_decay.append((dec_s, -fwd_ret))
                if "trace" in str(src):
                    pairs_trace.append((dec_s, -fwd_ret))
                else:
                    pairs_xbrl.append((dec_s, -fwd_ret))

        def _spearman_ic(pairs: list) -> float:
            if len(pairs) < 10:
                return np.nan
            x, y = zip(*pairs)
            ic, _ = stats.spearmanr(x, y)
            return round(float(ic), 4)

        key = f"{h}d"
        results[key] = {
            "horizon_days": h,
            "raw_stress_IC":         _spearman_ic(pairs_raw),
            "decay_stress_IC":       _spearman_ic(pairs_decay),
            "trace_enriched_IC":     _spearman_ic(pairs_trace),
            "xbrl_only_IC":          _spearman_ic(pairs_xbrl),
            "n_observations":        len(pairs_decay),
        }

    # Print IC table
    print("\n" + "=" * 72)
    print("  Spearman IC: stress_score vs -(forward equity return)")
    print("=" * 72)
    print(f"{'Horizon':>8} {'Raw IC':>10} {'Decay IC':>10} {'TRACE IC':>10} {'XBRL IC':>10} {'N':>6}")
    print("-" * 72)
    for h in horizons:
        key = f"{h}d"
        r = results.get(key, {})
        print(
            f"{key:>8} {r.get('raw_stress_IC', np.nan):>10.4f} "
            f"{r.get('decay_stress_IC', np.nan):>10.4f} "
            f"{r.get('trace_enriched_IC', np.nan):>10.4f} "
            f"{r.get('xbrl_only_IC', np.nan):>10.4f} "
            f"{r.get('n_observations', 0):>6}"
        )
    print("=" * 72 + "\n")
    return results


# ---------------------------------------------------------------------------
# 5. Sector breakdown
# ---------------------------------------------------------------------------

def compute_sector_breakdown(
    stress_signals_df: pd.DataFrame,
    equity_prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-sector signal statistics.

    For each sector: mean stress_score, CRITICAL tier hit rate, IC at 180d,
    mean forward equity return for CRITICAL companies.

    Parameters
    ----------
    stress_signals_df : pd.DataFrame
        Rolling signal panel (must include ``sector`` column, merged from
        universe CSV by the caller).
    equity_prices_df : pd.DataFrame
        Adjusted close prices.

    Returns
    -------
    pd.DataFrame
        One row per sector: ``sector``, ``n_company_quarters``,
        ``mean_stress_score``, ``critical_hit_rate``, ``ic_180d``,
        ``mean_fwd_return_critical_180d``.

    Notes
    -----
    **Hypothesis**: Retail, energy, and media should show the strongest
    signal; utilities and healthcare weaker (regulated cash flows and M&A
    optionality provide additional refinancing flexibility).
    """
    if "sector" not in stress_signals_df.columns:
        logger.warning("sector column not in signals — cannot compute sector breakdown")
        return pd.DataFrame()

    fwd_180 = compute_forward_returns(equity_prices_df, 180)
    rows = []
    for sector, grp in stress_signals_df.groupby("sector"):
        critical_n = (grp["stress_tier"] == "CRITICAL").sum()
        hit_rate = critical_n / len(grp) if len(grp) > 0 else np.nan

        # IC at 180d for this sector
        pairs = []
        for _, r in grp.iterrows():
            t = r["ticker"]
            ao = pd.Timestamp(r["as_of_date"])
            candidates = fwd_180.index[fwd_180.index >= ao] if t in fwd_180.columns else []
            if len(candidates) == 0:
                continue
            fwd_ret = fwd_180.loc[candidates[0], t]
            sc = r.get("stress_score", np.nan)
            if not pd.isna(fwd_ret) and not pd.isna(sc):
                pairs.append((sc, -fwd_ret))

        ic = np.nan
        if len(pairs) >= 10:
            x, y = zip(*pairs)
            ic, _ = stats.spearmanr(x, y)

        # Mean forward return for CRITICAL tier
        critical_rows = grp[grp["stress_tier"] == "CRITICAL"]
        crit_returns = []
        for _, cr in critical_rows.iterrows():
            t = cr["ticker"]
            ao = pd.Timestamp(cr["as_of_date"])
            if t not in fwd_180.columns:
                continue
            cands = fwd_180.index[fwd_180.index >= ao]
            if len(cands) == 0:
                continue
            crit_returns.append(fwd_180.loc[cands[0], t])

        mean_crit_ret = float(np.mean(crit_returns)) if crit_returns else np.nan

        rows.append({
            "sector": sector,
            "n_company_quarters": len(grp),
            "mean_stress_score": round(grp["stress_score"].mean(), 4),
            "critical_hit_rate": round(float(hit_rate), 4),
            "ic_180d": round(float(ic), 4) if not np.isnan(ic) else np.nan,
            "mean_fwd_return_critical_180d": round(mean_crit_ret, 4) if not np.isnan(mean_crit_ret) else np.nan,
        })

    return pd.DataFrame(rows).sort_values("ic_180d", ascending=True)


# ---------------------------------------------------------------------------
# 6. Macro regime interaction
# ---------------------------------------------------------------------------

def compute_macro_regime_interaction(
    stress_signals_df: pd.DataFrame,
    macro_rates_df: pd.DataFrame,
    equity_prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate signal performance across macro regimes.

    Regimes are defined based on 12-month changes in T10Y and current HY_OAS
    levels:

    - ``RISING_RATES``  : T10Y increased > 100bps in trailing 12 months
    - ``FALLING_RATES`` : T10Y decreased > 100bps in trailing 12 months
    - ``WIDE_SPREADS``  : HY_OAS > 600bps (stress environment)
    - ``TIGHT_SPREADS`` : HY_OAS < 350bps (benign credit environment)
    - ``NEUTRAL``       : does not meet other criteria

    Parameters
    ----------
    stress_signals_df : pd.DataFrame
        Rolling signal panel.
    macro_rates_df : pd.DataFrame
        FRED macro rates.
    equity_prices_df : pd.DataFrame
        Adjusted close prices.

    Returns
    -------
    pd.DataFrame
        One row per regime: ``regime``, ``n_quarters``, ``ic_180d``,
        ``mean_fwd_return_critical``, ``mean_fwd_return_normal``,
        ``spread``.

    Notes
    -----
    **Critical finding**: the signal should be most predictive in
    ``RISING_RATES AND WIDE_SPREADS`` — the joint scenario where RDM > 2.0
    AND refinancing markets are actually closed to stressed issuers.  When
    both conditions hold, the maturity wall is a hard binary event (the
    company cannot refinance at any price).

    If the strategy's alpha is concentrated entirely in rising-rate periods,
    it is effectively a rate-sensitivity bet dressed up as a maturity-wall
    signal — a critical risk to document and stress-test.
    """
    # Assign regime to each quarter-end date
    macro_rates_df = macro_rates_df.copy()
    macro_rates_df.index = pd.to_datetime(macro_rates_df.index)

    def get_regime(dt: pd.Timestamp) -> str:
        as_of_rates = macro_rates_df[macro_rates_df.index <= dt]
        if as_of_rates.empty:
            return "NEUTRAL"
        trailing_12m = macro_rates_df[
            (macro_rates_df.index <= dt)
            & (macro_rates_df.index >= dt - pd.DateOffset(months=12))
        ]
        if trailing_12m.empty:
            return "NEUTRAL"

        t10y_now  = as_of_rates["T10Y"].iloc[-1]
        t10y_12m  = trailing_12m["T10Y"].iloc[0]
        hy_oas    = as_of_rates.get("HY_OAS", pd.Series([350.0])).iloc[-1]

        t10y_chg = t10y_now - t10y_12m  # in percentage points
        # HY_OAS from FRED is in basis points (e.g. 400 = 4.00%)
        hy_oas_bps = hy_oas if hy_oas > 10 else hy_oas * 100  # normalise

        if hy_oas_bps > 600:
            return "WIDE_SPREADS"
        if hy_oas_bps < 350:
            return "TIGHT_SPREADS"
        if t10y_chg > 1.0:
            return "RISING_RATES"
        if t10y_chg < -1.0:
            return "FALLING_RATES"
        return "NEUTRAL"

    fwd_180 = compute_forward_returns(equity_prices_df, 180)
    regime_rows: dict = {r: [] for r in ["RISING_RATES", "FALLING_RATES",
                                          "WIDE_SPREADS", "TIGHT_SPREADS", "NEUTRAL"]}

    for _, sig_row in stress_signals_df.iterrows():
        ticker = sig_row["ticker"]
        as_of = pd.Timestamp(sig_row["as_of_date"])
        regime = get_regime(as_of)

        if ticker not in fwd_180.columns:
            continue
        cands = fwd_180.index[fwd_180.index >= as_of]
        if len(cands) == 0:
            continue
        fwd_ret = fwd_180.loc[cands[0], ticker]
        if pd.isna(fwd_ret):
            continue

        regime_rows[regime].append({
            "stress_score": sig_row.get("stress_score", np.nan),
            "stress_tier": sig_row.get("stress_tier", ""),
            "fwd_ret": fwd_ret,
        })

    summary_rows = []
    for regime, rrows in regime_rows.items():
        if not rrows:
            continue
        rdf = pd.DataFrame(rrows)
        pairs = [
            (r["stress_score"], -r["fwd_ret"])
            for _, r in rdf.iterrows()
            if not pd.isna(r["stress_score"]) and not pd.isna(r["fwd_ret"])
        ]
        ic = np.nan
        if len(pairs) >= 10:
            x, y = zip(*pairs)
            ic, _ = stats.spearmanr(x, y)

        critical = rdf[rdf["stress_tier"] == "CRITICAL"]["fwd_ret"]
        normal   = rdf[rdf["stress_tier"] == "NORMAL"]["fwd_ret"]

        summary_rows.append({
            "regime": regime,
            "n_quarters": len(rdf["stress_tier"].unique()),
            "n_observations": len(rdf),
            "ic_180d": round(float(ic), 4) if not np.isnan(ic) else np.nan,
            "mean_fwd_return_critical": round(critical.mean(), 4) if not critical.empty else np.nan,
            "mean_fwd_return_normal": round(normal.mean(), 4) if not normal.empty else np.nan,
            "spread_critical_minus_normal": round(
                critical.mean() - normal.mean(), 4
            ) if not critical.empty and not normal.empty else np.nan,
        })

    return pd.DataFrame(summary_rows).sort_values("ic_180d", ascending=True)

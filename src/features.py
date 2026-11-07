"""
features.py
===========
Signal feature engineering for the Corporate Bond Maturity Wall strategy.

This module computes the three building blocks of the composite stress signal
and assembles them into a single, cross-sectionally ranked panel:

1. **MCS** — Maturity Concentration Score: fraction of total forward debt
   maturing in the single peak quarter.

2. **RDM** — Refinancing Difficulty Multiplier: current market refinancing
   rate divided by the yield-at-issuance proxy for the maturing instrument.

3. **decay_weight** — Exponential time-decay that sharpens the signal as the
   peak maturity date approaches and suppresses noise for distant maturities.

Composite stress formula (canonical definition — implemented here only):

.. math::

    \\text{stress\\_score} = \\text{MCS} \\times \\text{RDM} \\times \\text{decay\\_weight}(\\text{months\\_to\\_peak})

The ``raw_stress`` variant (from the original strategy spec) uses
:math:`1 / \\text{months\\_to\\_peak}` directly, which has a singularity at
``months = 0``.  Both are computed; comparison is in the notebook.

**IMPORTANT:** The composite stress formula is defined **once** in this
module (``compute_composite_stress_signal``) and imported everywhere else.
No duplication.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.utils import (
    get_quarter_label,
    get_quarter_end,
    get_all_quarter_ends,
    months_between,
    winsorize_series,
    lag_dataframe,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Maturity Concentration Score (MCS)
# ---------------------------------------------------------------------------

def compute_maturity_concentration_score(
    quarterly_schedule_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute MCS for each company as of a given snapshot date.

    MCS = peak_quarter_debt / total_debt_outstanding over the forward
    ``config.LOOKFORWARD_QUARTERS`` quarters.

    Parameters
    ----------
    quarterly_schedule_df : pd.DataFrame
        Output of ``xbrl_parser.convert_annual_to_quarterly_schedule``,
        containing columns ``cik``, ``ticker``, ``filing_date``,
        ``quarter_end_date``, ``debt_maturing_mm``.
    as_of_date : pd.Timestamp
        Snapshot date for the signal computation.  Only filings available
        before ``as_of_date - SIGNAL_UPDATE_LAG_DAYS`` are used.

    Returns
    -------
    pd.DataFrame
        One row per company with columns: ``ticker``, ``as_of_date``,
        ``MCS``, ``peak_quarter_label``, ``peak_quarter_debt_mm``,
        ``total_debt_outstanding_mm``, ``months_to_peak_maturity``,
        ``mcs_bucket``.
    """
    as_of = pd.Timestamp(as_of_date)
    lag = pd.tseries.offsets.BusinessDay(config.SIGNAL_UPDATE_LAG_DAYS)
    data_cutoff = as_of - lag

    # Forward window: next LOOKFORWARD_QUARTERS quarters
    forward_quarters = pd.date_range(
        start=as_of, periods=config.LOOKFORWARD_QUARTERS + 1, freq="QE"
    )
    window_start = forward_quarters[0]
    window_end   = forward_quarters[-1]

    rows = []
    for (cik, ticker), grp in quarterly_schedule_df.groupby(["cik", "ticker"], sort=False):
        # Use most recent filing available as of cutoff
        available = grp[grp["filing_date"] <= data_cutoff]
        if available.empty:
            continue
        latest_filing = available["filing_date"].max()
        latest_grp = available[available["filing_date"] == latest_filing]

        # Filter to forward window
        forward = latest_grp[
            (latest_grp["quarter_end_date"] >= window_start)
            & (latest_grp["quarter_end_date"] <= window_end)
        ].copy()

        if forward.empty:
            continue

        # Aggregate by quarter (sum within quarter if multiple rows)
        qsum = forward.groupby("quarter_label")["debt_maturing_mm"].sum()
        total_debt = qsum.sum()

        if total_debt <= 0:
            rows.append({
                "ticker": ticker, "cik": cik, "as_of_date": as_of,
                "MCS": 0.0, "peak_quarter_label": None,
                "peak_quarter_debt_mm": 0.0,
                "total_debt_outstanding_mm": 0.0,
                "months_to_peak_maturity": np.nan,
                "mcs_bucket": "LOW",
                "data_error": True,
            })
            continue

        peak_label = qsum.idxmax()
        peak_debt  = qsum.max()
        mcs = peak_debt / total_debt

        # Months to peak quarter
        try:
            peak_end = get_quarter_end(peak_label)
            # months to beginning of peak quarter
            peak_start = peak_end - pd.offsets.MonthEnd(2)  # ~first day of quarter
            m_to_peak = months_between(as_of, peak_start)
        except Exception:
            m_to_peak = np.nan

        mcs_bucket = (
            "EXTREME" if mcs > config.MCS_EXTREME_THRESHOLD else
            "HIGH"    if mcs > config.MCS_HIGH_THRESHOLD    else
            "MEDIUM"  if mcs > 0.20                         else
            "LOW"
        )

        rows.append({
            "ticker": ticker,
            "cik": cik,
            "as_of_date": as_of,
            "MCS": round(mcs, 4),
            "peak_quarter_label": peak_label,
            "peak_quarter_debt_mm": round(peak_debt, 1),
            "total_debt_outstanding_mm": round(total_debt, 1),
            "months_to_peak_maturity": round(m_to_peak, 2) if not np.isnan(m_to_peak) else np.nan,
            "mcs_bucket": mcs_bucket,
            "data_error": False,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Months to peak maturity (proximity classification)
# ---------------------------------------------------------------------------

def compute_months_to_peak_maturity(
    mcs_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Enrich MCS DataFrame with proximity classification.

    Classifies companies by how far away their peak maturity quarter is:

    - ``"NEAR"``    : ≤ 6 months  → signal active, maximum urgency
    - ``"MEDIUM"``  : ≤ 18 months → signal active, watch
    - ``"WATCH"``   : ≤ 36 months → monitoring only
    - ``"DISTANT"`` : > 36 months → background

    Parameters
    ----------
    mcs_df : pd.DataFrame
        Output of ``compute_maturity_concentration_score``.
    as_of_date : pd.Timestamp
        Snapshot date (for logging context).

    Returns
    -------
    pd.DataFrame
        ``mcs_df`` with added ``proximity_label`` column.
    """
    def classify(m: float) -> str:
        if pd.isna(m):
            return "UNKNOWN"
        if m <= config.MONTHS_NEAR_THRESHOLD:
            return "NEAR"
        if m <= config.MONTHS_MEDIUM_THRESHOLD:
            return "MEDIUM"
        if m <= 36:
            return "WATCH"
        return "DISTANT"

    df = mcs_df.copy()
    df["proximity_label"] = df["months_to_peak_maturity"].apply(classify)
    return df


# ---------------------------------------------------------------------------
# 3. Refinancing Difficulty Multiplier (RDM)
# ---------------------------------------------------------------------------

def compute_refinancing_difficulty_multiplier(
    mcs_df: pd.DataFrame,
    bond_stack_df: pd.DataFrame,
    macro_rates_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    universe_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute RDM for each company.

    .. math::

        \\text{RDM} = \\frac{\\text{current\\_refinancing\\_rate}}
                           {\\text{yield\\_at\\_issuance\\_proxy}}

    ``yield_at_issuance_proxy`` hierarchy:

    1. Coupon of the specific TRACE bond maturing in the peak quarter
       (most accurate — bond-specific).
    2. WAC from all outstanding fixed-rate bonds (less accurate — blends
       different rate vintages).
    3. Fallback: 5.0% (sector-average for HY issuers) with DATA_ERROR flag.

    ``current_refinancing_rate``:

    - HY issuers:  T10Y + HY_OAS  (from FRED)
    - IG issuers:  T10Y + IG_OAS  (from FRED)

    Parameters
    ----------
    mcs_df : pd.DataFrame
        Output of ``compute_months_to_peak_maturity``.
    bond_stack_df : pd.DataFrame
        TRACE bond stack for all companies (concatenated).  May be empty
        if TRACE data is unavailable — falls back to WAC proxy.
    macro_rates_df : pd.DataFrame
        Wide FRED data: columns T10Y, HY_OAS, IG_OAS.
    as_of_date : pd.Timestamp
        Snapshot date.
    universe_df : pd.DataFrame, optional
        Company universe CSV; used to determine debt_category for each ticker
        (HY vs IG spread selection).

    Returns
    -------
    pd.DataFrame
        ``mcs_df`` enriched with: ``RDM``, ``yield_at_issuance_proxy``,
        ``current_refinancing_rate``, ``rate_data_source``, ``rdm_bucket``,
        ``data_quality_flag``.
    """
    as_of = pd.Timestamp(as_of_date)

    # Macro rates as of as_of_date (most recent available)
    macro_snapshot = macro_rates_df[macro_rates_df.index <= as_of]
    if macro_snapshot.empty:
        logger.warning("No macro rate data available as of %s — using defaults", as_of)
        t10y_pct   = 4.5
        hy_oas_pct = 4.0
        ig_oas_pct = 1.2
    else:
        latest_macro = macro_snapshot.iloc[-1]
        t10y_pct   = float(latest_macro.get("T10Y", 4.5))
        hy_oas_pct = float(latest_macro.get("HY_OAS", 4.0) / 100)  # OAS in bps → pct
        ig_oas_pct = float(latest_macro.get("IG_OAS", 1.2) / 100)

    # Build CUSIP → coupon lookup for peak-quarter bond matching
    peak_coupon_map: dict = {}
    if bond_stack_df is not None and not bond_stack_df.empty:
        for (ticker, ticker_grp) in bond_stack_df.groupby("ticker"):
            for _, co in mcs_df[mcs_df["ticker"] == ticker].iterrows():
                peak_lbl = co.get("peak_quarter_label")
                if not peak_lbl:
                    continue
                try:
                    peak_start = get_quarter_end(peak_lbl) - pd.DateOffset(months=3)
                    peak_end   = get_quarter_end(peak_lbl)
                except Exception:
                    continue
                maturing_in_peak = ticker_grp[
                    (ticker_grp["maturity_date"] >= peak_start)
                    & (ticker_grp["maturity_date"] <= peak_end)
                    & (~ticker_grp["is_matured"].fillna(True))
                    & (ticker_grp["coupon_type"] == "fixed")
                ]
                if not maturing_in_peak.empty:
                    # Use face-value-weighted coupon of maturing bonds
                    w_coup = (
                        (maturing_in_peak["coupon_rate"] * maturing_in_peak["outstanding_face_mm"]).sum()
                        / maturing_in_peak["outstanding_face_mm"].sum()
                    )
                    peak_coupon_map[ticker] = w_coup

    # Universe debt_category lookup
    cat_map: dict = {}
    if universe_df is not None and "debt_category" in universe_df.columns:
        cat_map = universe_df.set_index("ticker")["debt_category"].to_dict()

    # WAC lookup per ticker
    from src.trace_parser import compute_weighted_avg_coupon
    wac_map: dict = {}
    if bond_stack_df is not None and not bond_stack_df.empty:
        for ticker, grp in bond_stack_df.groupby("ticker"):
            wac_map[ticker] = compute_weighted_avg_coupon(grp)

    rows = []
    for _, row in mcs_df.iterrows():
        ticker = row["ticker"]
        data_quality = "CLEAN"

        # Determine debt category → spread selection
        cat = cat_map.get(ticker, "high_yield")
        is_hy = "investment_grade" not in cat.lower()
        current_refi_rate = t10y_pct + (hy_oas_pct if is_hy else ig_oas_pct)

        # yield_at_issuance proxy: priority order
        if ticker in peak_coupon_map and not pd.isna(peak_coupon_map[ticker]):
            yai = peak_coupon_map[ticker]
            rate_source = "trace_bond_specific"
        elif ticker in wac_map and not pd.isna(wac_map.get(ticker)):
            yai = wac_map[ticker]
            rate_source = "wac_proxy"
            data_quality = "WAC_PROXY_USED"
        else:
            yai = 5.0  # sector-average fallback for HY
            rate_source = "fallback_default"
            data_quality = "DATA_ERROR"

        # RDM computation
        if yai == 0.0:
            rdm = 1.0
            data_quality = "DATA_ERROR"
        else:
            rdm = current_refi_rate / yai

        # Floating-rate bonds are less meaningful for RDM
        if bond_stack_df is not None and not bond_stack_df.empty:
            ticker_bonds = bond_stack_df[bond_stack_df["ticker"] == ticker]
            if not ticker_bonds.empty:
                float_frac = (
                    ticker_bonds[ticker_bonds["coupon_type"] == "floating"]["outstanding_face_mm"].sum()
                    / ticker_bonds["outstanding_face_mm"].sum().clip(lower=1e-6)
                )
                if float_frac > 0.5:
                    data_quality = "FLOATING_RATE_DOMINANT"

        rdm_bucket = (
            "SEVERE"   if rdm > 2.0 else
            "HIGH"     if rdm > config.RDM_STRESS_THRESHOLD else
            "MODERATE" if rdm > 1.0 else
            "BENIGN"
        )

        row_dict = row.to_dict()
        row_dict.update({
            "RDM": round(rdm, 4),
            "yield_at_issuance_proxy": round(yai, 4),
            "current_refinancing_rate": round(current_refi_rate, 4),
            "rate_data_source": rate_source,
            "rdm_bucket": rdm_bucket,
            "data_quality_flag": data_quality,
        })
        rows.append(row_dict)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Decay weight
# ---------------------------------------------------------------------------

def compute_signal_decay_weight(
    months_to_peak: float,
    halflife: float = float(config.SIGNAL_DECAY_HALFLIFE),
) -> float:
    """
    Exponential decay weight that sharpens the signal as maturity approaches.

    .. math::

        w(m) = \\exp\\!\\left(
            \\ln(2) \\cdot \\frac{M_{\\mathrm{med}} - m}{h}
        \\right)

    where :math:`M_{\\mathrm{med}}` = ``MONTHS_MEDIUM_THRESHOLD`` (18),
    :math:`m` = ``months_to_peak``, :math:`h` = ``halflife``.

    Reference values (halflife = 12 months, :math:`M_{med}` = 18):

    ============  ===========  ===========
    months_to_peak  weight    interpretation
    ============  ===========  ===========
    48            ≈ 0.18       very distant — negligible weight
    36            ≈ 0.35       distant — low weight
    24            ≈ 0.71       watch
    18            ≈ 1.00       medium boundary — neutral
    12            ≈ 1.41       elevated
     6            ≈ 2.00       near — doubled weight
     0            ≈ 2.83       imminent — maximum weight (2^1.5)
    ============  ===========  ===========

    Note: values ≈ 4× at ``months=0`` and ≈ 0.25 at ``months=36``
    require ``halflife=9`` months.  The config default of 12 months
    produces the values shown above, which are more conservative —
    reducing allocation amplification at very short horizons.

    Parameters
    ----------
    months_to_peak : float
        Months from signal date to start of peak maturity quarter.
    halflife : float
        Half-life in months.  Default is ``config.SIGNAL_DECAY_HALFLIFE``.

    Returns
    -------
    float
        Decay weight (> 0).

    Notes
    -----
    This formulation directly addresses the *carry-bleed* problem described
    in the strategy spec: "short a stock that goes up 30% before the thesis
    plays out."  By down-weighting early (distant) signals and amplifying
    near-term (imminent) signals, the decay function reduces the position
    size during the early carry-bleed phase and concentrates exposure in the
    window where the short is most likely to pay off.  See the notebook
    Section 6 for a comparison of ``raw_stress`` (no decay) vs
    ``stress_score`` (decay-adjusted) backtest Sharpe ratios.
    """
    if pd.isna(months_to_peak) or months_to_peak < 0:
        months_to_peak = 0.0
    exponent = math.log(2) * (config.MONTHS_MEDIUM_THRESHOLD - months_to_peak) / halflife
    return math.exp(exponent)


# ---------------------------------------------------------------------------
# 5. Composite stress signal (THE canonical formula)
# ---------------------------------------------------------------------------

def compute_composite_stress_signal(
    mcs_df: pd.DataFrame,
    rdm_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the composite stress score for each company.

    **This is the single canonical implementation of the stress formula.**
    All downstream modules (``signal.py``, ``backtest.py``) import and call
    this function — the formula is never duplicated.

    Two variants are computed:

    1. ``raw_stress``: uses :math:`1 / \\text{months\\_to\\_peak}` directly
       (the spec formula).  Has a singularity at ``months = 0``; clipped at
       a maximum of 1 / 0.5 = 2.0.

    2. ``stress_score`` (preferred): uses the exponential ``decay_weight``.
       Numerically stable and better-behaved at very short horizons.

    .. math::

        \\text{stress\\_score} = \\text{MCS} \\times \\text{RDM}
                                  \\times w(\\text{months\\_to\\_peak})

    Parameters
    ----------
    mcs_df : pd.DataFrame
        Output of ``compute_refinancing_difficulty_multiplier`` (already
        contains RDM columns merged in).
    rdm_df : pd.DataFrame
        Same reference as mcs_df (RDM already merged); kept as separate
        parameter for API symmetry with the spec.

    Returns
    -------
    pd.DataFrame
        Signal panel: ``ticker``, ``as_of_date``, ``MCS``, ``RDM``,
        ``months_to_peak_maturity``, ``raw_stress``, ``stress_score``,
        ``decay_weight``, ``stress_tier``, ``mcs_bucket``, ``rdm_bucket``,
        ``proximity_label``, ``data_quality_flag``.

    Notes
    -----
    Stress tier assignment uses cross-sectional decile ranking within each
    ``as_of_date`` snapshot:

    - ``"CRITICAL"``  — stress decile 10 (top)
    - ``"ELEVATED"``  — stress deciles 7–9
    - ``"WATCH"``     — stress deciles 4–6
    - ``"NORMAL"``    — stress deciles 1–3 (bottom)
    """
    # rdm_df is expected to already contain all MCS columns (merged upstream)
    # Use rdm_df as the primary df since it has both MCS and RDM columns
    df = rdm_df.copy() if not rdm_df.empty else mcs_df.copy()

    # Merge any missing columns from mcs_df
    if not mcs_df.empty and not rdm_df.empty:
        for col in mcs_df.columns:
            if col not in df.columns:
                df = df.merge(
                    mcs_df[["ticker", "as_of_date", col]],
                    on=["ticker", "as_of_date"],
                    how="left",
                    suffixes=("", "_mcs"),
                )

    # Compute decay weight per row
    df["decay_weight"] = df["months_to_peak_maturity"].apply(
        lambda m: compute_signal_decay_weight(m)
    )

    # raw_stress: 1/months formula (clipped at 2.0 to avoid singularity)
    df["raw_stress"] = (
        df["MCS"] * df["RDM"]
        * df["months_to_peak_maturity"].apply(
            lambda m: 1.0 / max(m, 0.5) if not pd.isna(m) and m > 0 else 2.0
        )
    )

    # stress_score: decay-adjusted (canonical for ranking and backtest)
    df["stress_score"] = df["MCS"] * df["RDM"] * df["decay_weight"]

    # Winsorise to prevent data-error outliers from dominating rankings
    df["stress_score"] = winsorize_series(df["stress_score"])

    # Stress tier assignment: within-snapshot decile ranking
    def assign_tiers(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.copy()
        if len(grp) < 5:
            grp["stress_decile"] = np.nan
            grp["stress_tier"] = "INSUFFICIENT_UNIVERSE"
            return grp
        grp["stress_decile"] = pd.qcut(
            grp["stress_score"].rank(method="first"),
            q=10,
            labels=False,
        ) + 1  # 1-indexed
        grp["stress_tier"] = grp["stress_decile"].apply(
            lambda d: (
                "CRITICAL" if d == 10 else
                "ELEVATED" if d >= 7  else
                "WATCH"    if d >= 4  else
                "NORMAL"
            ) if not pd.isna(d) else "UNKNOWN"
        )
        return grp

    df = df.groupby("as_of_date", group_keys=False).apply(assign_tiers)

    # Override data_quality_flag from PARTIAL coverage
    if "coverage_flag" in df.columns:
        df.loc[df["coverage_flag"] == "PARTIAL", "data_quality_flag"] = (
            df.loc[df["coverage_flag"] == "PARTIAL", "data_quality_flag"]
            .where(df["data_quality_flag"] != "CLEAN", "PARTIAL_XBRL")
        )

    output_cols = [
        "ticker", "cik", "as_of_date",
        "MCS", "RDM", "months_to_peak_maturity",
        "raw_stress", "stress_score", "decay_weight",
        "stress_tier", "stress_decile",
        "mcs_bucket", "rdm_bucket", "proximity_label",
        "peak_quarter_label", "peak_quarter_debt_mm", "total_debt_outstanding_mm",
        "yield_at_issuance_proxy", "current_refinancing_rate", "rate_data_source",
        "data_quality_flag",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    return df[output_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Rolling signal history
# ---------------------------------------------------------------------------

def build_rolling_signal_history(
    universe_df: pd.DataFrame,
    macro_rates_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a point-in-time rolling signal panel from BACKTEST_START to BACKTEST_END.

    Computes the composite stress signal at every quarter-end date, using only
    XBRL filings that were publicly available at each snapshot (strictly
    enforces the ``SIGNAL_UPDATE_LAG_DAYS`` look-ahead guard).

    Parameters
    ----------
    universe_df : pd.DataFrame
        Loaded company universe CSV.
    macro_rates_df : pd.DataFrame
        FRED macro rates; must cover ``BACKTEST_START`` to ``BACKTEST_END``.

    Returns
    -------
    pd.DataFrame
        Long-format panel: all companies × all quarter-end dates.
        Saved to ``data/processed/rolling_stress_signals.parquet``.

    Notes
    -----
    This function is the primary dataset for all downstream analysis.  It
    must be constructed with strict point-in-time discipline:

    - Filing dates are the actual EDGAR accession dates (from XBRL facts JSON),
      not fiscal year-end dates.
    - The ``SIGNAL_UPDATE_LAG_DAYS`` lag is applied to each filing date
      individually, not to the panel as a whole.
    - The ``lag_dataframe`` utility from ``utils.py`` is applied to the panel
      before it enters any backtest function to provide a final look-ahead guard.
    """
    from src.xbrl_parser import build_universe_maturity_schedules
    from src.trace_parser import build_issuer_bond_stack

    # Load or build quarterly schedule
    if config.UNIVERSE_SCHEDULES_PARQUET.exists():
        logger.info("Loading cached universe maturity schedules")
        quarterly_schedule = pd.read_parquet(config.UNIVERSE_SCHEDULES_PARQUET)
    else:
        logger.info("Building universe maturity schedules from scratch")
        quarterly_schedule = build_universe_maturity_schedules(universe_df)

    if quarterly_schedule is None or quarterly_schedule.empty:
        logger.error("No quarterly schedules available — cannot build rolling signals")
        return pd.DataFrame()

    # Ensure date columns are timestamps
    for col in ["filing_date", "quarter_end_date"]:
        if col in quarterly_schedule.columns:
            quarterly_schedule[col] = pd.to_datetime(quarterly_schedule[col], errors="coerce")

    # Load/build bond stacks for TRACE enrichment
    bond_stacks = []
    for _, co in universe_df[universe_df.get("include_flag", pd.Series(1, index=universe_df.index)) == 1].iterrows():
        try:
            bs = build_issuer_bond_stack(str(co["cik"]).zfill(10), co["ticker"])
            if not bs.empty:
                bond_stacks.append(bs)
        except Exception as exc:
            logger.debug("TRACE enrichment skipped for %s: %s", co["ticker"], exc)
    combined_bond_stack = pd.concat(bond_stacks, ignore_index=True) if bond_stacks else pd.DataFrame()

    # Iterate over quarter-end dates
    quarter_ends = get_all_quarter_ends(config.BACKTEST_START, config.BACKTEST_END)
    signal_frames = []

    for qe in quarter_ends:
        as_of = pd.Timestamp(qe)
        logger.debug("Computing signals as of %s", as_of.date())

        mcs_df = compute_maturity_concentration_score(quarterly_schedule, as_of)
        if mcs_df.empty:
            continue
        mcs_df = compute_months_to_peak_maturity(mcs_df, as_of)
        rdm_df = compute_refinancing_difficulty_multiplier(
            mcs_df, combined_bond_stack, macro_rates_df, as_of, universe_df
        )
        signal_df = compute_composite_stress_signal(mcs_df, rdm_df)
        signal_frames.append(signal_df)

    if not signal_frames:
        logger.error("No signal frames generated — check schedule and rate data coverage")
        return pd.DataFrame()

    rolling_df = pd.concat(signal_frames, ignore_index=True)
    rolling_df.to_parquet(config.ROLLING_SIGNALS_PARQUET)
    logger.info(
        "Rolling signal history saved: %d rows, %d companies, %d quarters",
        len(rolling_df),
        rolling_df["ticker"].nunique(),
        rolling_df["as_of_date"].nunique(),
    )
    return rolling_df

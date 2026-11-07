"""
xbrl_parser.py
==============

XBRL Debt Maturity Schedule Extraction for the Maturity Wall Strategy
----------------------------------------------------------------------

**The XBRL Tag Inconsistency Problem**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The US-GAAP XBRL taxonomy provides *standard* tag names for debt maturity
schedules (e.g. ``LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths``).
In practice, fewer than 60% of SEC filers use these standard tags.  The
remaining companies use:

1. **Company-extension tags**: Tagged in the company's own namespace, e.g.
   ``hca:LongTermDebtMaturingInYearOne`` (HCA Healthcare, pre-2018) or
   ``chk:AggregateAnnualMaturitiesOfLongTermDebtYearOne`` (Chesapeake Energy).
   These tags are defined in the company's custom taxonomy XSD file, which
   is referenced in the filing's XBRL instance document.

2. **Deprecated standard tags**: FASB updates the taxonomy periodically.
   The 2015 update deprecated ``LongTermDebtMaturitiesRepaymentsOfPrincipalInYears1Through5``
   in favour of individual year tags.  Some filers continued using the
   deprecated aggregated form for years.

3. **Rounding inconsistencies**: Values may be in USD, thousands, or millions
   depending on the ``decimals`` attribute in the XBRL instance.  The API
   response usually normalises to USD but this must be verified.

4. **Duplicate reporting**: A company may report maturity data in *both*
   ``LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths`` AND
   a company-extension tag for the same fiscal year.  Naive summation
   double-counts.  The deduplication rule used here: *take the maximum
   value from all tags mapping to the same bucket and same filing date*.
   This is conservative for stress analysis (errs toward higher estimated
   maturities) and avoids the gross-vs-net ambiguity.

5. **Lease contamination**: ``FinanceLeaseLiabilityPaymentsDueNextTwelveMonths``
   looks identical in structure to the debt maturity tag but represents lease
   obligations — not bond debt.  Including it inflates the maturity schedule
   by 15–40% for capital-intensive companies.  All lease and purchase-
   obligation tags are in the EXCLUDE list and are explicitly filtered out.

**Design Choices**
~~~~~~~~~~~~~~~~~~
- Two-pass parsing: standard tags first, then fuzzy extension-tag search.
- Tag taxonomy CSV (``data/reference/xbrl_tag_taxonomy.csv``) is the living
  dictionary that maps discovered tags to standard bucket labels.
- The UNKNOWN rate (tags that could not be classified) is the primary data-
  quality KPI.  Target: < 10% of total facts.
- validation step (``validate_xbrl_against_10k_text``) compares the sum of the
  maturity schedule against the LongTermDebt balance-sheet line.  Coverage
  ratios < 0.5 or > 1.5 trigger exclusion or USE_WITH_CAUTION flags.

**Example tag variants observed in the wild** (non-exhaustive):
::

    Standard:
      us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths
      us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo
      us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive

    Company extensions:
      hca:LongTermDebtMaturingInYearOne       (HCA Healthcare, 2014–2017)
      chk:AggregateAnnualMaturitiesOfLongTermDebtYearOne  (Chesapeake, 2014–2019)
      dis:LongTermDebtMaturitiesYearOne       (Disney, pre-2020)
      ctl:LongTermDebtDueInOneYear            (CenturyLink/Lumen, 2015–2021)
      ge:DebtMaturitiesRepaymentOfPrincipalInYearOne  (GE Capital, 2014–2018)

    Must exclude:
      us-gaap:OperatingLeaseLiabilityPaymentsDueNextTwelveMonths  (lease)
      us-gaap:PurchaseObligationDueInNextTwelveMonths              (supply)
      us-gaap:ContractualObligationDueInNextTwelveMonths           (mixed)

Entry Point (CLI smoke test)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    python -m src.xbrl_parser --cik 0000860731

    Runs the full extraction pipeline for HCA Healthcare (CIK 860731),
    prints the maturity schedule table, coverage_flag, and validation
    recommendation.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

import config
from src.utils import get_quarter_label, get_quarter_end

logger = logging.getLogger(__name__)

# Regex patterns for fuzzy bucket classification
_BUCKET_PATTERNS = [
    (re.compile(r"(?i)(NextTwelveMonths|InYearOne|YearOne|Year1|DueIn1Year|DueInNextYear|InNextYear|RemainderOfFiscalYear|CurrentPortion)"), "Y1"),
    (re.compile(r"(?i)(InYearTwo|YearTwo|Year2|DueIn2Year)"), "Y2"),
    (re.compile(r"(?i)(InYearThree|YearThree|Year3|DueIn3Year)"), "Y3"),
    (re.compile(r"(?i)(InYearFour|YearFour|Year4|DueIn4Year)"), "Y4"),
    (re.compile(r"(?i)(InYearFive|YearFive|Year5|DueIn5Year)"), "Y5"),
    (re.compile(r"(?i)(AfterYearFive|Thereafter|AfterYear5|BeyondFiveYears|Due(?:After|Beyond)5)"), "Y5plus"),
    (re.compile(r"(?i)(RemainderOfFiscalYear|RemainderOf(?:Current)?Fiscal)"), "remainder_fy"),
    (re.compile(r"(?i)(InRollingYearTwo)"), "Y2"),
    (re.compile(r"(?i)(InRollingYearThree)"), "Y3"),
    (re.compile(r"(?i)(InRollingYearFour)"), "Y4"),
    (re.compile(r"(?i)(InRollingYearFive)"), "Y5"),
    (re.compile(r"(?i)(InNextRollingTwelveMonths|InNextRolling12)"), "Y1"),
    (re.compile(r"(?i)(AfterRollingYearFive)"), "Y5plus"),
]

# Strings in tag names that indicate the tag is a debt maturity variant
_FUZZY_INCLUDE_TERMS = [t.lower() for t in config.XBRL_FUZZY_MATCH_TERMS]

# Strings in tag names that indicate the tag must be excluded
_EXCLUDE_TERMS = [t.lower() for t in config.XBRL_EXCLUDE_TERMS]


# ---------------------------------------------------------------------------
# 1. Extract raw debt maturity facts
# ---------------------------------------------------------------------------

def extract_debt_maturity_facts(
    xbrl_facts: dict,
    tag_taxonomy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Search XBRL facts JSON for all debt-maturity-related tag observations.

    Implements a two-pass extraction strategy:

    **Pass 1 — Standard tags**: Exact lookup of every tag name in
    ``config.XBRL_MATURITY_TAGS`` within the ``us-gaap`` namespace.

    **Pass 2 — Extension tags**: Fuzzy search across all namespaces for any
    tag whose name contains at least one of ``config.XBRL_FUZZY_MATCH_TERMS``
    and does NOT contain any of ``config.XBRL_EXCLUDE_TERMS``.  This catches
    company-specific extensions not covered by the standard list.

    Parameters
    ----------
    xbrl_facts : dict
        Raw JSON dict from ``data_loader.fetch_company_xbrl_facts``.
    tag_taxonomy_df : pd.DataFrame
        Loaded ``xbrl_tag_taxonomy.csv`` — used to check if a tag should be
        excluded (``maturity_bucket == "EXCLUDE"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``cik``, ``tag_name``, ``tag_namespace``, ``period_end``,
        ``accession_number``, ``value_usd``, ``unit``, ``is_standard_tag``,
        ``maturity_bucket_raw``.
        Empty DataFrame if no matching tags found.

    Notes
    -----
    The EDGAR XBRL facts JSON is structured as::

        facts -> {namespace} -> {tag_name} -> units -> {unit} -> [{observations}]

    Each observation has keys: ``accn`` (accession), ``end`` (period end),
    ``val`` (numeric value), ``form`` (10-K, 10-Q, etc.).

    Only ``form`` values in ``["10-K", "10-K/A"]`` are retained for the annual
    maturity schedule.  10-Q observations are extracted separately and are
    useful for intra-year updates but are not the primary source.
    """
    cik = xbrl_facts.get("cik", "UNKNOWN")
    entity = xbrl_facts.get("entityName", "UNKNOWN")
    facts_by_ns: dict = xbrl_facts.get("facts", {})

    # Build exclude set from taxonomy
    exclude_tags: set = set()
    if not tag_taxonomy_df.empty and "maturity_bucket" in tag_taxonomy_df.columns:
        excl = tag_taxonomy_df[tag_taxonomy_df["maturity_bucket"] == "EXCLUDE"]["tag_name"]
        exclude_tags = set(excl.str.lower())

    rows = []
    n_standard = 0
    n_extension = 0

    for namespace, tags in facts_by_ns.items():
        for tag_name, tag_data in tags.items():
            tag_lower = tag_name.lower()

            # Skip excluded tags
            if tag_lower in exclude_tags:
                continue
            if any(excl_term in tag_lower for excl_term in _EXCLUDE_TERMS):
                continue

            is_standard = (
                namespace == "us-gaap"
                and tag_name in config.XBRL_MATURITY_TAGS
            )

            # Pass 1: exact standard tag match
            is_fuzzy = not is_standard and any(
                term in tag_lower for term in _FUZZY_INCLUDE_TERMS
            )

            if not (is_standard or is_fuzzy):
                continue

            # Extract USD unit observations
            units: dict = tag_data.get("units", {})
            usd_obs = units.get("USD", [])
            if not usd_obs:
                continue

            for obs in usd_obs:
                form = obs.get("form", "")
                if not isinstance(form, str):
                    form = ""
                # Keep both 10-K and 10-Q but flag for downstream filtering
                rows.append({
                    "cik": str(cik),
                    "entity_name": entity,
                    "tag_name": tag_name,
                    "tag_namespace": namespace,
                    "period_end": obs.get("end", ""),
                    "accession_number": obs.get("accn", ""),
                    "form_type": form,
                    "value_usd": float(obs.get("val", 0.0)),
                    "unit": "USD",
                    "is_standard_tag": is_standard,
                    "maturity_bucket_raw": "UNKNOWN",  # resolved next
                })
            if is_standard:
                n_standard += 1
            else:
                n_extension += 1

    logger.info(
        "CIK %s (%s): %d standard tags found, %d extension tags found, %d total facts",
        cik, entity, n_standard, n_extension, len(rows),
    )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")

    # Resolve maturity bucket for each tag
    df["maturity_bucket_raw"] = df["tag_name"].apply(
        lambda t: resolve_maturity_bucket(t, tag_taxonomy_df, pd.Timestamp.today())
    )

    n_unknown = (df["maturity_bucket_raw"] == "UNKNOWN").sum()
    pct_unknown = 100.0 * n_unknown / len(df) if len(df) else 0.0
    logger.info(
        "CIK %s: %.1f%% of facts have UNKNOWN bucket (target < 10%%)",
        cik, pct_unknown,
    )
    if pct_unknown > 10:
        logger.warning(
            "CIK %s: UNKNOWN rate %.1f%% exceeds 10%% — taxonomy needs extension",
            cik, pct_unknown,
        )

    return df


# ---------------------------------------------------------------------------
# 2. Resolve maturity bucket
# ---------------------------------------------------------------------------

def resolve_maturity_bucket(
    tag_name: str,
    tag_taxonomy_df: pd.DataFrame,
    filing_date: pd.Timestamp,
) -> str:
    """
    Map a raw XBRL tag name to a standardised maturity bucket label.

    Resolution priority
    -------------------
    1. Exact match in ``tag_taxonomy_df`` (primary — checked first).
    2. Regex pattern matching on the tag name (secondary).
    3. Substring catch-all for "AfterYearFive" / "Thereafter" (tertiary).
    4. Returns ``"UNKNOWN"`` if no match — these are logged for manual
       taxonomy extension.

    Parameters
    ----------
    tag_name : str
        Raw XBRL tag name (e.g.
        ``"LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths"``
        or a company extension).
    tag_taxonomy_df : pd.DataFrame
        Loaded taxonomy CSV.
    filing_date : pd.Timestamp
        Filing date context (reserved for future date-conditional taxonomy
        rules — e.g. a tag may change meaning across taxonomy versions).

    Returns
    -------
    str
        One of: ``"Y1"``, ``"Y2"``, ``"Y3"``, ``"Y4"``, ``"Y5"``,
        ``"Y5plus"``, ``"current"``, ``"remainder_fy"``, ``"TOTAL"``,
        ``"EXCLUDE"``, ``"UNKNOWN"``.
    """
    # 1. Exact taxonomy match (case-sensitive)
    if not tag_taxonomy_df.empty and "tag_name" in tag_taxonomy_df.columns:
        # Match on the local tag name (ignoring namespace prefix)
        local_tag = tag_name.split(":")[-1]
        match = tag_taxonomy_df[tag_taxonomy_df["tag_name"] == local_tag]
        if not match.empty:
            return match.iloc[0]["maturity_bucket"]

    # 2. Regex pattern matching
    for pattern, bucket in _BUCKET_PATTERNS:
        if pattern.search(tag_name):
            return bucket

    # 3. Catch-all substrings
    tag_lower = tag_name.lower()
    if "current" in tag_lower and "longterm" in tag_lower:
        return "current"
    if "total" in tag_lower or "aggregate" in tag_lower:
        return "TOTAL"

    return "UNKNOWN"


# ---------------------------------------------------------------------------
# 3. Build annual maturity schedule
# ---------------------------------------------------------------------------

def build_annual_maturity_schedule(
    cik: str,
    raw_facts_df: pd.DataFrame,
    filing_history_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct an annual maturity schedule (Y1–Y5+) for each 10-K filing.

    Deduplication rule: when multiple XBRL tags map to the same maturity
    bucket for the same (accession, bucket) pair, take the **maximum** value.
    This conservative approach avoids undercounting from net-vs-gross tag
    ambiguity while preventing accidental summation of duplicates.

    Parameters
    ----------
    cik : str
        SEC CIK number (for logging and output labeling).
    raw_facts_df : pd.DataFrame
        Output of ``extract_debt_maturity_facts``.
    filing_history_df : pd.DataFrame
        Optionally pre-loaded filing history (from EDGAR submissions API).
        If empty, the filing dates are inferred from the XBRL period_end
        dates in ``raw_facts_df``.

    Returns
    -------
    pd.DataFrame
        Columns: ``cik``, ``entity_name``, ``filing_date``, ``period_end``,
        ``accession_number``, ``Y1``, ``Y2``, ``Y3``, ``Y4``, ``Y5``,
        ``Y5plus``, ``total_reported``, ``coverage_flag``.
        One row per annual filing (10-K) that contains at least one debt
        maturity fact.

    Notes
    -----
    ``coverage_flag`` values:

    - ``"COMPLETE"`` — all of Y1, Y2, Y3, Y4, Y5 present (Y5plus optional).
    - ``"PARTIAL"``  — 2–4 of the primary buckets present.
    - ``"MINIMAL"``  — 0–1 buckets present; XBRL parsing effectively failed.

    Only COMPLETE and PARTIAL rows should be used in downstream signal
    computation.
    """
    if raw_facts_df is None or raw_facts_df.empty:
        logger.warning("CIK %s: no raw facts to build maturity schedule from", cik)
        return pd.DataFrame()

    # Filter to 10-K annual filings and meaningful maturity buckets
    annual_buckets = {"Y1", "Y2", "Y3", "Y4", "Y5", "Y5plus", "remainder_fy"}
    df = raw_facts_df[
        (raw_facts_df["form_type"].isin(["10-K", "10-K/A"]))
        & (raw_facts_df["maturity_bucket_raw"].isin(annual_buckets))
        & (raw_facts_df["value_usd"] > 0)
    ].copy()

    if df.empty:
        logger.warning("CIK %s: no 10-K maturity facts after filtering", cik)
        return pd.DataFrame()

    # Normalise remainder_fy → Y1 for annual schedule
    df["bucket"] = df["maturity_bucket_raw"].replace("remainder_fy", "Y1")

    # Deduplication: max value per (accession, bucket)
    df_dedup = (
        df.groupby(["accession_number", "period_end", "bucket"], as_index=False)["value_usd"]
        .max()
    )

    # Pivot to wide format: one row per filing
    pivot = df_dedup.pivot_table(
        index=["accession_number", "period_end"],
        columns="bucket",
        values="value_usd",
        aggfunc="max",
    ).reset_index()
    pivot.columns.name = None

    # Ensure all bucket columns exist
    for col in ["Y1", "Y2", "Y3", "Y4", "Y5", "Y5plus"]:
        if col not in pivot.columns:
            pivot[col] = np.nan

    # Convert USD → millions
    for col in ["Y1", "Y2", "Y3", "Y4", "Y5", "Y5plus"]:
        pivot[col] = pivot[col] / 1e6

    pivot["total_reported"] = pivot[["Y1", "Y2", "Y3", "Y4", "Y5", "Y5plus"]].sum(axis=1)

    # Coverage flag
    primary_buckets = ["Y1", "Y2", "Y3", "Y4", "Y5"]
    pivot["n_buckets"] = pivot[primary_buckets].notna().sum(axis=1)
    pivot["coverage_flag"] = np.where(
        pivot["n_buckets"] >= 5, "COMPLETE",
        np.where(pivot["n_buckets"] >= 2, "PARTIAL", "MINIMAL")
    )

    pivot.rename(columns={"period_end": "period_end", "accession_number": "accession_number"}, inplace=True)
    pivot["cik"] = cik
    pivot["entity_name"] = raw_facts_df["entity_name"].iloc[0] if "entity_name" in raw_facts_df.columns else ""
    pivot["filing_date"] = pivot["period_end"]  # best approximation without submission history

    # Try to enrich with actual filing dates from submission history
    if not filing_history_df.empty and "accession_number" in filing_history_df.columns:
        filing_map = filing_history_df.set_index("accession_number")["filing_date"].to_dict()
        pivot["filing_date"] = pivot["accession_number"].map(filing_map).fillna(pivot["period_end"])

    pivot["filing_date"] = pd.to_datetime(pivot["filing_date"], errors="coerce")
    pivot = pivot.sort_values("filing_date").reset_index(drop=True)

    flag_dist = pivot["coverage_flag"].value_counts().to_dict()
    logger.info(
        "CIK %s: schedule built — %d filings | coverage: %s",
        cik, len(pivot), flag_dist,
    )
    return pivot[[
        "cik", "entity_name", "filing_date", "period_end", "accession_number",
        "Y1", "Y2", "Y3", "Y4", "Y5", "Y5plus", "total_reported", "coverage_flag", "n_buckets",
    ]]


# ---------------------------------------------------------------------------
# 4. Annual → quarterly schedule
# ---------------------------------------------------------------------------

def convert_annual_to_quarterly_schedule(
    schedule_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distribute annual maturity buckets evenly across four quarters.

    The XBRL schedule reports annual maturities (Y1 = all debt due in the
    next 12 months).  This function converts to quarterly granularity by
    dividing each annual bucket evenly across 4 quarters — a deliberate
    simplification documented as such.

    When TRACE bond data is available (``interpolation_method`` =
    ``"trace_enriched"``), the ``features.py`` module overrides this
    distribution with actual bond-level maturity dates.

    Parameters
    ----------
    schedule_df : pd.DataFrame
        Output of ``build_annual_maturity_schedule``.

    Returns
    -------
    pd.DataFrame
        Long-format quarterly schedule: one row per (company, quarter).
        Columns: ``cik``, ``entity_name``, ``filing_date``,
        ``quarter_label``, ``quarter_end_date``, ``debt_maturing_mm``,
        ``interpolation_method``.

    Notes
    -----
    The even-distribution assumption is most problematic for companies with
    known concentrated quarterly maturities (e.g. a $1bn bond due in March
    that would be in Y1 Q1 only, not spread across all four quarters).
    The TRACE enrichment path in ``features.py`` corrects this for companies
    with CUSIP-level data.

    The simplification is documented explicitly in the notebook's
    ``Section 4: Maturity Schedule Construction`` discussion.
    """
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame()

    rows = []
    bucket_to_year_offset = {
        "Y1": 0, "Y2": 1, "Y3": 2, "Y4": 3, "Y5": 4, "Y5plus": 5,
    }

    for _, row in schedule_df.iterrows():
        filing_date = pd.Timestamp(row["filing_date"])
        filing_quarter = pd.Period(filing_date, freq="Q")

        for bucket, year_offset in bucket_to_year_offset.items():
            annual_amount = row.get(bucket, np.nan)
            if pd.isna(annual_amount) or annual_amount <= 0:
                continue
            quarterly_amount = annual_amount / 4.0  # even distribution assumption

            for q_within_year in range(4):
                total_quarter_offset = year_offset * 4 + q_within_year
                target_period = filing_quarter + total_quarter_offset
                quarter_label = f"{target_period.year}Q{target_period.quarter}"
                quarter_end = get_quarter_end(quarter_label)

                rows.append({
                    "cik": row["cik"],
                    "entity_name": row.get("entity_name", ""),
                    "filing_date": filing_date,
                    "quarter_label": quarter_label,
                    "quarter_end_date": quarter_end,
                    "debt_maturing_mm": quarterly_amount,
                    "interpolation_method": "even_distribution",
                    "source_bucket": bucket,
                })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result = result.sort_values(["cik", "filing_date", "quarter_end_date"]).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------

def validate_xbrl_against_10k_text(
    cik: str,
    schedule_df: pd.DataFrame,
    xbrl_facts: Optional[dict] = None,
) -> dict:
    """
    Sanity-check XBRL maturity schedule totals against LongTermDebt balances.

    Computes a *coverage_ratio* for each filing:

    .. math::

        \\text{coverage\\_ratio} = \\frac{\\text{Y1}+\\text{Y2}+\\text{Y3}+\\text{Y4}+\\text{Y5}+\\text{Y5plus}}{\\text{LongTermDebt}}

    Expected range: **0.85–1.15**.  Ratios outside this range indicate:

    - ``< 0.5``: Severe undercounting.  The maturity schedule is only
      capturing part of total debt (e.g. bonds excluded, extension tags
      missed).  A coverage_ratio of 0.3 means we are capturing 30% of the
      debt maturity schedule.  The MCS computed from incomplete data will
      **systematically understate concentration risk**.  These issuers must
      be excluded.
    - ``> 1.5``: Likely double-counting from tag duplication.  May indicate
      that both a standard tag and an extension tag are being included for
      the same underlying instrument.  Also requires exclusion or manual review.

    Parameters
    ----------
    cik : str
        SEC CIK number.
    schedule_df : pd.DataFrame
        Annual maturity schedule from ``build_annual_maturity_schedule``.
    xbrl_facts : dict, optional
        Raw XBRL JSON (used to extract LongTermDebt totals directly).
        If ``None``, validation uses the ``total_reported`` column from
        ``schedule_df`` only and skips the LongTermDebt cross-check.

    Returns
    -------
    dict
        Keys: ``cik``, ``n_filings``, ``mean_coverage_ratio``,
        ``min_coverage_ratio``, ``max_coverage_ratio``,
        ``outlier_filings`` (list of accession numbers), ``recommendation``
        (``"USE"``, ``"USE_WITH_CAUTION"``, ``"EXCLUDE"``).
    """
    if schedule_df is None or schedule_df.empty:
        return {
            "cik": cik,
            "n_filings": 0,
            "mean_coverage_ratio": np.nan,
            "min_coverage_ratio": np.nan,
            "max_coverage_ratio": np.nan,
            "outlier_filings": [],
            "recommendation": "EXCLUDE",
        }

    # Extract LongTermDebt totals from XBRL facts if available
    ltd_by_accn: dict = {}
    if xbrl_facts is not None:
        facts = xbrl_facts.get("facts", {})
        for ltd_tag in ["LongTermDebt", "LongTermDebtNoncurrent", "DebtLongTermAndShortTermCombinedAmount"]:
            tag_data = facts.get("us-gaap", {}).get(ltd_tag, {})
            for obs in tag_data.get("units", {}).get("USD", []):
                if obs.get("form") in ("10-K", "10-K/A"):
                    accn = obs.get("accn", "")
                    val_mm = float(obs.get("val", 0)) / 1e6
                    if val_mm > 0:
                        # Keep max (prefer gross)
                        ltd_by_accn[accn] = max(ltd_by_accn.get(accn, 0), val_mm)

    coverage_ratios = []
    outlier_filings = []

    for _, row in schedule_df.iterrows():
        accn = row.get("accession_number", "")
        total_maturity = row.get("total_reported", 0.0) or 0.0

        ltd = ltd_by_accn.get(accn, 0.0)
        if ltd <= 0:
            continue  # can't validate without LTD
        coverage = total_maturity / ltd
        coverage_ratios.append(coverage)

        if coverage < 0.5 or coverage > 1.5:
            outlier_filings.append({
                "accession_number": accn,
                "filing_date": str(row.get("filing_date", "")),
                "coverage_ratio": round(coverage, 3),
            })

    if not coverage_ratios:
        mean_cov = np.nan
        min_cov = np.nan
        max_cov = np.nan
        recommendation = "USE_WITH_CAUTION"
    else:
        mean_cov = float(np.mean(coverage_ratios))
        min_cov = float(np.min(coverage_ratios))
        max_cov = float(np.max(coverage_ratios))

        if mean_cov < 0.5:
            recommendation = "EXCLUDE"
        elif mean_cov > 1.5 or min_cov < 0.3:
            recommendation = "EXCLUDE"
        elif len(outlier_filings) / max(len(coverage_ratios), 1) > 0.3:
            recommendation = "USE_WITH_CAUTION"
        else:
            recommendation = "USE"

    logger.info(
        "CIK %s validation: mean_coverage=%.2f, rec=%s, %d outlier filings",
        cik, mean_cov if not np.isnan(mean_cov) else -1, recommendation, len(outlier_filings),
    )

    return {
        "cik": cik,
        "n_filings": len(schedule_df),
        "mean_coverage_ratio": round(mean_cov, 3) if not np.isnan(mean_cov) else None,
        "min_coverage_ratio": round(min_cov, 3) if not np.isnan(min_cov) else None,
        "max_coverage_ratio": round(max_cov, 3) if not np.isnan(max_cov) else None,
        "outlier_filings": outlier_filings,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# 6. Universe orchestrator
# ---------------------------------------------------------------------------

def build_universe_maturity_schedules(
    universe_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the full XBRL pipeline for every company in the universe.

    Pipeline per company:
    1. ``data_loader.fetch_company_xbrl_facts``
    2. ``extract_debt_maturity_facts``
    3. ``build_annual_maturity_schedule``
    4. ``convert_annual_to_quarterly_schedule``
    5. ``validate_xbrl_against_10k_text``

    Parameters
    ----------
    universe_df : pd.DataFrame
        Loaded ``company_universe.csv`` with columns ``ticker``, ``cik``,
        ``include_flag``.

    Returns
    -------
    pd.DataFrame
        Combined quarterly maturity schedule for all included companies.
        Saved to ``data/processed/universe_maturity_schedules.parquet``.
        Quality report saved to ``data/processed/xbrl_quality_report.csv``.

    Notes
    -----
    Companies with ``include_flag == 0`` or recommendation ``"EXCLUDE"``
    are not included in the output panel but are recorded in the quality report.

    Aggregate log line::

        Universe XBRL coverage: {n_complete}/{n_total} complete,
        {n_partial} partial, {n_minimal} minimal/excluded
    """
    from src.data_loader import fetch_company_xbrl_facts

    tag_taxonomy = pd.read_csv(config.XBRL_TAG_TAXONOMY_CSV)
    included = universe_df[universe_df.get("include_flag", pd.Series(1)) == 1]

    all_quarterly = []
    quality_rows = []

    n_complete = n_partial = n_minimal = 0

    for _, co in included.iterrows():
        ticker = co["ticker"]
        cik = str(co["cik"]).zfill(10)
        logger.info("Processing %s (CIK %s)", ticker, cik)

        try:
            xbrl_facts = fetch_company_xbrl_facts(cik)
            raw_facts_df = extract_debt_maturity_facts(xbrl_facts, tag_taxonomy)
            annual_df = build_annual_maturity_schedule(cik, raw_facts_df, pd.DataFrame())
            quarterly_df = convert_annual_to_quarterly_schedule(annual_df)
            val_result = validate_xbrl_against_10k_text(cik, annual_df, xbrl_facts)

            if not annual_df.empty:
                flag_counts = annual_df["coverage_flag"].value_counts()
                n_complete += int(flag_counts.get("COMPLETE", 0))
                n_partial  += int(flag_counts.get("PARTIAL", 0))
                n_minimal  += int(flag_counts.get("MINIMAL", 0))

            quality_rows.append({
                "ticker": ticker,
                "cik": cik,
                "n_annual_filings": len(annual_df),
                "complete_filings": int(annual_df["coverage_flag"].eq("COMPLETE").sum()) if not annual_df.empty else 0,
                "partial_filings": int(annual_df["coverage_flag"].eq("PARTIAL").sum()) if not annual_df.empty else 0,
                "minimal_filings": int(annual_df["coverage_flag"].eq("MINIMAL").sum()) if not annual_df.empty else 0,
                "mean_coverage_ratio": val_result.get("mean_coverage_ratio"),
                "recommendation": val_result.get("recommendation", "EXCLUDE"),
                "n_quarterly_rows": len(quarterly_df),
            })

            if val_result["recommendation"] != "EXCLUDE" and not quarterly_df.empty:
                quarterly_df["ticker"] = ticker
                all_quarterly.append(quarterly_df)
            else:
                logger.warning("Excluding %s from signal panel (recommendation: %s)",
                               ticker, val_result["recommendation"])

        except Exception as exc:
            logger.error("Pipeline failed for %s / %s: %s", ticker, cik, exc, exc_info=True)
            quality_rows.append({
                "ticker": ticker, "cik": cik, "n_annual_filings": 0,
                "complete_filings": 0, "partial_filings": 0, "minimal_filings": 0,
                "mean_coverage_ratio": None, "recommendation": "EXCLUDE",
                "n_quarterly_rows": 0,
            })

    # Save quality report
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(config.XBRL_QUALITY_REPORT_CSV, index=False)
    logger.info("XBRL quality report saved: %s", config.XBRL_QUALITY_REPORT_CSV)

    n_total = len(included)
    logger.info(
        "Universe XBRL coverage: %d/%d complete, %d partial, %d minimal/excluded",
        n_complete, n_total, n_partial, n_minimal,
    )

    if not all_quarterly:
        logger.error("No quarterly schedules built for any company — check XBRL connectivity")
        return pd.DataFrame()

    master_df = pd.concat(all_quarterly, ignore_index=True)
    master_df.to_parquet(config.UNIVERSE_SCHEDULES_PARQUET)
    logger.info(
        "Universe maturity schedules saved: %s (%d rows, %d companies)",
        config.UNIVERSE_SCHEDULES_PARQUET, len(master_df), master_df["cik"].nunique(),
    )
    return master_df


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """Entry-point for ``python -m src.xbrl_parser --cik <CIK>``."""
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run the XBRL maturity-schedule extraction pipeline for one company."
    )
    parser.add_argument(
        "--cik", required=True,
        help="SEC CIK number (e.g. 0000860731 for HCA Healthcare)",
    )
    args = parser.parse_args()

    cik = args.cik.lstrip("0").zfill(10)
    logger.info("=== XBRL Pipeline Smoke Test: CIK %s ===", cik)

    from src.data_loader import fetch_company_xbrl_facts
    tag_taxonomy = pd.read_csv(config.XBRL_TAG_TAXONOMY_CSV)

    xbrl_facts = fetch_company_xbrl_facts(cik)
    entity = xbrl_facts.get("entityName", "UNKNOWN")
    logger.info("Entity: %s", entity)

    raw_facts_df = extract_debt_maturity_facts(xbrl_facts, tag_taxonomy)
    logger.info("Raw facts extracted: %d rows", len(raw_facts_df))

    annual_df = build_annual_maturity_schedule(cik, raw_facts_df, pd.DataFrame())
    if annual_df.empty:
        logger.error("No annual maturity schedule could be built — XBRL data insufficient")
        sys.exit(1)

    print("\n" + "=" * 72)
    print(f"  Annual Maturity Schedule — {entity} (CIK: {cik})")
    print("=" * 72)
    display_cols = ["filing_date", "Y1", "Y2", "Y3", "Y4", "Y5", "Y5plus",
                    "total_reported", "coverage_flag"]
    display_cols = [c for c in display_cols if c in annual_df.columns]
    print(annual_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    val = validate_xbrl_against_10k_text(cik, annual_df, xbrl_facts)
    print("\n" + "-" * 72)
    print(f"  Validation: mean_coverage_ratio = {val['mean_coverage_ratio']}")
    print(f"  Recommendation: {val['recommendation']}")
    print(f"  Outlier filings: {len(val['outlier_filings'])}")
    print("-" * 72 + "\n")


if __name__ == "__main__":
    _cli_main()

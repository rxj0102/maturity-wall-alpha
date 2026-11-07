"""
trace_parser.py
===============
FINRA TRACE bond reference data and CUSIP discovery for the maturity-wall
strategy.

What this module provides
-------------------------
- CUSIP discovery from EDGAR filing text (8-K, S-3, prospectus filings)
- Bond reference data fetch: coupon, maturity date, outstanding face value
- Per-issuer bond stack construction with computed fields
- Weighted average coupon (WAC) as a yield_at_issuance proxy

FINRA TRACE Data Availability
------------------------------
FINRA TRACE **historical transaction data** (individual trade prices,
volumes, time-stamps) requires a paid FINRA data subscription for bulk
programmatic access.  This module uses FINRA's **public bond reference
data** (coupon rate, maturity date, outstanding face value, credit ratings),
which is freely accessible via the FINRA Markets public API and bond search
page.

The bond reference data is the correct input for this strategy's
``yield_at_issuance`` proxy.  Historical trade price/volume data would
provide better price-discovery information (e.g. for marking distressed
bonds to market) but is not required for the maturity-wall thesis, which
is fundamentally about the *calendar* of debt maturities rather than
current trading levels.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import requests

import config
from src.utils import edgar_limiter, exponential_backoff

logger = logging.getLogger(__name__)

# CUSIP regex: 9 alphanumeric characters (first 6 issuer, 2 issue, 1 check)
CUSIP_PATTERN = re.compile(r"\b([0-9A-Z]{8}[0-9])\b")


# ---------------------------------------------------------------------------
# 1. CUSIP discovery from EDGAR filings
# ---------------------------------------------------------------------------

def build_cusip_list_from_edgar(cik: str) -> List[str]:
    """
    Discover bond CUSIPs from EDGAR filings for a given CIK.

    Searches 8-K filings (press releases for new bond issuances) and
    prospectus-type filings (S-3, 424B2, 424B3, 424B5) for 9-character
    CUSIP strings.

    Parameters
    ----------
    cik : str
        SEC CIK number.

    Returns
    -------
    list of str
        Unique CUSIP strings discovered (9-character alphanumeric).
        Empty list if none found or network unavailable.

    Notes
    -----
    Cached to ``data/raw/trace/{cik}_cusips.json``.

    EDGAR's full-text search API (``efts.sec.gov``) is used to locate
    filings mentioning "CUSIP".  The filing text is then scraped for the
    CUSIP pattern.  This approach finds CUSIPs in the body of indenture
    agreements and bond offering circulars but is not exhaustive — some
    issuers embed CUSIPs only in exhibits that require additional fetching.
    """
    cik_padded = str(cik).lstrip("0").zfill(10)
    cache_path = config.RAW_TRACE_DIR / f"{cik_padded}_cusips.json"

    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)

    cusips: set = set()
    session = requests.Session()
    session.headers.update(config.EDGAR_HEADERS)

    # Search EDGAR submissions for relevant filing types
    submissions_url = config.EDGAR_SUBMISSIONS_URL.format(cik=cik_padded)
    try:
        edgar_limiter()
        resp = session.get(submissions_url, timeout=20)
        resp.raise_for_status()
        submissions = resp.json()
    except Exception as exc:
        logger.warning("Could not fetch EDGAR submissions for CIK %s: %s", cik_padded, exc)
        return []

    # Extract recent filings index
    filings = submissions.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    dates = filings.get("filingDate", [])

    target_forms = {"8-K", "S-3", "424B2", "424B3", "424B5", "S-11"}
    prospectus_accessions = [
        (acc, d) for acc, form, d in zip(accessions, forms, dates)
        if form in target_forms
    ][:20]  # limit to 20 most recent to avoid excessive requests

    for accn, filing_date in prospectus_accessions:
        accn_clean = accn.replace("-", "")
        filing_idx_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik_padded)}/{accn_clean}/{accn}-index.htm"
        )
        try:
            edgar_limiter()
            r = session.get(filing_idx_url, timeout=15)
            found = set(CUSIP_PATTERN.findall(r.text))
            # Filter obvious non-CUSIPs (all zeros, all letters, etc.)
            valid = {c for c in found if _is_plausible_cusip(c)}
            cusips.update(valid)
            if valid:
                logger.debug(
                    "CIK %s filing %s (%s): found CUSIPs %s",
                    cik_padded, accn, filing_date, valid,
                )
        except Exception as exc:
            logger.debug("EDGAR filing fetch failed %s: %s", accn, exc)

    result = sorted(cusips)
    with open(cache_path, "w") as f:
        json.dump(result, f)
    logger.info("CIK %s: discovered %d CUSIPs from EDGAR filings", cik_padded, len(result))
    return result


def _is_plausible_cusip(cusip: str) -> bool:
    """Quick heuristic to filter obvious non-CUSIP 9-char strings."""
    if len(cusip) != 9:
        return False
    if cusip == cusip[0] * 9:
        return False  # all same character
    if cusip.isdigit() and int(cusip) == 0:
        return False
    return True


# ---------------------------------------------------------------------------
# 2. Bond reference data for a single CUSIP
# ---------------------------------------------------------------------------

def fetch_bond_reference_data(cusip: str) -> dict:
    """
    Fetch and enrich FINRA bond reference data for one CUSIP.

    Calls ``data_loader.fetch_trace_bond_data`` and adds computed fields:
    ``years_to_maturity``, ``is_matured``, ``coupon_type``.

    Parameters
    ----------
    cusip : str
        9-character CUSIP identifier.

    Returns
    -------
    dict
        All fields from ``fetch_trace_bond_data`` plus:

        - ``years_to_maturity`` : float — (maturity_date - today) / 365.25
        - ``is_matured`` : bool — True if maturity_date < today
        - ``coupon_type`` : str — ``"fixed"``, ``"floating"``, or ``"zero"``

    Notes
    -----
    ``coupon_type`` classification heuristic:

    - ``coupon_rate == 0.0`` → ``"zero"``
    - ``coupon_rate >= 0.01`` AND ``bond_type`` does not contain ``"float"``
      / ``"frn"`` / ``"variable"`` → ``"fixed"``
    - Otherwise → ``"floating"``
    """
    from src.data_loader import fetch_trace_bond_data

    data = fetch_trace_bond_data(cusip)
    if "error" in data:
        return data

    today = pd.Timestamp.today()
    maturity_str = data.get("maturity_date", "")
    try:
        maturity_ts = pd.Timestamp(maturity_str)
    except Exception:
        maturity_ts = pd.NaT

    data["is_matured"] = (maturity_ts < today) if pd.notna(maturity_ts) else True
    data["years_to_maturity"] = (
        (maturity_ts - today).days / 365.25
        if pd.notna(maturity_ts) else np.nan
    )

    coupon = float(data.get("coupon_rate", 0.0) or 0.0)
    bond_type = str(data.get("bond_type", "")).lower()
    if coupon == 0.0:
        data["coupon_type"] = "zero"
    elif any(ft in bond_type for ft in ("float", "frn", "variable", "adjustable")):
        data["coupon_type"] = "floating"
    else:
        data["coupon_type"] = "fixed"

    return data


# ---------------------------------------------------------------------------
# 3. Per-issuer bond stack
# ---------------------------------------------------------------------------

def build_issuer_bond_stack(cik: str, ticker: str) -> pd.DataFrame:
    """
    Build a complete CUSIP-level bond stack for one issuer.

    Discovers CUSIPs via EDGAR filing search, fetches reference data for each,
    and compiles into a DataFrame.

    Parameters
    ----------
    cik : str
        SEC CIK number.
    ticker : str
        Equity ticker (for logging and file naming).

    Returns
    -------
    pd.DataFrame
        Columns: ``cik``, ``ticker``, ``cusip``, ``coupon_rate``,
        ``maturity_date``, ``outstanding_face_mm``, ``coupon_type``,
        ``years_to_maturity``, ``is_matured``, ``moody_rating``,
        ``sp_rating``.
        Sorted by ``maturity_date`` ascending.

    Notes
    -----
    Saved to ``data/raw/trace/{ticker}_bond_stack.csv``.

    Logs a summary:

    .. code-block::

        {ticker}: {n_bonds} bonds found, total face ${total:.0f}MM,
        earliest maturity {earliest}, latest maturity {latest}
    """
    cache_path = config.RAW_TRACE_DIR / f"{ticker}_bond_stack.csv"
    if cache_path.exists():
        logger.info("Bond stack cache hit: %s", cache_path)
        return pd.read_csv(cache_path)

    cusips = build_cusip_list_from_edgar(cik)
    if not cusips:
        logger.warning("%s (CIK %s): no CUSIPs discovered — TRACE enrichment unavailable", ticker, cik)
        return pd.DataFrame()

    rows = []
    for cusip in cusips:
        data = fetch_bond_reference_data(cusip)
        if "error" in data:
            logger.debug("No data for CUSIP %s: %s", cusip, data.get("error"))
            continue
        rows.append({
            "cik": cik,
            "ticker": ticker,
            "cusip": cusip,
            "coupon_rate": data.get("coupon_rate", np.nan),
            "maturity_date": data.get("maturity_date", ""),
            "outstanding_face_mm": data.get("outstanding_face_value", 0.0),
            "coupon_type": data.get("coupon_type", "fixed"),
            "years_to_maturity": data.get("years_to_maturity", np.nan),
            "is_matured": data.get("is_matured", True),
            "moody_rating": data.get("moody_rating", "NR"),
            "sp_rating": data.get("sp_rating", "NR"),
        })

    if not rows:
        logger.warning("%s: CUSIP fetch returned no valid bond data", ticker)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")
    df = df.sort_values("maturity_date").reset_index(drop=True)

    total_face = df["outstanding_face_mm"].sum()
    outstanding = df[~df["is_matured"].fillna(True)]
    earliest = outstanding["maturity_date"].min()
    latest = outstanding["maturity_date"].max()
    logger.info(
        "%s: %d bonds found, total face $%.0fMM, "
        "earliest maturity %s, latest maturity %s",
        ticker, len(df), total_face,
        earliest.strftime("%Y-%m") if pd.notna(earliest) else "N/A",
        latest.strftime("%Y-%m") if pd.notna(latest) else "N/A",
    )

    df.to_csv(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# 4. Weighted Average Coupon
# ---------------------------------------------------------------------------

def compute_weighted_avg_coupon(bond_stack_df: pd.DataFrame) -> float:
    """
    Compute the weighted average coupon (WAC) across outstanding fixed-rate bonds.

    .. math::

        \\text{WAC} = \\frac{\\sum_i \\text{coupon}_i \\times \\text{face}_i}
                           {\\sum_i \\text{face}_i}

    Parameters
    ----------
    bond_stack_df : pd.DataFrame
        Output of ``build_issuer_bond_stack``.

    Returns
    -------
    float
        WAC in percent (e.g. 6.25 means 6.25% annual coupon).
        Returns ``np.nan`` if no valid bonds are available.

    Notes
    -----
    **Limitations of WAC as a yield_at_issuance proxy:**

    WAC is an imprecise proxy for ``yield_at_issuance`` because it conflates
    bonds issued at different times and in different rate environments.  For
    a company that issued bonds in 2015 (low-rate era), 2018 (rising-rate
    tightening cycle), and 2021 (COVID-era ZIRP), the WAC blends three
    distinct rate regimes into a single number.

    A better proxy is the **coupon of the specific bond maturing in the
    peak stress quarter**, which reflects the rate environment at the time
    that particular instrument was issued.  The TRACE enrichment path in
    ``features.py`` implements this bond-specific approach when CUSIP-level
    data is available, falling back to WAC only when it is not.

    Additionally, WAC uses coupon rate rather than yield-to-maturity (YTM)
    at issuance.  For bonds issued at a discount (below par), YTM > coupon;
    for premium bonds, YTM < coupon.  The true YTM at issuance requires the
    original issue price, which is in the bond prospectus but not in TRACE
    reference data.  The resulting bias is small for most investment-grade
    bonds (typically issued near par) but can be material for deeply
    discounted high-yield paper.

    Floating-rate bonds are excluded from WAC because their effective rate
    is determined by the reference index (SOFR + spread), not the nominal
    coupon — including them would bias WAC toward the spread component only.
    """
    if bond_stack_df is None or bond_stack_df.empty:
        return np.nan

    # Exclude matured and floating-rate bonds
    eligible = bond_stack_df[
        (~bond_stack_df["is_matured"].fillna(True))
        & (bond_stack_df["coupon_type"] == "fixed")
        & (bond_stack_df["coupon_rate"].notna())
        & (bond_stack_df["coupon_rate"] > 0)
        & (bond_stack_df["outstanding_face_mm"] > 0)
    ].copy()

    if eligible.empty:
        logger.warning("WAC: no eligible fixed-rate outstanding bonds found")
        return np.nan

    wac = (eligible["coupon_rate"] * eligible["outstanding_face_mm"]).sum() / eligible["outstanding_face_mm"].sum()
    logger.debug("WAC computed from %d bonds: %.2f%%", len(eligible), wac)
    return float(wac)

"""
data_loader.py
==============
Data acquisition layer for the Corporate Bond Maturity Wall strategy.

Responsibilities
----------------
- Fetch and cache EDGAR XBRL company-facts JSON (full filing history)
- Fetch and cache FRED macroeconomic rate series
- Fetch yfinance equity prices
- Fetch FINRA TRACE bond reference data (coupon, maturity, outstanding)
- Orchestrate bond stack assembly: XBRL debt schedule + TRACE enrichment

All network I/O is rate-limited via ``src.utils.edgar_limiter``.
Caching is disk-based (data/raw/ and data/processed/) to support
offline/reproducible runs.  See config.py for all paths and thresholds.

FINRA TRACE Note
----------------
FINRA TRACE historical transaction data (individual trade prices and
volumes) requires a paid FINRA data subscription for bulk programmatic
access.  This module uses FINRA's **bond reference data** endpoint
(coupon, maturity date, outstanding face value, credit ratings), which
is freely accessible via the public FINRA Markets API.  This is the
appropriate data for computing yield_at_issuance proxies and building
the bond maturity calendar.  Historical trade price / volume data is not
required for this strategy and is not fetched here.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

import config
from src.utils import edgar_limiter, exponential_backoff

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_session() -> requests.Session:
    """Return a requests Session with EDGAR-compliant headers."""
    s = requests.Session()
    s.headers.update(config.EDGAR_HEADERS)
    return s


def _retry_get(
    session: requests.Session,
    url: str,
    max_retries: int = config.MAX_RETRIES,
    use_limiter: bool = True,
) -> requests.Response:
    """
    GET with exponential backoff on 429 / 503.

    Parameters
    ----------
    session : requests.Session
    url : str
    max_retries : int
    use_limiter : bool
        If True, consumes an EDGAR rate-limiter token before each request.

    Returns
    -------
    requests.Response

    Raises
    ------
    requests.HTTPError
        If all retries are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            if use_limiter:
                edgar_limiter()
            resp = session.get(url, timeout=30)
            if resp.status_code in (429, 503):
                wait = exponential_backoff(attempt)
                logger.warning(
                    "Rate-limited (HTTP %s) on %s; sleeping %.1fs",
                    resp.status_code, url, wait,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = exponential_backoff(attempt)
                logger.warning("Request error on %s (attempt %d): %s; retrying in %.1fs",
                               url, attempt + 1, exc, wait)
                time.sleep(wait)
    raise requests.HTTPError(f"All {max_retries} retries exhausted for {url}") from last_exc


# ---------------------------------------------------------------------------
# 1. EDGAR XBRL Company Facts
# ---------------------------------------------------------------------------

def fetch_company_xbrl_facts(cik: str) -> dict:
    """
    Download the full XBRL company facts JSON from EDGAR for one company.

    EDGAR's ``/api/xbrl/companyfacts/CIK{cik}.json`` endpoint returns every
    XBRL fact ever tagged by the company across all filings — typically 5–20 MB
    for a large filer.  A single call returns the complete filing history
    (no pagination).

    Results are cached to ``data/raw/xbrl/{cik}_facts.json``.  Subsequent
    calls load from cache and skip the network request.

    Parameters
    ----------
    cik : str
        SEC CIK number (zero-padded to 10 digits or plain).

    Returns
    -------
    dict
        Parsed XBRL facts JSON.  Top-level keys: ``entityType``, ``cik``,
        ``entityName``, ``facts`` (the main payload — a nested dict of
        namespaces → tags → units → filing observations).

    Notes
    -----
    The EDGAR XBRL API enforces a rate limit of ~10 requests/second.  The
    ``edgar_limiter`` context manager is called before each request.
    Exponential backoff (2s, 4s, 8s) handles 429/503 responses.
    """
    cik_padded = cik.lstrip("0").zfill(10)
    cache_path = config.RAW_XBRL_DIR / f"{cik_padded}_facts.json"

    if cache_path.exists():
        logger.info("XBRL cache hit: %s", cache_path)
        with open(cache_path, "r") as f:
            data = json.load(f)
        n_facts = sum(
            len(vals.get("units", {}).get("USD", []))
            for ns in data.get("facts", {}).values()
            for vals in ns.values()
        )
        size_mb = cache_path.stat().st_size / 1_048_576
        logger.info("CIK %s: ~%d USD facts, %.1f MB (from cache)", cik_padded, n_facts, size_mb)
        return data

    url = config.EDGAR_XBRL_FACTS_URL.format(cik=cik_padded)
    logger.info("Fetching XBRL facts: %s", url)
    session = _get_session()
    resp = _retry_get(session, url)

    data = resp.json()
    with open(cache_path, "w") as f:
        json.dump(data, f)

    size_mb = len(resp.content) / 1_048_576
    n_facts = sum(
        len(vals.get("units", {}).get("USD", []))
        for ns in data.get("facts", {}).values()
        for vals in ns.values()
    )
    entity = data.get("entityName", "UNKNOWN")
    logger.info(
        "CIK %s (%s): %d USD facts, %.1f MB — cached to %s",
        cik_padded, entity, n_facts, size_mb, cache_path,
    )
    return data


# ---------------------------------------------------------------------------
# 2. FRED Rate Series
# ---------------------------------------------------------------------------

def fetch_fred_series(
    series_id: str,
    start_date: str = "2010-01-01",
) -> pd.Series:
    """
    Download a FRED time series as a pandas Series, with local caching.

    Downloads the CSV directly from ``fred.stlouisfed.org`` (no API key
    required for this endpoint).  The cache is refreshed automatically if the
    local file is older than ``config.FRED_CACHE_TTL_DAYS`` days.

    Missing observations (weekends, holidays) are forward-filled for up to
    5 business days to produce a continuous daily series suitable for
    merging with equity price data.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g. ``"DGS10"`` for the 10-year Treasury).
    start_date : str
        ISO date string; observations before this date are excluded.

    Returns
    -------
    pd.Series
        Daily series, index=pd.DatetimeIndex, values=float, name=series_id.
        Values are in the native FRED unit (percent for rates, bps for OAS).

    Notes
    -----
    FRED publishes ``"."`` for missing values in the CSV.  These are replaced
    with ``NaN`` before forward-filling.  A maximum forward-fill of 5 business
    days is applied to avoid propagating staleness across long holiday periods
    (e.g., the Christmas–New Year gap in Treasury trading).
    """
    cache_path = config.RAW_FRED_DIR / f"{series_id}.csv"
    ttl = timedelta(days=config.FRED_CACHE_TTL_DAYS)
    need_refresh = True

    if cache_path.exists():
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if age < ttl:
            logger.info("FRED cache hit: %s (age: %.0fd)", series_id, age.days)
            need_refresh = False

    if need_refresh:
        url = config.FRED_API_URL.format(series=series_id)
        logger.info("Fetching FRED series: %s", url)
        session = _get_session()
        resp = _retry_get(session, url, use_limiter=False)
        with open(cache_path, "wb") as f:
            f.write(resp.content)
        logger.info("FRED %s cached to %s", series_id, cache_path)

    df = pd.read_csv(cache_path, index_col=0, parse_dates=True, na_values=[".", ""])
    df.index = pd.to_datetime(df.index)
    s = df.iloc[:, 0].astype(float)
    s.name = series_id
    s = s[s.index >= pd.Timestamp(start_date)]
    # Forward-fill up to 5 business days for weekends / holidays
    s = s.resample("B").last().ffill(limit=5)
    return s


def fetch_all_fred_series(start_date: str = "2010-01-01") -> pd.DataFrame:
    """
    Download all FRED series defined in ``config.FRED_SERIES``.

    Parameters
    ----------
    start_date : str
        ISO date string passed to each ``fetch_fred_series`` call.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame: index=date, columns=series keys
        (``T10Y``, ``T2Y``, ``HY_OAS``, ``IG_OAS``, ``SOFR``).

    Notes
    -----
    The resulting DataFrame is saved to ``data/processed/macro_rates.parquet``
    for use by downstream feature and backtest modules.
    """
    series_dict = {}
    for key, series_id in config.FRED_SERIES.items():
        logger.info("Downloading FRED series: %s -> %s", key, series_id)
        s = fetch_fred_series(series_id, start_date=start_date)
        series_dict[key] = s

    df = pd.DataFrame(series_dict)
    df.index.name = "date"
    # Align index to business days
    df = df.resample("B").last().ffill(limit=5)
    df.to_parquet(config.MACRO_RATES_PARQUET)
    logger.info(
        "Macro rates saved to %s: %d rows, %d series",
        config.MACRO_RATES_PARQUET, len(df), len(df.columns),
    )
    return df


# ---------------------------------------------------------------------------
# 3. Equity Prices (yfinance)
# ---------------------------------------------------------------------------

def fetch_equity_prices(
    tickers: List[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download adjusted-close equity prices from Yahoo Finance (yfinance).

    Results are cached to ``data/processed/equity_prices.parquet`` and
    loaded from cache if it already covers the requested date range.

    Parameters
    ----------
    tickers : list of str
        Equity tickers (Yahoo Finance format, e.g. ``["M", "HCA", "CHK"]``).
    start : str
        ISO start date (inclusive).
    end : str
        ISO end date (inclusive).

    Returns
    -------
    pd.DataFrame
        Adjusted close prices; index=date, columns=tickers.
        Columns with > 80% missing data are dropped and logged.

    Notes
    -----
    yfinance is used for convenience and reproducibility; it is not suitable
    for production trading systems where survivorship bias and split-adjustment
    discrepancies can affect results.  For a research backtest, it is the
    standard open-source choice.
    """
    import yfinance as yf

    cache_path = config.EQUITY_PRICES_PARQUET

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached_start = cached.index.min().strftime("%Y-%m-%d")
        cached_end   = cached.index.max().strftime("%Y-%m-%d")
        covered_tickers = set(cached.columns)
        need_refresh = (
            cached_start > start
            or cached_end < end
            or not set(tickers).issubset(covered_tickers)
        )
        if not need_refresh:
            logger.info("Equity price cache hit: %d tickers, %s to %s", len(tickers), start, end)
            return cached[tickers].loc[start:end]

    logger.info("Downloading equity prices for %d tickers via yfinance", len(tickers))
    # yfinance auto_adjust returns adjusted close
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Drop columns that are mostly missing
    missing_frac = prices.isna().mean()
    bad_tickers = missing_frac[missing_frac > 0.80].index.tolist()
    if bad_tickers:
        logger.warning("Dropping %d tickers with >80%% missing prices: %s", len(bad_tickers), bad_tickers)
        prices = prices.drop(columns=bad_tickers)

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    prices.to_parquet(cache_path)
    logger.info(
        "Equity prices saved: %d tickers, %s to %s -> %s",
        len(prices.columns), start, end, cache_path,
    )
    return prices


# ---------------------------------------------------------------------------
# 4. FINRA TRACE Bond Reference Data
# ---------------------------------------------------------------------------

def fetch_trace_bond_data(cusip: str) -> dict:
    """
    Fetch bond reference data for a single CUSIP from FINRA Markets API.

    Uses FINRA's public bond facts endpoint as the primary source.  Falls back
    to scraping the FINRA bond search page if the API returns no data.

    **Data available here (free):** issuer name, coupon rate, maturity date,
    outstanding face value, bond type, Moody's / S&P ratings.

    **Data NOT available here (subscription required):** Historical TRACE
    trade prices, transaction volumes, intraday quotes.  This module
    intentionally does not attempt to access subscription-only data.

    Parameters
    ----------
    cusip : str
        9-character CUSIP identifier (e.g. ``"404121AC3"``).

    Returns
    -------
    dict
        Keys: ``cusip``, ``issuer``, ``coupon_rate`` (float, annualized %),
        ``maturity_date`` (str ISO), ``outstanding_face_value`` (float, MM),
        ``bond_type`` (str), ``moody_rating`` (str), ``sp_rating`` (str),
        ``fetch_source`` (``"api"`` or ``"scrape"``).
        Returns ``{"cusip": cusip, "error": "...", "fetch_source": "none"}``
        on failure.

    Notes
    -----
    FINRA TRACE historical **transaction** data (actual trade prices and
    volumes) requires a FINRA data subscription for bulk access.  The bond
    reference data used here (coupon, maturity, outstanding) is freely
    accessible and is the appropriate input for the RDM yield_at_issuance
    proxy computation in this strategy.
    """
    cache_path = config.RAW_TRACE_DIR / f"{cusip}_facts.json"
    if cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)

    result = _fetch_trace_via_api(cusip)
    if "error" in result:
        logger.info("FINRA API failed for %s; trying scrape", cusip)
        result = _fetch_trace_via_scrape(cusip)

    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result


def _fetch_trace_via_api(cusip: str) -> dict:
    """Attempt FINRA Markets bond facts API."""
    url = config.FINRA_BOND_FACTS_URL.format(cusip=cusip)
    try:
        session = _get_session()
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return {"cusip": cusip, "error": f"HTTP {resp.status_code}", "fetch_source": "api"}
        data = resp.json()
        # FINRA API returns a list; take first result
        if not data:
            return {"cusip": cusip, "error": "empty_response", "fetch_source": "api"}
        item = data[0] if isinstance(data, list) else data

        coupon_raw = item.get("couponRate") or item.get("coupon") or 0.0
        maturity_raw = item.get("maturityDate") or item.get("maturity", "")
        outstanding_raw = item.get("issueAmount") or item.get("outstandingAmount") or 0.0

        return {
            "cusip": cusip,
            "issuer": item.get("issuerName") or item.get("issuer", ""),
            "coupon_rate": float(coupon_raw),
            "maturity_date": str(maturity_raw)[:10],
            "outstanding_face_value": float(outstanding_raw) / 1e6,  # convert to MM
            "bond_type": item.get("bondType") or item.get("instrumentType", ""),
            "moody_rating": item.get("moodysRating") or item.get("moodyRating", "NR"),
            "sp_rating": item.get("spRating") or item.get("standardPoorsRating", "NR"),
            "fetch_source": "api",
        }
    except Exception as exc:
        logger.debug("FINRA API exception for %s: %s", cusip, exc)
        return {"cusip": cusip, "error": str(exc), "fetch_source": "api"}


def _fetch_trace_via_scrape(cusip: str) -> dict:
    """Fall-back: parse FINRA bond search page with BeautifulSoup."""
    url = f"https://www.finra.org/investors/bonds/bond-search?cusip={cusip}"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research bot; contact: research@example.com)"}
        resp = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(resp.text, "lxml")
        rows = soup.find_all("tr")
        data_map: dict = {}
        for row in rows:
            cells = row.find_all("td")
            if len(cells) == 2:
                key = cells[0].get_text(strip=True)
                val = cells[1].get_text(strip=True)
                data_map[key.lower()] = val

        coupon = 0.0
        for k in ("coupon", "coupon rate", "interest rate"):
            if k in data_map:
                try:
                    coupon = float(data_map[k].replace("%", "").strip())
                except ValueError:
                    pass

        maturity = ""
        for k in ("maturity date", "maturity", "due date"):
            if k in data_map:
                maturity = data_map[k][:10]

        return {
            "cusip": cusip,
            "issuer": data_map.get("issuer name", data_map.get("issuer", "")),
            "coupon_rate": coupon,
            "maturity_date": maturity,
            "outstanding_face_value": 0.0,  # not reliably available via scrape
            "bond_type": data_map.get("bond type", data_map.get("type", "")),
            "moody_rating": data_map.get("moody's", "NR"),
            "sp_rating": data_map.get("s&p", "NR"),
            "fetch_source": "scrape",
        }
    except Exception as exc:
        logger.warning("FINRA scrape failed for %s: %s", cusip, exc)
        return {"cusip": cusip, "error": str(exc), "fetch_source": "none"}


# ---------------------------------------------------------------------------
# 5. Bond Stack Builder
# ---------------------------------------------------------------------------

def build_bond_stack(ticker: str, cik: str) -> pd.DataFrame:
    """
    Assemble a per-issuer bond stack combining XBRL and TRACE data.

    Merges the XBRL annual maturity schedule (Y1–Y5+) with bond-level
    reference data from TRACE where available.  For companies with no TRACE
    CUSIPs, the bond stack is populated from XBRL only.

    Parameters
    ----------
    ticker : str
        Company ticker (used for file naming and logging).
    cik : str
        SEC CIK number.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``cik``, ``cusip_if_known``,
        ``maturity_bucket`` (Y1/Y2/…/Y5plus), ``debt_amount_mm``,
        ``coupon_rate_if_known``, ``maturity_date_if_known``,
        ``data_source`` (``"xbrl_only"``, ``"xbrl_plus_trace"``,
        ``"trace_only"``).

    Notes
    -----
    Saved to ``data/processed/{ticker}_bond_stack.csv``.
    """
    from src.trace_parser import build_issuer_bond_stack
    from src.xbrl_parser import (
        extract_debt_maturity_facts,
        build_annual_maturity_schedule,
    )

    out_path = config.PROCESSED_DIR / f"{ticker}_bond_stack.csv"
    if out_path.exists():
        logger.info("Bond stack cache hit: %s", out_path)
        return pd.read_csv(out_path, parse_dates=["maturity_date_if_known"])

    # -- XBRL schedule --
    try:
        tag_taxonomy = pd.read_csv(config.XBRL_TAG_TAXONOMY_CSV)
        xbrl_facts = fetch_company_xbrl_facts(cik)
        raw_facts_df = extract_debt_maturity_facts(xbrl_facts, tag_taxonomy)
        schedule_df = build_annual_maturity_schedule(cik, raw_facts_df, pd.DataFrame())
        xbrl_rows = []
        if not schedule_df.empty:
            latest = schedule_df.sort_values("filing_date").iloc[-1]
            for bucket in ["Y1", "Y2", "Y3", "Y4", "Y5", "Y5plus"]:
                amt = latest.get(bucket, 0.0) or 0.0
                if amt > 0:
                    xbrl_rows.append({
                        "ticker": ticker,
                        "cik": cik,
                        "cusip_if_known": None,
                        "maturity_bucket": bucket,
                        "debt_amount_mm": float(amt),
                        "coupon_rate_if_known": None,
                        "maturity_date_if_known": None,
                        "data_source": "xbrl_only",
                    })
    except Exception as exc:
        logger.warning("XBRL extraction failed for %s/%s: %s", ticker, cik, exc)
        xbrl_rows = []

    # -- TRACE enrichment --
    try:
        trace_df = build_issuer_bond_stack(cik, ticker)
        if not trace_df.empty:
            for _, bond in trace_df.iterrows():
                if not bond.get("is_matured", True):
                    xbrl_rows.append({
                        "ticker": ticker,
                        "cik": cik,
                        "cusip_if_known": bond["cusip"],
                        "maturity_bucket": None,  # filled by trace_parser
                        "debt_amount_mm": bond.get("outstanding_face_mm", 0.0),
                        "coupon_rate_if_known": bond.get("coupon_rate"),
                        "maturity_date_if_known": bond.get("maturity_date"),
                        "data_source": "xbrl_plus_trace" if xbrl_rows else "trace_only",
                    })
    except Exception as exc:
        logger.warning("TRACE enrichment failed for %s/%s: %s", ticker, cik, exc)

    bond_stack_df = pd.DataFrame(xbrl_rows)
    if bond_stack_df.empty:
        logger.warning("Empty bond stack for %s — no XBRL or TRACE data", ticker)
    else:
        bond_stack_df.to_csv(out_path, index=False)
        logger.info("Bond stack saved: %s (%d rows)", out_path, len(bond_stack_df))
    return bond_stack_df

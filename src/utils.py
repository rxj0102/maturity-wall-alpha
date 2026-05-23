"""
utils.py
========
Shared utility functions and classes for the maturity-wall strategy.

Contents
--------
- RateLimiter    : Token-bucket rate limiter (context manager) for EDGAR API
- lag_dataframe  : Look-ahead bias prevention wrapper
- get_quarter_label / get_quarter_end : Quarter string <-> date conversion
- months_between : Precise month difference using dateutil.relativedelta
- winsorize_series : Clip outliers at given quantiles
- setup_logging  : Consistent log format across all modules
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Optional, List

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure a module-level logger with a consistent format.

    Parameters
    ----------
    level : int
        Logging level (e.g. ``logging.DEBUG``, ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Token-bucket rate limiter for HTTP requests.

    Designed primarily for the SEC EDGAR API (max 10 requests/second) but
    configurable for any endpoint.  Can be used as a plain callable or as a
    context manager.

    Parameters
    ----------
    max_calls : int
        Maximum number of calls allowed per ``period`` seconds.
    period : float
        Time window in seconds.  Default is 1.0 (i.e., calls per second).
    min_delay : float
        Minimum sleep between calls in seconds (SEC EDGAR rate-limit guidance).

    Examples
    --------
    >>> limiter = RateLimiter(max_calls=10, period=1.0, min_delay=0.15)
    >>> with limiter:
    ...     response = requests.get(url)

    Or as a plain call:

    >>> limiter()
    >>> response = requests.get(url)
    """

    def __init__(
        self,
        max_calls: int = 10,
        period: float = 1.0,
        min_delay: float = 0.15,
    ) -> None:
        self.max_calls = max_calls
        self.period = period
        self.min_delay = min_delay
        self._lock = threading.Lock()
        self._calls: List[float] = []

    def __call__(self) -> None:
        """Block until a token is available, then consume one token."""
        with self._lock:
            now = time.monotonic()
            # Remove timestamps older than the period window
            self._calls = [t for t in self._calls if now - t < self.period]
            if len(self._calls) >= self.max_calls:
                sleep_for = self.period - (now - self._calls[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
            # Enforce minimum inter-request delay
            if self._calls:
                elapsed = time.monotonic() - self._calls[-1]
                if elapsed < self.min_delay:
                    time.sleep(self.min_delay - elapsed)
            self._calls.append(time.monotonic())

    @contextmanager
    def __enter__(self):  # type: ignore[return]
        self()
        return self

    def __exit__(self, *args: object) -> None:  # noqa: D401
        pass


# Singleton limiter for EDGAR (module-level; imported by data_loader and xbrl_parser)
edgar_limiter = RateLimiter(max_calls=8, period=1.0, min_delay=0.15)


# ---------------------------------------------------------------------------
# Look-ahead Bias Guard
# ---------------------------------------------------------------------------

def lag_dataframe(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """
    Shift a DataFrame forward by ``periods`` to prevent look-ahead bias.

    Mandatory wrapper before any signal data enters a backtest function.
    Applies ``DataFrame.shift(periods)`` on all columns, which introduces
    ``NaN`` for the first ``periods`` rows.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data indexed by date (or any ordered index).
    periods : int
        Number of periods to lag.  Default is 1 (one period / one quarter).

    Returns
    -------
    pd.DataFrame
        Lagged copy of ``df`` with same index.

    Notes
    -----
    This function operates on the assumption that the DataFrame's index is
    time-ordered.  For a quarterly signal panel, ``periods=1`` means the
    signal from Q4 2022 is first available for trading in Q1 2023.
    """
    if periods < 0:
        raise ValueError(f"lag_dataframe: periods must be >= 0, got {periods}")
    return df.shift(periods)


# ---------------------------------------------------------------------------
# Quarter Utilities
# ---------------------------------------------------------------------------

def get_quarter_label(date: pd.Timestamp) -> str:
    """
    Return the quarter label (e.g. ``"2023Q2"``) for a given date.

    Parameters
    ----------
    date : pd.Timestamp
        Any date within the desired quarter.

    Returns
    -------
    str
        Quarter label in the format ``"{YYYY}Q{q}"``.

    Examples
    --------
    >>> get_quarter_label(pd.Timestamp("2023-05-15"))
    '2023Q2'
    """
    ts = pd.Timestamp(date)
    return f"{ts.year}Q{ts.quarter}"


def get_quarter_end(label: str) -> pd.Timestamp:
    """
    Return the last calendar day of the quarter identified by ``label``.

    Parameters
    ----------
    label : str
        Quarter label in the format ``"{YYYY}Q{q}"`` (e.g. ``"2023Q2"``).

    Returns
    -------
    pd.Timestamp
        Last day of the specified quarter.

    Raises
    ------
    ValueError
        If ``label`` does not match the expected format.

    Examples
    --------
    >>> get_quarter_end("2023Q2")
    Timestamp('2023-06-30 00:00:00')
    """
    try:
        year, qstr = label.split("Q")
        quarter = int(qstr)
        year = int(year)
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid quarter label: '{label}'.  Expected format: '2023Q2'")
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"Quarter must be 1–4, got {quarter}")
    month_end = quarter * 3
    # pandas Period gives us the last day easily
    return pd.Period(f"{year}Q{quarter}", freq="Q").to_timestamp(how="end").normalize()


def get_all_quarter_ends(start: str, end: str) -> List[pd.Timestamp]:
    """
    Return a list of quarter-end dates between ``start`` and ``end``.

    Parameters
    ----------
    start : str
        ISO date string for the start of the range (inclusive).
    end : str
        ISO date string for the end of the range (inclusive).

    Returns
    -------
    list of pd.Timestamp
        Quarter-end dates (March 31, June 30, September 30, December 31).
    """
    idx = pd.date_range(start=start, end=end, freq="QE")
    return list(idx)


# ---------------------------------------------------------------------------
# Month Arithmetic
# ---------------------------------------------------------------------------

def months_between(date1: pd.Timestamp, date2: pd.Timestamp) -> float:
    """
    Compute precise month difference between two dates using ``relativedelta``.

    Parameters
    ----------
    date1 : pd.Timestamp
        Earlier date (start).
    date2 : pd.Timestamp
        Later date (end).

    Returns
    -------
    float
        Fractional months from ``date1`` to ``date2``.  Positive when
        ``date2 > date1``.

    Notes
    -----
    Uses ``dateutil.relativedelta`` for the integer part and adds the
    remaining days as a fractional month (days / 30.4375).  This is more
    accurate than simply dividing elapsed days by 30 for the decay-weight
    computation, where even a few days of imprecision matters at very short
    horizons.
    """
    d1 = pd.Timestamp(date1)
    d2 = pd.Timestamp(date2)
    rd = relativedelta(d2, d1)
    integer_months = rd.years * 12 + rd.months
    fractional = rd.days / 30.4375
    return integer_months + fractional


# ---------------------------------------------------------------------------
# Winsorisation
# ---------------------------------------------------------------------------

def winsorize_series(
    s: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """
    Clip a Series at the given quantile bounds to handle outliers.

    Parameters
    ----------
    s : pd.Series
        Input numeric series.
    lower : float
        Lower quantile bound (default 1st percentile).
    upper : float
        Upper quantile bound (default 99th percentile).

    Returns
    -------
    pd.Series
        Winsorised series with the same index and name as ``s``.

    Notes
    -----
    Applied to ``stress_score`` before cross-sectional ranking to prevent
    data-error outliers (e.g. coverage_ratio = 0.01 companies) from
    dominating the top-decile portfolio.
    """
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


# ---------------------------------------------------------------------------
# Exponential Backoff Helper
# ---------------------------------------------------------------------------

def exponential_backoff(attempt: int, base: float = 2.0, max_wait: float = 30.0) -> float:
    """
    Compute wait time for exponential backoff.

    Parameters
    ----------
    attempt : int
        Zero-indexed retry attempt number.
    base : float
        Backoff base in seconds (default 2.0: 2s, 4s, 8s, …).
    max_wait : float
        Maximum wait cap in seconds.

    Returns
    -------
    float
        Seconds to sleep before the next retry.
    """
    return min(base ** (attempt + 1), max_wait)


# ---------------------------------------------------------------------------
# Forward-return Helper
# ---------------------------------------------------------------------------

def compute_forward_returns(
    prices: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Compute forward returns over a fixed horizon for a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices; index=date, columns=tickers.
    horizon_days : int
        Calendar days to look forward.

    Returns
    -------
    pd.DataFrame
        Forward returns; same shape as ``prices``.  NaN for the last
        ``horizon_days`` rows (no future price available).

    Notes
    -----
    Uses ``shift(-horizon_days)`` on business-day resampled prices to align
    future prices with the current date row.
    """
    fwd_prices = prices.shift(-horizon_days)
    return (fwd_prices / prices) - 1.0

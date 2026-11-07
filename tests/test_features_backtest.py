"""Offline tests: point-in-time maturity features, decay weights and backtest mechanics."""

import math

import numpy as np
import pandas as pd
import pytest

import config
from src import backtest as bt
from src.features import (
    compute_maturity_concentration_score,
    compute_months_to_peak_maturity,
    compute_signal_decay_weight,
)
from src.utils import compute_forward_returns, get_all_quarter_ends, get_quarter_end, get_quarter_label


# ----------------------------------------------------------------- features
def _schedule(filing_date, amounts, cik="1", ticker="AAA", start="2023-03-31"):
    qes = pd.date_range(start, periods=len(amounts), freq="QE")
    return pd.DataFrame(
        {
            "cik": cik,
            "ticker": ticker,
            "filing_date": pd.Timestamp(filing_date),
            "quarter_end_date": qes,
            "quarter_label": [get_quarter_label(q) for q in qes],
            "debt_maturing_mm": amounts,
        }
    )


def test_mcs_is_peak_quarter_share_of_total():
    sched = _schedule("2023-02-15", [100.0, 700.0, 100.0, 100.0])
    out = compute_maturity_concentration_score(sched, pd.Timestamp("2023-03-31"))
    row = out.iloc[0]
    assert row["MCS"] == pytest.approx(0.7)
    assert row["mcs_bucket"] == "EXTREME"
    assert row["total_debt_outstanding_mm"] == pytest.approx(1000.0)
    assert row["peak_quarter_label"] == get_quarter_label(pd.Timestamp("2023-06-30"))


def test_mcs_ignores_filings_not_yet_public():
    """A filing dated after as_of - lag must not influence the snapshot (point in time)."""
    old = _schedule("2023-02-01", [100.0, 100.0, 100.0, 100.0])
    new = _schedule("2023-03-30", [900.0, 10.0, 10.0, 10.0])  # filed 1 day before as_of: inside the lag
    both = pd.concat([old, new], ignore_index=True)
    as_of = pd.Timestamp("2023-03-31")
    a = compute_maturity_concentration_score(old, as_of).iloc[0]
    b = compute_maturity_concentration_score(both, as_of).iloc[0]
    assert b["MCS"] == a["MCS"] == pytest.approx(0.25)


def test_mcs_uses_new_filing_once_lag_has_elapsed():
    old = _schedule("2023-02-01", [100.0, 100.0, 100.0, 100.0])
    new = _schedule("2023-03-01", [900.0, 10.0, 10.0, 10.0])
    out = compute_maturity_concentration_score(pd.concat([old, new]), pd.Timestamp("2023-03-31"))
    assert out.iloc[0]["MCS"] == pytest.approx(900 / 930, abs=1e-4)


def test_mcs_zero_debt_flagged_as_data_error():
    out = compute_maturity_concentration_score(_schedule("2023-02-01", [0.0, 0.0]), pd.Timestamp("2023-03-31"))
    assert bool(out.iloc[0]["data_error"]) and out.iloc[0]["MCS"] == 0.0


def test_proximity_labels():
    df = pd.DataFrame({"months_to_peak_maturity": [3, 12, 30, 48, np.nan]})
    out = compute_months_to_peak_maturity(df, pd.Timestamp("2023-03-31"))
    assert out["proximity_label"].tolist() == ["NEAR", "MEDIUM", "WATCH", "DISTANT", "UNKNOWN"]


def test_decay_weight_shape():
    w0 = compute_signal_decay_weight(0.0)
    w_med = compute_signal_decay_weight(config.MONTHS_MEDIUM_THRESHOLD)
    w_far = compute_signal_decay_weight(config.MONTHS_MEDIUM_THRESHOLD + config.SIGNAL_DECAY_HALFLIFE)
    assert w_med == pytest.approx(1.0)
    assert w_far == pytest.approx(0.5)  # one half-life further out halves the weight
    assert w0 > w_med > w_far > 0
    assert compute_signal_decay_weight(-3.0) == pytest.approx(w0)  # past peak treated as imminent
    assert compute_signal_decay_weight(np.nan) == pytest.approx(w0)


# ------------------------------------------------------------------- utils
def test_quarter_helpers_round_trip():
    ts = pd.Timestamp("2022-11-15")
    assert get_quarter_end(get_quarter_label(ts)) == pd.Timestamp("2022-12-31")
    assert get_all_quarter_ends("2022-01-01", "2022-12-31") == list(pd.date_range("2022-03-31", periods=4, freq="QE"))


def test_forward_returns_look_forward_only():
    px = pd.DataFrame({"A": np.arange(1.0, 11.0)}, index=pd.bdate_range("2023-01-02", periods=10))
    fwd = compute_forward_returns(px, 2)
    assert fwd["A"].iloc[0] == pytest.approx(3.0 / 1.0 - 1)
    assert fwd["A"].iloc[-2:].isna().all()


# ---------------------------------------------------------------- backtest
def test_cost_stack_short_vs_long():
    net, tc, borrow = bt._apply_cost_stack(0.05, is_short=True)
    assert tc == pytest.approx(config.TRANSACTION_COST_BPS / 1e4)
    assert borrow == pytest.approx(config.SHORT_BORROW_COST_BPS / 1e4 / 4)
    assert net == pytest.approx(0.05 - tc - borrow)
    net_l, _, b_l = bt._apply_cost_stack(0.05, is_short=False)
    assert b_l == 0.0 and net_l > net


def test_drawdown_series_matches_peak_fraction():
    dd = bt._compute_drawdown_series(pd.Series([1.0, 1.5, 2.0, 1.0, 1.2]))
    assert dd.iloc[3] == pytest.approx(-0.5) and dd.iloc[2] == 0


def test_quarterly_returns_are_forward_from_entry_label():
    idx = pd.bdate_range("2023-01-02", "2023-12-29")
    px = pd.DataFrame({"A": np.linspace(100, 200, len(idx))}, index=idx)
    qes = get_all_quarter_ends("2023-03-01", "2023-12-31")
    qr = bt._get_quarterly_returns(px, qes)
    entry = qes[0]
    e = px.index[px.index.searchsorted(entry)]
    x = px.index[px.index.searchsorted(qes[1])]
    assert qr.loc[entry, "A"] == pytest.approx(px.loc[x, "A"] / px.loc[e, "A"] - 1)


def test_short_only_backtest_shorts_critical_names_and_charges_costs(monkeypatch):
    monkeypatch.setattr(config, "BACKTEST_START", "2023-03-01")
    monkeypatch.setattr(config, "BACKTEST_END", "2023-12-31")
    idx = pd.bdate_range("2023-01-02", "2023-12-29")
    n = len(idx)
    px = pd.DataFrame({"LOSER": 100 * 0.999 ** np.arange(n), "WINNER": 100 * 1.001 ** np.arange(n)}, index=idx)
    qes = get_all_quarter_ends("2023-03-01", "2023-12-31")
    sig = pd.DataFrame(
        [(q, t, "CRITICAL" if t == "LOSER" else "LOW", 1.0) for q in qes for t in ("LOSER", "WINNER")],
        columns=["as_of_date", "ticker", "stress_tier", "stress_score"],
    )
    res = bt.run_short_only_backtest(sig, px)
    active = res[res["n_shorts"] > 0]
    assert len(active) >= 2
    assert (active["short_return"] > 0).all()  # shorting a decliner earns
    assert (active["net_return"] < active["gross_return"]).all()  # costs applied
    assert active["tc_drag"].iloc[0] == pytest.approx(config.TRANSACTION_COST_BPS / 1e4)

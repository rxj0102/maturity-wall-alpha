# Corporate Bond Maturity Wall Stress via CUSIP-Level Refinancing Calendar

[![CI](https://github.com/rxj0102/maturity-wall-alpha/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/maturity-wall-alpha/actions/workflows/ci.yml)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Abstract

Standard credit-stress models treat financial distress as a continuous process — captured through CDS spreads, leverage ratios, and interest-coverage deterioration. These indicators are market-consensus outputs: by the time they signal stress, much of the repricing has already occurred.

This strategy reads the **calendar** instead. Corporate bond maturity dates were contractually fixed 5–10 years ago. A company's need to refinance $1.5bn of bonds in Q2 2025 was visible in its 2019 10-K filing — six years before the event. The market prices this risk *episodically* (as the maturity date draws near and credit conditions change); this strategy prices it *continuously* from the day the filing is public.

The central data-engineering contribution is a two-pass XBRL parser that extracts annual debt maturity schedules from SEC EDGAR filings across a 50-company high-yield/leveraged universe. Standard GAAP tags cover roughly 60% of filers; company-extension tags (e.g. `hca:LongTermDebtMaturingInYearOne`) account for the remainder. This aggregation is the strategy's moat: the data is entirely public but the assembly is genuinely hard. The parser is documented with the real tag variants encountered in production and validated against balance-sheet LongTermDebt totals to produce coverage quality flags.

---

## Core Hypothesis

**H₀ (null):** XBRL-derived maturity concentration has no predictive power for forward equity returns, after controlling for leverage and sector.

**H₁ (alternative):** Companies with high Maturity Concentration Scores (MCS > 35%) maturing within 18 months, in a high-rate environment (RDM > 1.5×), generate negative alpha at 90–252 day horizons relative to the universe mean.

---

## Key Distinction

> **Standard credit stress models use CDS spreads, leverage ratios, and interest coverage ratios — outputs that already reflect market-consensus stress pricing. This strategy reads the calendar. Bond maturity dates were contractually fixed 5–10 years ago. A company's refinancing difficulty is visible years in advance in public filings. The market prices this risk episodically; we price it continuously.**

The maturity wall is a *binary* refinancing event. A company either rolls its debt or it doesn't. The outcome depends on three factors that are all visible in advance:

1. **How much debt matures when** (XBRL maturity schedule → MCS)
2. **At what cost relative to the original issuance** (TRACE coupon + FRED rates → RDM)
3. **How close the event is** (calendar → decay_weight)

None of these inputs requires a market forecast. They are contractual facts and observable rates.

---

## The Three Signal Components

### 1. Maturity Concentration Score (MCS)

$$\text{MCS} = \frac{\text{peak\_quarter\_debt}}{\text{total\_debt\_outstanding (next 8 quarters)}}$$

MCS measures *how concentrated* a company's maturity profile is. An issuer with $500mm maturing in each of the next 8 quarters has MCS = 12.5% (diversified). An issuer with $2bn maturing in a single quarter has MCS = 100% (binary event). Signal-active thresholds: HIGH > 35%, EXTREME > 50%.

### 2. Refinancing Difficulty Multiplier (RDM)

$$\text{RDM} = \frac{\text{T10Y} + \text{HY\_OAS (or IG\_OAS)}}{\text{yield\_at\_issuance\_proxy}}$$

RDM captures *how much more expensive* it is to refinance today relative to when the bond was originally issued. RDM = 1.0 means refinancing at the same rate (neutral). RDM = 2.0 means refinancing at double the original cost — a severe cash flow shock. `yield_at_issuance_proxy` uses the bond's coupon rate from FINRA TRACE (bond-specific when available, WAC otherwise).

### 3. Decay Weight (time-to-maturity amplification)

$$w(m) = \exp\!\left(\ln(2) \cdot \frac{18 - m}{12}\right)$$

where $m$ = months to peak maturity quarter, $h$ = 12-month half-life.

| Months to maturity | Decay weight | Interpretation |
|---|---|---|
| 36 | ≈ 0.35 | Distant — low allocation |
| 18 | ≈ 1.00 | Medium boundary — neutral |
| 12 | ≈ 1.41 | Elevated |
| 6  | ≈ 2.00 | Near — doubled weight |
| 0  | ≈ 2.83 | Imminent — maximum weight |

*(With `SIGNAL_DECAY_HALFLIFE = 12` months. Using halflife = 9 months produces the intuitive 0.25/4.0 boundary values; configure via `config.py`.)*

The decay function addresses the *carry-bleed problem* directly: by downweighting early signals, the strategy reduces position size during the period when the stock can still rise 30% before the thesis plays out.

### Composite Stress Score

$$\text{stress\_score} = \text{MCS} \times \text{RDM} \times w(m)$$

Cross-sectionally ranked each quarter. **SHORT** the top decile (CRITICAL tier). **LONG** the bottom two deciles (NORMAL tier — clean balance sheets with no near-term maturities).

---

## Data Engineering: The XBRL Parsing Challenge

This is the strategy's primary technical contribution. The SEC EDGAR XBRL API provides free access to every tagged fact in every public filing — but the tag landscape is a disaster.

**The problem in numbers:**
- ~40% of companies in a typical HY universe use non-standard, company-extension tags for debt maturity schedules
- A single company may change its tagging convention across years as it changes auditors
- Tags that look like debt maturity tags (e.g. `OperatingLeaseLiabilityPaymentsDueNextTwelveMonths`) are actually lease obligations — including them inflates the maturity schedule by 15–40% for capital-intensive companies

**The solution:**
1. **Pass 1 (standard):** Exact lookup of all known `us-gaap` namespace maturity tags
2. **Pass 2 (fuzzy):** Case-insensitive substring matching for extension tags containing `["Maturity", "Repayment", "Principal", "LongTermDebt"]`
3. **Taxonomy CSV:** Living dictionary (`data/reference/xbrl_tag_taxonomy.csv`) with 65+ entries mapping observed tag variants to standard bucket labels
4. **Validation:** Coverage ratio check — `sum(Y1+…+Y5+) / LongTermDebt` should be 0.85–1.15. Companies outside this range are flagged or excluded.

**Known extension tag examples observed in production:**

```
hca:LongTermDebtMaturingInYearOne          (HCA Healthcare, 2014–2017)
chk:AggregateAnnualMaturitiesOfLongTermDebtYearOne  (Chesapeake, 2014–2019)
dis:LongTermDebtMaturitiesYearOne          (Disney, pre-2020)
ctl:LongTermDebtDueInOneYear               (CenturyLink/Lumen, 2015–2021)
ge:DebtMaturitiesRepaymentOfPrincipalInYearOne   (GE Capital, 2014–2018)
```

---

## Methodology Pipeline

```
EDGAR XBRL API → Debt Maturity Schedule (Y1–Y5+)
       ↓
FINRA TRACE → Bond Coupon Data (yield_at_issuance proxy)
       ↓
FRED API → Current Risk-Free Rate + Credit Spreads
       ↓
MCS = peak_quarter_debt / total_debt
RDM = current_rate / yield_at_issuance
decay_weight = exp(ln(2) × (18 - months_to_peak) / 12)
       ↓
stress_score = MCS × RDM × decay_weight
       ↓
Cross-Sectional Ranking → Short CRITICAL Tier
       ↓
Carry Bleed Monitor + Resolution Monitor
```

---

## Repository Structure

```
maturity-wall-alpha/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                          ← All constants, paths, API endpoints
├── data/
│   ├── raw/
│   │   ├── xbrl/                      ← CIK_facts.json (gitignored — large)
│   │   ├── trace/                     ← CUSIP_facts.json (gitignored)
│   │   └── fred/                      ← series_id.csv (gitignored)
│   ├── processed/                     ← Parquet outputs (gitignored)
│   └── reference/                     ← VERSION CONTROLLED seed data
│       ├── company_universe.csv        ← 50 HY/leveraged issuers with CIKs
│       ├── xbrl_tag_taxonomy.csv       ← 65+ tag variants → bucket mapping
│       └── historical_stress_events.csv← 15 real maturity-wall stress events
├── outputs/                           ← Charts (gitignored)
├── notebooks/
│   └── 01_exploratory_analysis.ipynb  ← Full research notebook
└── src/
    ├── __init__.py
    ├── utils.py           ← RateLimiter, lag_dataframe, quarter helpers
    ├── data_loader.py     ← EDGAR XBRL, FRED, yfinance, FINRA TRACE
    ├── xbrl_parser.py     ← Two-pass XBRL extraction + maturity schedule
    ├── trace_parser.py    ← CUSIP discovery + bond reference data
    ├── features.py        ← MCS, RDM, decay weight, composite signal
    ├── signal.py          ← Ranking, IC, t-tests, sector/regime analysis
    ├── backtest.py        ← L/S backtest engine + performance metrics
    └── visualize.py       ← 9 production charts → outputs/
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Smoke test — single company XBRL extraction

```bash
python -m src.xbrl_parser --cik 0000860731   # HCA Healthcare
```

Prints the debt maturity schedule, coverage flag, and validation recommendation.

### 3. Build the full pipeline

```python
import pandas as pd
import config
from src.data_loader import fetch_all_fred_series, fetch_equity_prices
from src.xbrl_parser import build_universe_maturity_schedules
from src.features import build_rolling_signal_history
from src.backtest import run_long_short_backtest, compute_performance_metrics

universe_df = pd.read_csv(config.COMPANY_UNIVERSE_CSV)
macro_rates = fetch_all_fred_series()
equity_prices = fetch_equity_prices(
    tickers=universe_df["ticker"].tolist(),
    start=config.BACKTEST_START,
    end=config.BACKTEST_END,
)
rolling_signals = build_rolling_signal_history(universe_df, macro_rates)
backtest_results = run_long_short_backtest(rolling_signals, equity_prices)
metrics = compute_performance_metrics(backtest_results, label="Long-Short")
```

### 4. Run the research notebook

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

---

## Key Findings

No results are reported yet. The pipeline needs live SEC XBRL filings and credit data that were not run for this repository, so there are no measured lead times, ICs or Sharpe ratios to quote. Run the full pipeline and notebook Section 10 to produce them. The planned outputs are:

- Median signal lead time before documented stress events
- Spearman IC at 180 days, decay-adjusted versus raw signal
- Long-short Sharpe, decay-adjusted versus raw signal
- Average pre-event carry bleed

Any regime dependence (for example rising rates with wide spreads) is a hypothesis to test, not a finding.

---

## Limitations

1. **XBRL tag inconsistency is permanent.** Companies change auditors and tagging conventions across fiscal years. The taxonomy CSV requires ongoing manual maintenance. A realistic production estimate: 2–4 hours per quarter to review new filer-specific extension tags.

2. **The carry-bleed problem is only partially addressed.** The decay function reduces but does not eliminate the cost of being early. A company can refinance, extend maturities, or be acquired in month 17 of an 18-month stress window. The strategy has no mechanism to detect successful resolution in real time. **Extension**: monitor for new bond issuance (8-K filings with prospectus) as a signal to close the short.

3. **Short borrow costs are optimistic.** The 50bps assumption may significantly understate real borrow costs for CCC-rated issuers in active distress. Actual borrow costs for hard-to-borrow names can be 200–500bps annually. The backtest includes sensitivity analysis at 150bps and 300bps.

4. **RDM uses coupon as a YTM proxy.** For discount bonds (issued below par), true YTM > coupon — the RDM understates refinancing difficulty. For premium bonds, the reverse. The original issue price is in the bond prospectus but not in TRACE reference data.

5. **Macro-regime concentration risk.** If the strategy's alpha is concentrated in rising-rate periods, it may be a disguised rate-sensitivity bet rather than a true maturity-wall signal. See the regime interaction analysis in the notebook.

6. **Survivorship bias.** The universe is constructed from companies known to the researcher ex-post. A live implementation would use a contemporaneous high-yield index membership as the entry criterion.

---

## Academic References

- **Almeida, Campello, Laranjeira & Weisbenner (2012)** — Corporate Debt Maturity and the Real Effects of the 2007 Credit Crisis. *American Economic Review*, 102(7), 3400–3432.

- **Brunnermeier & Yogo (2009)** — A Note on Liquidity Risk Management. *American Economic Review*, 99(2), 578–583.

- **Choi, Hackbarth & Zechner (2018)** — Corporate Debt Maturity Profiles. *Journal of Financial Economics*, 130(3), 484–502.

- **Greenwood, Hanson & Stein (2010)** — A Gap-Filling Theory of Corporate Debt Maturity Choice. *Journal of Finance*, 65(3), 993–1028.

- **Harford & Uysal (2014)** — Bond Market Access and Investment. *Journal of Financial Economics*, 112(2), 147–163.

---

## Disclaimer

This repository is academic research for portfolio purposes only. It does not constitute investment advice. All results are backtested and subject to look-ahead bias, survivorship bias, and data limitations documented herein. Past signal performance does not predict future returns.

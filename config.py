"""
config.py
=========
Central configuration for the Corporate Bond Maturity Wall Stress strategy.

All constants, API endpoints, signal parameters, backtest settings, and path
definitions live here.  No other module should hardcode paths or thresholds.
"""

from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Repository Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.resolve()

DATA_DIR          = REPO_ROOT / "data"
RAW_DIR           = DATA_DIR / "raw"
RAW_XBRL_DIR      = RAW_DIR / "xbrl"
RAW_TRACE_DIR     = RAW_DIR / "trace"
RAW_FRED_DIR      = RAW_DIR / "fred"
PROCESSED_DIR     = DATA_DIR / "processed"
REFERENCE_DIR     = DATA_DIR / "reference"
OUTPUTS_DIR       = REPO_ROOT / "outputs"
NOTEBOOKS_DIR     = REPO_ROOT / "notebooks"
SRC_DIR           = REPO_ROOT / "src"

# Ensure runtime directories exist
for _d in [RAW_XBRL_DIR, RAW_TRACE_DIR, RAW_FRED_DIR, PROCESSED_DIR, OUTPUTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reference File Paths
# ---------------------------------------------------------------------------
COMPANY_UNIVERSE_CSV        = REFERENCE_DIR / "company_universe.csv"
XBRL_TAG_TAXONOMY_CSV       = REFERENCE_DIR / "xbrl_tag_taxonomy.csv"
HISTORICAL_STRESS_EVENTS_CSV = REFERENCE_DIR / "historical_stress_events.csv"

# Processed artefact paths
MACRO_RATES_PARQUET            = PROCESSED_DIR / "macro_rates.parquet"
EQUITY_PRICES_PARQUET          = PROCESSED_DIR / "equity_prices.parquet"
UNIVERSE_SCHEDULES_PARQUET     = PROCESSED_DIR / "universe_maturity_schedules.parquet"
XBRL_QUALITY_REPORT_CSV        = PROCESSED_DIR / "xbrl_quality_report.csv"
STRESS_SIGNALS_PARQUET         = PROCESSED_DIR / "stress_signals.parquet"
ROLLING_SIGNALS_PARQUET        = PROCESSED_DIR / "rolling_stress_signals.parquet"

# ---------------------------------------------------------------------------
# SEC EDGAR Endpoints
# ---------------------------------------------------------------------------
EDGAR_COMPANY_SEARCH  = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?company={name}&CIK=&type=10-K&dateb=&owner=include"
    "&count=10&search_text=&action=getcompany"
)
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_XBRL_FACTS_URL  = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_FILING_VIEWER   = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"

# EDGAR full-text search (for CUSIP discovery)
EDGAR_FULL_TEXT_SEARCH = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%22CUSIP%22&dateRange=custom&startdt={start}&enddt={end}"
    "&entity={cik}&forms=8-K,S-3,424B2,424B3,424B5"
)

# ---------------------------------------------------------------------------
# FRED API Endpoints
# ---------------------------------------------------------------------------
FRED_API_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

FRED_SERIES = {
    "T10Y":   "DGS10",          # 10-year Treasury constant maturity
    "T2Y":    "DGS2",           # 2-year Treasury
    "HY_OAS": "BAMLH0A0HYM2",  # ICE BofA US High Yield Index OAS (free via FRED)
    "IG_OAS": "BAMLC0A0CM",    # ICE BofA US Corporate Index OAS
    "SOFR":   "SOFR",           # Secured Overnight Financing Rate
}

# ---------------------------------------------------------------------------
# FINRA TRACE Endpoints
# ---------------------------------------------------------------------------
FINRA_BOND_FACTS_URL = "https://markets.finra.org/bonds/bondFacts?cusip={cusip}"
FINRA_BOND_SEARCH_URL = "https://www.finra.org/investors/bonds/bond-search"

# ---------------------------------------------------------------------------
# XBRL Tag Variants for Debt Maturity Schedule
# ---------------------------------------------------------------------------
# Standard us-gaap tags recognised in the first parsing pass.
# Companies also use company-specific extension tags (fuzzy-matched in pass 2).
# NOTE: Tags like OperatingLeaseLiabilityPaymentsDueNextTwelveMonths look
#       similar but represent lease obligations, NOT debt — they are explicitly
#       excluded in xbrl_parser.py.
XBRL_MATURITY_TAGS = [
    # Y1 — due in next 12 months
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
    # Y2 – Y5
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour",
    "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive",
    # Y5+
    "LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive",
    # Fiscal-year remainder (used in 10-Q filings)
    "LongTermDebtMaturitiesRepaymentsOfPrincipalRemainderOfFiscalYear",
    # Aggregate / balance sheet level
    "DebtInstrumentMaturityDate",
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "LongTermDebtCurrent",
    "ShortTermBorrowings",
    # Senior notes / debentures
    "SeniorLongTermNotes",
    "UnsecuredLongTermDebt",
    # Finance lease principal
    "FinanceLeaseLiabilityPaymentsDueNextTwelveMonths",
]

# Substrings that trigger fuzzy-match pass 2 for extension tags
XBRL_FUZZY_MATCH_TERMS = [
    "Maturity", "Repayment", "Principal", "LongTermDebt",
    "RepaymentsOfPrincipal", "MaturitiesRepayments",
]

# Substrings that indicate a tag should be EXCLUDED (non-debt obligations)
XBRL_EXCLUDE_TERMS = [
    "OperatingLease", "FinanceLease", "Pension", "PostRetirement",
    "Contingent", "Purchase", "Construction", "Revenue", "Minimum",
]

# ---------------------------------------------------------------------------
# Signal Parameters
# ---------------------------------------------------------------------------
MCS_HIGH_THRESHOLD      = 0.35   # MCS > 35% = concentrated maturity risk
MCS_EXTREME_THRESHOLD   = 0.50   # MCS > 50% = binary refinancing event
RDM_STRESS_THRESHOLD    = 1.50   # RDM > 1.5 = refinancing at 50%+ higher rate
MONTHS_NEAR_THRESHOLD   = 6      # ≤ 6 months = near-term stress
MONTHS_MEDIUM_THRESHOLD = 18     # 6–18 months = medium-term stress watch
SIGNAL_DECAY_HALFLIFE   = 12     # months: half-life of decay weight function
MIN_DEBT_SIZE_MM        = 100    # minimum $100 M total debt to enter universe
LOOKFORWARD_QUARTERS    = 8      # quarters of maturity schedule to extract

# ---------------------------------------------------------------------------
# Backtest Parameters
# ---------------------------------------------------------------------------
BACKTEST_START            = "2014-01-01"
BACKTEST_END              = "2023-12-31"
REBALANCE_FREQUENCY       = "quarterly"
TRANSACTION_COST_BPS      = 15    # round-trip; reflects wider HY bid-ask
SHORT_BORROW_COST_BPS     = 50    # annualised borrow cost for stressed names
SIGNAL_UPDATE_LAG_DAYS    = 5     # business days after 10-K filing date before
                                   # the signal is live (avoid look-ahead bias)

# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------
REQUEST_DELAY_SECONDS = 0.15   # SEC EDGAR policy: ≤ 10 req/sec
MAX_RETRIES           = 3
BACKOFF_BASE_SECONDS  = 2.0    # exponential backoff: 2s, 4s, 8s
FRED_CACHE_TTL_DAYS   = 7      # refresh FRED cache if older than 7 days

# ---------------------------------------------------------------------------
# EDGAR User-Agent (required by SEC robots.txt)
# ---------------------------------------------------------------------------
EDGAR_USER_AGENT = os.getenv(
    "EDGAR_USER_AGENT",
    "maturity-wall-alpha research@example.com",
)
EDGAR_HEADERS = {"User-Agent": EDGAR_USER_AGENT, "Accept-Encoding": "gzip, deflate"}

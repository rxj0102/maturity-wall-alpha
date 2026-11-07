"""
src — Corporate Bond Maturity Wall Stress Strategy
===================================================

Package structure
-----------------
data_loader   : EDGAR XBRL, FRED, TRACE, and yfinance data acquisition
xbrl_parser   : Two-pass XBRL tag extraction and maturity schedule construction
trace_parser  : FINRA TRACE bond reference data and CUSIP discovery
features      : MCS, RDM, decay weight, and composite stress signal
signal        : Cross-sectional ranking, IC tests, sector / regime analysis
backtest      : Long-short backtest engine with cost stack and event overlay
visualize     : Production-quality chart library (outputs/ directory)
utils         : Shared utilities — rate limiter, look-ahead guard, helpers
"""

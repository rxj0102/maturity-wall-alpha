"""Check every CIK in data/reference/company_universe.csv against its SEC EDGAR filer name (including former names).

Output: results/universe_check.csv. A row passes if a distinctive word of the listed company_name appears in the
current or a former EDGAR name. Usage: EDGAR_USER_AGENT="Name email" python scripts/verify_universe.py
"""
import json, os, re, time, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": os.getenv("EDGAR_USER_AGENT", "maturity-wall-alpha research")}
STOP = {"inc", "corp", "co", "the", "company", "group", "holdings", "corporation", "plc", "trust", "&", "associates", "and"}


def words(s):
    return {w for w in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if w not in STOP and len(w) > 2}


def main():
    u = pd.read_csv(ROOT / "data" / "reference" / "company_universe.csv", comment="#", dtype={"cik": str})
    rows = []
    for _, r in u.iterrows():
        cik = r.cik.zfill(10)
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=UA), timeout=30))
            names = [d["name"]] + [f["name"] for f in d.get("formerNames", [])]
        except Exception as e:
            names = [f"lookup failed: {e}"]
        time.sleep(0.12)
        ok = any(words(r.company_name) & words(n) for n in names)
        rows.append({"ticker": r.ticker, "company_name": r.company_name, "cik": cik, "edgar_name": names[0], "edgar_former_names": "; ".join(names[1:4]), "match": ok})
    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "universe_check.csv", index=False)
    print(f"{int(out.match.sum())} of {len(out)} rows match EDGAR")
    print(out[~out.match].to_string())


if __name__ == "__main__":
    main()

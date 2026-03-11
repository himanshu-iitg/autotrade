"""Download and manage NSE/BSE stock universe CSVs."""
import os
import requests
import pandas as pd
from config import NSE_CSV_PATH, BSE_CSV_PATH, DATA_DIR, NIFTY500_CSV_PATH

NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
BSE_COMPANY_LIST_URL = "https://www.bseindia.com/downloads1/List_of_companies.csv"
NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.nseindia.com/",
}


def download_nse_list() -> bool:
    """Download NSE equity list CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        r = requests.get(NSE_EQUITY_LIST_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
        with open(NSE_CSV_PATH, "wb") as f:
            f.write(r.content)
        print(f"  NSE equity list saved: {NSE_CSV_PATH}")
        return True
    except Exception as e:
        print(f"  Warning: Could not download NSE list: {e}")
        return False


def download_bse_list() -> bool:
    """Download BSE company list CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        r = requests.get(BSE_COMPANY_LIST_URL, headers={**HEADERS, "Referer": "https://www.bseindia.com/"}, timeout=30)
        r.raise_for_status()
        with open(BSE_CSV_PATH, "wb") as f:
            f.write(r.content)
        print(f"  BSE company list saved: {BSE_CSV_PATH}")
        return True
    except Exception as e:
        print(f"  Warning: Could not download BSE list: {e}")
        return False


def load_nse_stocks() -> pd.DataFrame:
    """Load NSE stock list into a DataFrame. Downloads if not present."""
    if not os.path.exists(NSE_CSV_PATH):
        download_nse_list()

    if not os.path.exists(NSE_CSV_PATH):
        return pd.DataFrame()

    try:
        df = pd.read_csv(NSE_CSV_PATH)
        # Normalize column names
        df.columns = [c.strip().upper() for c in df.columns]

        # NSE CSV columns: SYMBOL, NAME OF COMPANY, SERIES, DATE OF LISTING, PAID UP VALUE, MARKET LOT, ISIN NUMBER, FACE VALUE
        rename = {}
        for col in df.columns:
            if "SYMBOL" in col:
                rename[col] = "SYMBOL"
            elif "NAME" in col or "COMPANY" in col:
                rename[col] = "COMPANY_NAME"
            elif "ISIN" in col:
                rename[col] = "ISIN"
            elif "SERIES" in col:
                rename[col] = "SERIES"
        df = df.rename(columns=rename)

        # Keep only EQ series (equity shares)
        if "SERIES" in df.columns:
            df = df[df["SERIES"].str.strip() == "EQ"]

        # Add .NS suffix for yfinance
        df["YF_TICKER"] = df["SYMBOL"].str.strip() + ".NS"
        df["EXCHANGE"] = "NSE"
        return df[["SYMBOL", "COMPANY_NAME", "YF_TICKER", "EXCHANGE"]].dropna(subset=["SYMBOL"])
    except Exception as e:
        print(f"  Warning: Could not parse NSE CSV: {e}")
        return pd.DataFrame()


def load_bse_stocks() -> pd.DataFrame:
    """Load BSE stock list. Downloads if not present."""
    if not os.path.exists(BSE_CSV_PATH):
        download_bse_list()

    if not os.path.exists(BSE_CSV_PATH):
        return pd.DataFrame()

    try:
        df = pd.read_csv(BSE_CSV_PATH, encoding="latin-1")
        df.columns = [c.strip().upper() for c in df.columns]

        rename = {}
        for col in df.columns:
            if col in ("SECURITY_CODE", "SCRIP_CODE", "CODE"):
                rename[col] = "BSE_CODE"
            elif "SECURITY_NAME" in col or "COMPANY_NAME" in col or "SCRIP" in col:
                rename[col] = "COMPANY_NAME"
            elif "INDUSTRY" in col or "SECTOR" in col:
                rename[col] = "SECTOR"
            elif "ISIN" in col:
                rename[col] = "ISIN"
        df = df.rename(columns=rename)
        df["EXCHANGE"] = "BSE"
        return df
    except Exception as e:
        print(f"  Warning: Could not parse BSE CSV: {e}")
        return pd.DataFrame()


def download_nifty500_list() -> bool:
    """Download Nifty 500 constituent list CSV from NSE."""
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        r = requests.get(NIFTY500_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
        with open(NIFTY500_CSV_PATH, "wb") as f:
            f.write(r.content)
        print(f"  Nifty 500 list saved: {NIFTY500_CSV_PATH}")
        return True
    except Exception as e:
        print(f"  Warning: Could not download Nifty 500 list: {e}")
        return False


def load_nifty500_stocks() -> pd.DataFrame:
    """
    Load Nifty 500 constituent list (downloads if missing).
    Returns DataFrame with SYMBOL, COMPANY_NAME, INDUSTRY, YF_TICKER.
    Industry column enables fast sector pre-filtering without yfinance calls.
    """
    if not os.path.exists(NIFTY500_CSV_PATH):
        download_nifty500_list()

    if not os.path.exists(NIFTY500_CSV_PATH):
        return pd.DataFrame()

    try:
        df = pd.read_csv(NIFTY500_CSV_PATH)
        df.columns = [c.strip().upper() for c in df.columns]

        rename = {}
        for col in df.columns:
            if col == "SYMBOL":
                rename[col] = "SYMBOL"
            elif "COMPANY" in col or ("NAME" in col and "SYMBOL" not in col):
                rename[col] = "COMPANY_NAME"
            elif "INDUSTRY" in col:
                rename[col] = "INDUSTRY"
            elif "SERIES" in col:
                rename[col] = "SERIES"
            elif "ISIN" in col:
                rename[col] = "ISIN"
        df = df.rename(columns=rename)

        if "SERIES" in df.columns:
            df = df[df["SERIES"].str.strip() == "EQ"]

        df["YF_TICKER"] = df["SYMBOL"].str.strip() + ".NS"

        cols = [c for c in ["SYMBOL", "COMPANY_NAME", "INDUSTRY", "YF_TICKER"] if c in df.columns]
        return df[cols].dropna(subset=["SYMBOL"])
    except Exception as e:
        print(f"  Warning: Could not parse Nifty 500 CSV: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("Downloading NSE stock list...")
    download_nse_list()
    nse = load_nse_stocks()
    print(f"  NSE stocks loaded: {len(nse)}")
    if not nse.empty:
        print(nse.head(3).to_string())

    print("\nDownloading BSE company list...")
    download_bse_list()
    bse = load_bse_stocks()
    print(f"  BSE stocks loaded: {len(bse)}")
    if not bse.empty:
        print(bse.head(3).to_string())

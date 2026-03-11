import os

# Always resolve paths relative to this file's directory
# so the app works regardless of which directory it's launched from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load .env for local dev (no-op if file missing or dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

def _secret(key: str) -> str:
    """Read from st.secrets (Streamlit Cloud) with fallback to os.getenv (local)."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, ""))
    except Exception:
        return os.getenv(key, "")

ANTHROPIC_API_KEY = _secret("ANTHROPIC_API_KEY")
GEMINI_API_KEY    = _secret("GEMINI_API_KEY")
DATABASE_URL      = _secret("DATABASE_URL")

# RSS Feed URLs for Indian financial news
# Note: ET Markets and NSE Official block bots and return HTML — replaced with working feeds
RSS_FEEDS = [
    {"name": "Moneycontrol Latest",       "url": "http://www.moneycontrol.com/rss/latestnews.xml"},
    {"name": "Livemint Markets",          "url": "https://www.livemint.com/rss/markets"},
    {"name": "Livemint Economy",          "url": "https://www.livemint.com/rss/economy"},
    {"name": "Livemint Companies",        "url": "https://www.livemint.com/rss/companies"},
    {"name": "Livemint Industry",         "url": "https://www.livemint.com/rss/industry"},
    {"name": "Livemint Money",            "url": "https://www.livemint.com/rss/money"},
    {"name": "Business Standard Markets", "url": "https://www.business-standard.com/rss/markets-106.rss"},
    {"name": "Hindu BusinessLine Markets","url": "https://www.thehindubusinessline.com/markets/?service=rss"},
    {"name": "Hindu BusinessLine Economy","url": "https://www.thehindubusinessline.com/economy/?service=rss"},
    {"name": "Hindu BusinessLine Companies","url": "https://www.thehindubusinessline.com/companies/?service=rss"},
]

# NSE Sectors (GICS-style, as used by NSE India)
NSE_SECTORS = [
    "AUTOMOBILE AND AUTO COMPONENTS",
    "CAPITAL GOODS",
    "CHEMICALS",
    "CONSTRUCTION",
    "CONSTRUCTION MATERIALS",
    "CONSUMER DURABLES",
    "CONSUMER SERVICES",
    "DIVERSIFIED",
    "FAST MOVING CONSUMER GOODS",
    "FINANCIAL SERVICES",
    "FOREST MATERIALS",
    "HEALTHCARE",
    "INFORMATION TECHNOLOGY",
    "MEDIA ENTERTAINMENT & PUBLICATION",
    "METALS & MINING",
    "OIL GAS & CONSUMABLE FUELS",
    "POWER",
    "REALTY",
    "SERVICES",
    "TELECOMMUNICATION",
    "TEXTILES",
    "UTILITIES",
]

# Screener quality filter defaults (all configurable in UI)
SCREENER_DEFAULTS = {
    "min_market_cap_cr": 500,     # ₹500 Crore minimum
    "max_pe": 60,                  # P/E < 60
    "min_pe": 0,                   # Exclude negative P/E
    "max_debt_equity": 2.0,        # Debt/Equity < 2
    "max_stocks_per_theme": 15,    # Show top 15 per theme
}

# NSE market holidays (trading closed — weekends handled by Task Scheduler)
NSE_MARKET_HOLIDAYS = {
    # 2025
    "2025-01-26",  # Republic Day
    "2025-02-26",  # Mahashivratri
    "2025-03-14",  # Holi
    "2025-04-10",  # Gudi Padwa
    "2025-04-14",  # Dr. B.R. Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # Maharashtra Day
    "2025-08-15",  # Independence Day
    "2025-08-27",  # Ganesh Chaturthi
    "2025-10-02",  # Gandhi Jayanti / Dussehra
    "2025-10-21",  # Diwali – Laxmi Puja
    "2025-10-22",  # Diwali – Balipratipada
    "2025-11-05",  # Guru Nanak Jayanti
    "2025-12-25",  # Christmas
    # 2026
    "2026-01-26",  # Republic Day
    "2026-03-03",  # Holi
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. B.R. Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-08-15",  # Independence Day
    "2026-10-02",  # Gandhi Jayanti
    "2026-10-19",  # Diwali – Laxmi Puja (approx)
    "2026-11-24",  # Guru Nanak Jayanti (approx)
    "2026-12-25",  # Christmas
}

DB_PATH          = os.path.join(PROJECT_ROOT, "db", "stocks.db")
DATA_DIR         = os.path.join(PROJECT_ROOT, "data")
NSE_CSV_PATH     = os.path.join(PROJECT_ROOT, "data", "nse_stocks.csv")
BSE_CSV_PATH     = os.path.join(PROJECT_ROOT, "data", "bse_stocks.csv")
NIFTY500_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "nifty500.csv")

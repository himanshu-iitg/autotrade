"""Screen NSE stocks by sector + quality fundamentals using yfinance."""
import time
import pandas as pd
import yfinance as yf
from datetime import date
from modules.db import get_conn
from modules.nse_loader import load_nse_stocks, load_nifty500_stocks
from config import SCREENER_DEFAULTS


def _safe_float(val, default=None):
    try:
        f = float(val)
        return f if f == f else default  # NaN check
    except (TypeError, ValueError):
        return default


def get_fundamentals_batch(tickers: list[str], pause: float = 0.3) -> dict:
    """Fetch fundamentals for a list of tickers. Returns dict keyed by ticker."""
    results = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            results[ticker] = {
                "company_name": info.get("longName") or info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": _safe_float(info.get("marketCap")),
                "current_price": _safe_float(info.get("currentPrice") or info.get("regularMarketPrice")),
                "pe": _safe_float(info.get("trailingPE")),
                "pb": _safe_float(info.get("priceToBook")),
                "roe": _safe_float(info.get("returnOnEquity")),
                "debt_equity": _safe_float(info.get("debtToEquity")),
                "revenue_growth": _safe_float(info.get("revenueGrowth")),
                "earnings_growth": _safe_float(info.get("earningsGrowth")),
                "eps_ttm": _safe_float(info.get("trailingEps")),
            }
            time.sleep(pause)
        except Exception:
            results[ticker] = None
    return results


def _nifty500_industry_match(nse_industry: str, sectors: list[str]) -> bool:
    """
    Pre-filter: check if a Nifty 500 industry label matches any theme sector.
    Nifty 500 industry examples: 'Banks', 'Software', 'Pharmaceuticals',
    'Power', 'Metals & Mining', 'Infrastructure', 'Automobile', etc.
    """
    ind = (nse_industry or "").lower()
    # Direct keyword match from theme sectors
    for kw in [s.lower() for s in sectors]:
        if kw in ind:
            return True
    # Reverse: any word from industry appears in any sector name
    ind_words = [w for w in ind.replace("&", " ").replace("-", " ").split() if len(w) > 3]
    sector_text = " ".join(sectors).lower()
    for word in ind_words:
        if word in sector_text:
            return True
    # Curated industry→NSE-sector mappings
    mapping = {
        "banks": ["FINANCIAL SERVICES"],
        "software": ["INFORMATION TECHNOLOGY"],
        "pharmaceuticals": ["HEALTHCARE"],
        "power": ["POWER"],
        "metals": ["METALS & MINING"],
        "mining": ["METALS & MINING"],
        "automobile": ["AUTOMOBILE AND AUTO COMPONENTS"],
        "infrastructure": ["CONSTRUCTION", "CAPITAL GOODS"],
        "cement": ["CONSTRUCTION MATERIALS"],
        "telecom": ["TELECOMMUNICATION"],
        "media": ["MEDIA ENTERTAINMENT & PUBLICATION"],
        "realty": ["REALTY"],
        "textile": ["TEXTILES"],
        "chemicals": ["CHEMICALS"],
        "insurance": ["FINANCIAL SERVICES"],
        "consumer": ["FAST MOVING CONSUMER GOODS", "CONSUMER DURABLES", "CONSUMER SERVICES"],
        "retail": ["CONSUMER SERVICES"],
        "diversified": ["DIVERSIFIED"],
        "capital goods": ["CAPITAL GOODS"],
        "oil": ["OIL GAS & CONSUMABLE FUELS"],
        "gas": ["OIL GAS & CONSUMABLE FUELS"],
        "energy": ["POWER", "OIL GAS & CONSUMABLE FUELS"],
    }
    sectors_upper = [s.upper() for s in sectors]
    for kw, mapped in mapping.items():
        if kw in ind:
            if any(m in sectors_upper for m in mapped):
                return True
    return False


def screen_stocks_for_theme(
    theme_id: int,
    theme_name: str,
    sectors: list[str],
    filters: dict | None = None,
    max_stocks: int = 30,
    force: bool = False,
) -> list[dict]:
    """
    Screen Nifty 500 stocks matching given sectors + quality filters.
    Uses Nifty 500 constituent list with industry pre-filtering for relevant coverage.
    force=True clears today's cache before screening.
    """
    today = date.today().isoformat()
    cfg = {**SCREENER_DEFAULTS, **(filters or {})}

    conn = get_conn()
    c = conn.cursor()

    # Clear cache if forced
    if force:
        c.execute(
            "DELETE FROM screened_stocks WHERE session_date = ? AND theme_id = ?",
            (today, theme_id)
        )
        conn.commit()

    # Check DB cache (avoids repeat yfinance calls for same theme today)
    cached = c.execute(
        "SELECT * FROM screened_stocks WHERE session_date = ? AND theme_id = ?",
        (today, theme_id)
    ).fetchall()
    if cached:
        conn.close()
        return [dict(r) for r in cached]

    # ── Build ticker universe ─────────────────────────────────────────────────
    # Prefer Nifty 500 with industry pre-filtering (quality stocks + relevant sectors)
    nifty500_df = load_nifty500_stocks()

    if not nifty500_df.empty and "INDUSTRY" in nifty500_df.columns:
        # Pre-filter by Nifty 500 industry column — avoids wasting yfinance calls
        mask = nifty500_df["INDUSTRY"].fillna("").apply(
            lambda ind: _nifty500_industry_match(ind, sectors)
        )
        pre_filtered = nifty500_df[mask]

        if not pre_filtered.empty:
            tickers = pre_filtered["YF_TICKER"].dropna().tolist()
            print(f"  Nifty 500 pre-filter: {len(tickers)} stocks match sectors {sectors}")
        else:
            # No industry match — screen all Nifty 500 (shuffled for variety)
            tickers = nifty500_df["YF_TICKER"].dropna().sample(frac=1, random_state=42).tolist()
            print(f"  No Nifty 500 industry match; using full 500 (shuffled)")
    else:
        # Fallback: full NSE list (sorted alphabetically — less ideal)
        nse_df = load_nse_stocks()
        tickers = nse_df["YF_TICKER"].dropna().tolist()
        print(f"  Nifty 500 unavailable; using NSE full list ({len(tickers)} stocks)")

    # Cap batch size: check up to 4× target stocks to have enough after filtering
    batch = tickers[:max_stocks * 4]
    print(f"  Fetching fundamentals for {len(batch)} tickers (may take ~{len(batch) * 0.2:.0f}s)...")
    fundamentals = get_fundamentals_batch(batch, pause=0.2)

    # ── Apply quality filters ─────────────────────────────────────────────────
    results = []
    for ticker, info in fundamentals.items():
        if not info:
            continue

        yf_sector = (info.get("sector") or "").lower()
        yf_industry = (info.get("industry") or "").lower()

        # Sector match via yfinance sector/industry (secondary confirmation)
        sector_keywords = [s.lower() for s in sectors]
        sector_match = (
            any(kw in yf_sector or kw in yf_industry for kw in sector_keywords)
            or _sector_fuzzy_match(yf_sector, yf_industry, sectors)
        )
        if not sector_match:
            continue

        market_cap = info.get("market_cap") or 0
        market_cap_cr = market_cap / 1e7  # Convert to Crores

        pe = info.get("pe")
        debt_eq = info.get("debt_equity")
        eps = info.get("eps_ttm")
        price = info.get("current_price")

        if market_cap_cr < cfg["min_market_cap_cr"]:
            continue
        if pe is not None and (pe <= cfg["min_pe"] or pe > cfg["max_pe"]):
            continue
        if debt_eq is not None and debt_eq > cfg["max_debt_equity"] * 100:
            continue
        if eps is not None and eps <= 0:
            continue
        if not price or price <= 0:
            continue

        results.append({
            "ticker": ticker,
            "company_name": info.get("company_name", ""),
            "sector": info.get("sector", ""),
            "market_cap_cr": round(market_cap_cr, 2),
            "pe": round(pe, 2) if pe else None,
            "pb": round(info.get("pb") or 0, 2),
            "roe": round((info.get("roe") or 0) * 100, 2),
            "debt_equity": round((debt_eq or 0) / 100, 2),
            "revenue_growth": round((info.get("revenue_growth") or 0) * 100, 2),
            "eps_growth": round((info.get("earnings_growth") or 0) * 100, 2),
            "current_price": round(price, 2),
        })

    # Sort by market cap descending, take top N
    results.sort(key=lambda x: x["market_cap_cr"], reverse=True)
    results = results[:cfg["max_stocks_per_theme"]]

    # Save to DB
    for r in results:
        c.execute("""
            INSERT OR IGNORE INTO screened_stocks
            (session_date, theme_id, ticker, company_name, sector, market_cap_cr,
             pe, pb, roe, debt_equity, revenue_growth, eps_growth, current_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (today, theme_id, r["ticker"], r["company_name"], r["sector"],
              r["market_cap_cr"], r["pe"], r["pb"], r["roe"], r["debt_equity"],
              r["revenue_growth"], r["eps_growth"], r["current_price"]))

    conn.commit()
    conn.close()
    return results


def _sector_fuzzy_match(yf_sector: str, yf_industry: str, nse_sectors: list[str]) -> bool:
    """Map yfinance sector names to NSE sector names loosely."""
    mapping = {
        "technology": ["INFORMATION TECHNOLOGY"],
        "financial": ["FINANCIAL SERVICES"],
        "healthcare": ["HEALTHCARE"],
        "energy": ["OIL GAS & CONSUMABLE FUELS", "POWER"],
        "industrials": ["CAPITAL GOODS", "CONSTRUCTION"],
        "consumer": ["FAST MOVING CONSUMER GOODS", "CONSUMER DURABLES", "CONSUMER SERVICES"],
        "materials": ["METALS & MINING", "CHEMICALS", "CONSTRUCTION MATERIALS"],
        "real estate": ["REALTY"],
        "utilities": ["POWER", "UTILITIES"],
        "communication": ["TELECOMMUNICATION", "MEDIA ENTERTAINMENT & PUBLICATION"],
        "automobile": ["AUTOMOBILE AND AUTO COMPONENTS"],
        "pharma": ["HEALTHCARE"],
        "banking": ["FINANCIAL SERVICES"],
        "power": ["POWER"],
        "defence": ["CAPITAL GOODS"],
        "infrastructure": ["CONSTRUCTION", "CAPITAL GOODS"],
        "renewable": ["POWER"],
        "cement": ["CONSTRUCTION MATERIALS"],
        "steel": ["METALS & MINING"],
        "chemical": ["CHEMICALS"],
        "textile": ["TEXTILES"],
    }
    nse_upper = [s.upper() for s in nse_sectors]
    combined = (yf_sector + " " + yf_industry).lower()
    for keyword, mapped_sectors in mapping.items():
        if keyword in combined:
            if any(ms in nse_upper for ms in mapped_sectors):
                return True
    return False

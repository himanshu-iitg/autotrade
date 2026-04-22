"""Screen NSE stocks by sector + quality fundamentals using yfinance.

Key design decisions:
  - Nifty 500 industry pre-filter is the ONLY sector gate. The secondary
    yfinance sector check was a false-negative trap (US taxonomy vs NSE labels)
    and has been removed as a hard filter.
  - EPS filter is now soft: a stock passes if EPS > 0, OR if ROE > 10% and
    revenue growth > 5% (quality growth companies often have temporarily
    depressed EPS from exceptional items / capex cycles).
  - max_results controls the final output count directly (no more split between
    max_stocks param and SCREENER_DEFAULTS["max_stocks_per_theme"]).
  - New screen_nifty500() entry point for screen-first (no theme required).
"""
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


def _passes_quality_filters(info: dict, cfg: dict) -> tuple[bool, str]:
    """
    Apply fundamental quality filters. Returns (passes: bool, reason: str).

    EPS rule (soft): pass if EPS > 0, OR if the company shows quality
    signals (ROE > 10% and revenue growth > 5%) — catches great companies
    with temporarily depressed earnings (infra capex cycles, provisioning hits,
    exceptional write-offs, etc.).
    """
    market_cap    = info.get("market_cap") or 0
    market_cap_cr = market_cap / 1e7
    pe            = info.get("pe")
    debt_eq       = info.get("debt_equity")   # yfinance returns this as %, e.g. 150 = 1.5x
    eps           = info.get("eps_ttm")
    roe           = info.get("roe") or 0       # decimal, e.g. 0.15 = 15%
    rev_growth    = info.get("revenue_growth") or 0  # decimal, e.g. 0.12 = 12%
    price         = info.get("current_price")

    if market_cap_cr < cfg["min_market_cap_cr"]:
        return False, f"market_cap ₹{market_cap_cr:.0f}Cr < min {cfg['min_market_cap_cr']}"

    if not price or price <= 0:
        return False, "no price data"

    if pe is not None and pe <= cfg.get("min_pe", 0):
        return False, f"P/E {pe:.1f} <= min {cfg.get('min_pe', 0)}"

    if pe is not None and pe > cfg["max_pe"]:
        return False, f"P/E {pe:.1f} > max {cfg['max_pe']}"

    if debt_eq is not None and debt_eq > cfg["max_debt_equity"] * 100:
        return False, f"D/E {debt_eq/100:.1f}x > max {cfg['max_debt_equity']}x"

    # ── Soft EPS filter ────────────────────────────────────────────────────────
    # Block only if: EPS negative AND neither ROE nor revenue growth compensates
    if eps is not None and eps <= 0:
        quality_compensates = (roe * 100 > 10) or (rev_growth * 100 > 5)
        if not quality_compensates:
            return False, f"EPS {eps:.2f} <= 0 with no quality offset (ROE={roe*100:.1f}%, RevGrowth={rev_growth*100:.1f}%)"

    # ── Optional growth floor ──────────────────────────────────────────────────
    min_rev_growth = cfg.get("min_revenue_growth_pct")   # e.g. -5 (allow slight declines)
    min_roe        = cfg.get("min_roe_pct")               # e.g. 8
    if min_rev_growth is not None and rev_growth * 100 < min_rev_growth:
        return False, f"revenue growth {rev_growth*100:.1f}% < min {min_rev_growth}%"
    if min_roe is not None and roe * 100 < min_roe:
        return False, f"ROE {roe*100:.1f}% < min {min_roe}%"

    return True, "ok"


def _build_result(ticker: str, info: dict) -> dict:
    """Build a clean result dict from raw yfinance info."""
    pe      = info.get("pe")
    debt_eq = info.get("debt_equity") or 0
    price   = info.get("current_price")
    return {
        "ticker":         ticker,
        "company_name":   info.get("company_name", ""),
        "sector":         info.get("sector", ""),
        "industry":       info.get("industry", ""),
        "market_cap_cr":  round((info.get("market_cap") or 0) / 1e7, 2),
        "pe":             round(pe, 2) if pe else None,
        "pb":             round(info.get("pb") or 0, 2),
        "roe":            round((info.get("roe") or 0) * 100, 2),
        "debt_equity":    round(debt_eq / 100, 2),
        "revenue_growth": round((info.get("revenue_growth") or 0) * 100, 2),
        "eps_growth":     round((info.get("earnings_growth") or 0) * 100, 2),
        "eps_ttm":        info.get("eps_ttm"),
        "current_price":  round(price, 2) if price else None,
    }


def get_recently_seen_tickers(days_back: int = 3) -> set[str]:
    """Return tickers that appeared in any screen in the last N days (excluding today)."""
    try:
        from datetime import date, timedelta
        conn  = get_conn()
        c     = conn.cursor()
        since = (date.today() - timedelta(days=days_back)).isoformat()
        today = date.today().isoformat()
        rows  = c.execute(
            "SELECT DISTINCT ticker FROM screened_stocks WHERE session_date >= %s AND session_date < %s",
            (since, today),
        ).fetchall()
        conn.close()
        return {r["ticker"] for r in rows}
    except Exception:
        return set()


def screen_nifty500(
    sectors: list[str] | None = None,
    filters: dict | None = None,
    max_results: int = 30,
    cache_key: str = "all",
    force: bool = False,
    shuffle: bool = False,
    exclude_tickers: set[str] | None = None,
) -> list[dict]:
    """
    Screen-first entry point — no theme required.

    Screens the full Nifty 500 (or a sector-filtered subset) purely on
    fundamental quality. Call this first, then run news_triage on the results.

    sectors : NSE sector list to pre-filter by, or None for all 500
    max_results : final number of stocks to return (sorted by market cap)
    cache_key : used to namespace the DB cache (e.g. "all", "banking", "auto")
    """
    today = date.today().isoformat()
    cfg   = {**SCREENER_DEFAULTS, **(filters or {})}

    conn = get_conn()
    c    = conn.cursor()

    # Theme-agnostic cache: use theme_id = -1 and store cache_key in company_name prefix
    cache_theme_id = -1
    if force:
        c.execute(
            "DELETE FROM screened_stocks WHERE session_date = %s AND theme_id = %s",
            (today, cache_theme_id),
        )
        conn.commit()

    cached = c.execute(
        "SELECT * FROM screened_stocks WHERE session_date = %s AND theme_id = %s ORDER BY market_cap_cr DESC",
        (today, cache_theme_id),
    ).fetchall()
    if cached:
        conn.close()
        return [dict(r) for r in cached]

    conn.close()

    # Build universe
    nifty500_df = load_nifty500_stocks()
    if nifty500_df.empty:
        nse_df  = load_nse_stocks()
        tickers = nse_df["YF_TICKER"].dropna().tolist()
        print(f"  Nifty 500 unavailable; using full NSE list ({len(tickers)} stocks)")
    elif sectors and "INDUSTRY" in nifty500_df.columns:
        mask    = nifty500_df["INDUSTRY"].fillna("").apply(
            lambda ind: _nifty500_industry_match(ind, sectors)
        )
        filtered_df = nifty500_df[mask]
        tickers = filtered_df["YF_TICKER"].dropna().tolist() if not filtered_df.empty \
                  else nifty500_df["YF_TICKER"].dropna().tolist()
        print(f"  Nifty 500 sector pre-filter: {len(tickers)} stocks")
    else:
        tickers = nifty500_df["YF_TICKER"].dropna().tolist()
        print(f"  Screening full Nifty 500 ({len(tickers)} stocks)")

    # Discovery mode: shuffle so different stocks surface each run
    if shuffle:
        import random
        random.shuffle(tickers)

    # Exclude recently seen tickers if requested
    if exclude_tickers:
        before = len(tickers)
        tickers = [t for t in tickers if t not in exclude_tickers]
        print(f"  Excluded {before - len(tickers)} recently seen tickers")

    # Fetch in batches of max_results * 4 to have enough candidates after filtering
    batch = tickers[: max_results * 4]
    print(f"  Fetching fundamentals for {len(batch)} tickers (~{len(batch) * 0.2:.0f}s)...")
    fundamentals = get_fundamentals_batch(batch, pause=0.2)

    results = []
    for ticker, info in fundamentals.items():
        if not info:
            continue
        passes, reason = _passes_quality_filters(info, cfg)
        if not passes:
            continue
        results.append(_build_result(ticker, info))

    # Sort by market cap descending, take top N
    results.sort(key=lambda r: r.get("market_cap_cr") or 0, reverse=True)
    results = results[:max_results]

    # Persist to DB (theme_id = -1 = screen_nifty500 cache)
    if results:
        conn2 = get_conn()
        c2    = conn2.cursor()
        cols  = list(results[0].keys())
        for r in results:
            placeholders = ", ".join(["%s"] * (len(cols) + 2))
            col_str      = ", ".join(["session_date", "theme_id"] + cols)
            vals         = [today, cache_theme_id] + [r.get(c) for c in cols]
            c2.execute(
                f"INSERT INTO screened_stocks ({col_str}) VALUES ({placeholders}) "
                f"ON CONFLICT DO NOTHING",
                vals,
            )
        conn2.commit()
        conn2.close()

    print(f"  screen_nifty500 done: {len(results)} stocks passed filters")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Legacy theme-based screener (kept for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def screen_stocks_for_theme(
    theme: dict,
    sectors: list[str] | None = None,
    filters: dict | None = None,
    max_stocks: int | None = None,
    force: bool = False,
) -> list[dict]:
    """
    Screen Nifty 500 stocks for a given investment theme.

    theme   : dict with keys 'id', 'name', 'description', 'sectors'
    sectors : override theme sectors (optional)
    filters : override SCREENER_DEFAULTS keys (optional)
    max_stocks : final count cap (defaults to SCREENER_DEFAULTS["max_stocks_per_theme"])
    """
    today    = date.today().isoformat()
    theme_id = theme.get("id", 0)
    cfg      = {**SCREENER_DEFAULTS, **(filters or {})}
    max_n    = max_stocks or cfg.get("max_stocks_per_theme", 15)

    conn = get_conn()
    c    = conn.cursor()

    if force:
        c.execute(
            "DELETE FROM screened_stocks WHERE session_date = %s AND theme_id = %s",
            (today, theme_id),
        )
        conn.commit()

    cached = c.execute(
        "SELECT * FROM screened_stocks WHERE session_date = %s AND theme_id = %s ORDER BY market_cap_cr DESC",
        (today, theme_id),
    ).fetchall()
    if cached:
        conn.close()
        return [dict(r) for r in cached][:max_n]

    conn.close()

    effective_sectors = sectors or theme.get("sectors", [])

    # Build universe from Nifty 500 with optional sector pre-filter
    nifty500_df = load_nifty500_stocks()
    if nifty500_df.empty:
        nse_df  = load_nse_stocks()
        tickers = nse_df["YF_TICKER"].dropna().tolist()
    elif effective_sectors and "INDUSTRY" in nifty500_df.columns:
        mask        = nifty500_df["INDUSTRY"].fillna("").apply(
            lambda ind: _nifty500_industry_match(ind, effective_sectors)
        )
        filtered_df = nifty500_df[mask]
        tickers     = filtered_df["YF_TICKER"].dropna().tolist() if not filtered_df.empty                       else nifty500_df["YF_TICKER"].dropna().tolist()
    else:
        tickers = nifty500_df["YF_TICKER"].dropna().tolist()

    batch        = tickers[: max_n * 4]
    fundamentals = get_fundamentals_batch(batch, pause=0.2)

    results = []
    for ticker, info in fundamentals.items():
        if not info:
            continue
        passes, reason = _passes_quality_filters(info, cfg)
        if not passes:
            continue
        results.append(_build_result(ticker, info))

    results.sort(key=lambda r: r.get("market_cap_cr") or 0, reverse=True)
    results = results[:max_n]

    if results:
        conn2 = get_conn()
        c2    = conn2.cursor()
        cols  = list(results[0].keys())
        for r in results:
            placeholders = ", ".join(["%s"] * (len(cols) + 2))
            col_str      = ", ".join(["session_date", "theme_id"] + cols)
            vals         = [today, theme_id] + [r.get(c) for c in cols]
            c2.execute(
                f"INSERT INTO screened_stocks ({col_str}) VALUES ({placeholders}) "
                f"ON CONFLICT DO NOTHING",
                vals,
            )
        conn2.commit()
        conn2.close()

    return results

"""
News Triage — sector-aware, ticker-level news intelligence.

Instead of extracting broad themes from all headlines and hoping they map
to stocks, this module works in reverse:

  1. You already have a shortlist of fundamentally screened stocks.
  2. For each stock, we know its sector (e.g. FINANCIAL SERVICES).
  3. We look up what signals actually move that sector (e.g. repo rate for banks,
     FDA approvals for pharma, USD/INR for IT, coal prices for power).
  4. We search both yfinance news (stock-specific) and the existing news DB
     (macro signal keywords) for relevant items.
  5. We ask the LLM a narrow, focused question:
     "Here is HDFC Bank. Here is what matters for banks. Here is what the
      news is saying. Is this a POSITIVE, NEUTRAL, or NEGATIVE setup?"

The LLM's job shrinks from "discover themes from 200 headlines" to
"triage 5–8 focused news items for this specific stock." Much more reliable.
"""
import json
import time
import yfinance as yf
from modules.db import get_conn
from modules.llm_client import call_llm, _strip_fences

# ── Sector signal maps ────────────────────────────────────────────────────────
# For each NSE sector, we define:
#   "signals"  : human-readable signal names passed to the LLM as context
#   "keywords" : search terms for our news DB (case-insensitive substring match)
#
# The idea: the LLM is told "these are the things that move this sector"
# so its analysis is grounded, not hallucinated.

SECTOR_SIGNALS: dict[str, dict] = {

    "FINANCIAL SERVICES": {
        "signals": [
            "RBI repo rate decisions and monetary policy stance",
            "System credit growth (bank credit YoY %)",
            "Gross NPA / GNPA ratios and asset quality trends",
            "CASA ratio and deposit growth",
            "RBI regulatory actions, fines, or notices",
            "Liquidity conditions (CRR, SLR, OMO operations)",
            "Government borrowing programme and bond yields",
        ],
        "keywords": [
            "repo rate", "RBI", "monetary policy", "rate cut", "rate hike",
            "credit growth", "NPA", "GNPA", "NNPA", "CASA", "liquidity",
            "bank credit", "CRR", "SLR", "bond yield", "PSL", "NBFC",
            "financial stability", "banking regulation",
        ],
    },

    "INFORMATION TECHNOLOGY": {
        "signals": [
            "USD/INR exchange rate movement",
            "US and European IT spending outlook",
            "Large deal wins and total contract value (TCV)",
            "US recession risk or macroeconomic slowdown",
            "H-1B visa policy changes",
            "Employee attrition and hiring trends",
            "AI / GenAI adoption affecting deal pipeline",
            "NASSCOM outlook and industry forecasts",
        ],
        "keywords": [
            "USD", "rupee", "dollar", "deal win", "TCV", "US recession",
            "IT spending", "tech spending", "attrition", "visa", "H1B",
            "NASSCOM", "offshoring", "outsourcing", "GenAI", "AI deal",
            "Infosys guidance", "TCS", "Wipro results",
        ],
    },

    "HEALTHCARE": {
        "signals": [
            "US FDA approvals, warning letters, or 483 observations",
            "USFDA import alerts on Indian plants",
            "US generic drug pricing pressure",
            "NPPA drug price control orders",
            "API raw material prices (China supply)",
            "Domestic healthcare policy and insurance expansion",
            "New drug launches and patent cliffs",
        ],
        "keywords": [
            "FDA", "USFDA", "drug approval", "483 observation", "warning letter",
            "import alert", "ANDA", "generic", "NPPA", "drug price",
            "API", "active pharmaceutical", "pharma export", "biosimilar",
            "healthcare policy", "Ayushman", "insurance",
        ],
    },

    "AUTOMOBILE AND AUTO COMPONENTS": {
        "signals": [
            "Monthly vehicle retail sales (SIAM / FADA data)",
            "Steel and aluminium input cost movements",
            "Semiconductor chip supply situation",
            "EV penetration and PLI scheme progress",
            "Fuel prices (petrol, diesel, CNG)",
            "Two-wheeler rural demand (linked to monsoon and agri income)",
            "Export markets and currency competitiveness",
        ],
        "keywords": [
            "auto sales", "vehicle sales", "SIAM", "FADA", "EV", "electric vehicle",
            "PLI auto", "steel price", "aluminium price", "semiconductor",
            "chip shortage", "fuel price", "petrol diesel", "two wheeler",
            "passenger vehicle", "commercial vehicle", "Maruti", "Tata Motors",
        ],
    },

    "CAPITAL GOODS": {
        "signals": [
            "Government capex budget allocation and execution pace",
            "Infrastructure project awards and ordering pipeline",
            "Defence indigenisation orders (Make in India)",
            "PLI scheme disbursements for manufacturing",
            "Private capex revival signals",
            "Order book growth of L&T, BHEL, HAL",
        ],
        "keywords": [
            "capex", "government spending", "infrastructure order", "order win",
            "defence order", "PLI", "Make in India", "L&T order", "BHEL",
            "power plant", "railways capex", "road project", "NHAI",
            "manufacturing", "industrial output", "IIP",
        ],
    },

    "CONSTRUCTION": {
        "signals": [
            "Government infrastructure spend: NHAI, railways, smart cities",
            "Real estate new launches and housing demand",
            "Cement price trends",
            "Steel rebar prices",
            "Labour availability and project execution pace",
            "State government capex and election-year spending",
        ],
        "keywords": [
            "NHAI", "road construction", "highway", "smart city", "housing",
            "real estate launch", "cement price", "rebar", "steel price",
            "construction order", "infra spend", "DFC", "metro project",
        ],
    },

    "CONSTRUCTION MATERIALS": {
        "signals": [
            "Cement price and volume data (CMA monthly report)",
            "Petcoke and coal input cost trends",
            "Housing and infrastructure demand from government",
            "Limestone availability and mining policy",
            "Consolidation and pricing power in cement industry",
        ],
        "keywords": [
            "cement price", "cement volume", "cement demand", "petcoke", "coal price",
            "limestone", "housing demand", "CMA", "Ultratech", "Shree Cement",
            "ACC", "Ambuja", "cement despatch",
        ],
    },

    "FAST MOVING CONSUMER GOODS": {
        "signals": [
            "Rural consumption and agri income (linked to kharif/rabi crops)",
            "CPI inflation eating into household discretionary spend",
            "Commodity input costs: palm oil, wheat, sugar, crude derivatives",
            "Urban wage growth and employment data",
            "GST rate changes on FMCG categories",
            "Monsoon forecast and actual rainfall (kharif sowing)",
        ],
        "keywords": [
            "rural consumption", "FMCG volume", "palm oil", "wheat price",
            "sugar price", "rural income", "monsoon", "kharif", "CPI",
            "inflation", "GST", "Hindustan Unilever", "Nestle", "ITC",
            "consumer staples", "FMCG growth",
        ],
    },

    "CONSUMER DURABLES": {
        "signals": [
            "Urban discretionary spending trends",
            "Real estate cycle (housing drives white goods demand)",
            "Summer season and AC demand",
            "Input costs: copper, aluminium, steel",
            "Credit availability for consumer financing",
            "Import duty changes on components",
        ],
        "keywords": [
            "consumer durables", "AC sales", "white goods", "appliance",
            "copper price", "aluminium price", "housing", "consumer credit",
            "EMI", "Dixon", "Havells", "Voltas", "Blue Star",
        ],
    },

    "METALS & MINING": {
        "signals": [
            "China steel and metals demand / production data",
            "LME copper, aluminium, zinc prices",
            "Iron ore and coking coal prices",
            "Global PMI and industrial production",
            "Anti-dumping duties and trade policy",
            "Domestic infrastructure-led demand",
        ],
        "keywords": [
            "LME", "steel price", "iron ore", "coking coal", "copper",
            "aluminium", "zinc", "China PMI", "China steel", "PMI",
            "anti-dumping", "Tata Steel", "JSW", "Hindalco", "Vedanta",
            "NMDC", "SAIL",
        ],
    },

    "OIL GAS & CONSUMABLE FUELS": {
        "signals": [
            "Brent crude and WTI oil prices",
            "Government retail fuel price regulation (auto fuel subsidy)",
            "GRM (gross refining margins) for refiners",
            "Natural gas prices and city gas distribution expansion",
            "OPEC+ production decisions",
            "Petrochemical margins",
        ],
        "keywords": [
            "crude oil", "Brent", "oil price", "petrol price", "diesel price",
            "GRM", "refining margin", "natural gas", "city gas", "CGD",
            "OPEC", "petrochemical", "ONGC", "Reliance", "IOC", "BPCL",
            "petroleum", "LNG",
        ],
    },

    "POWER": {
        "signals": [
            "Coal prices and availability (domestic + imported)",
            "Renewable energy capacity addition and solar/wind tariffs",
            "Power demand growth and PLF (plant load factor)",
            "Government's 500 GW renewable target progress",
            "Electricity distribution company (DISCOM) health and payment dues",
            "Hydro output affected by rainfall",
        ],
        "keywords": [
            "coal price", "coal supply", "renewable energy", "solar tariff",
            "wind energy", "power demand", "PLF", "DISCOM", "electricity",
            "green hydrogen", "Adani Power", "NTPC", "Tata Power",
            "CESC", "power plant", "hydro", "500 GW",
        ],
    },

    "REALTY": {
        "signals": [
            "RBI repo rate impact on home loan EMIs",
            "Housing sales volumes (Knight Frank, PropEquity, Anarock reports)",
            "New project launches in tier-1 cities",
            "Affordable housing policy and PMAY",
            "Commercial real estate demand (office absorption)",
            "Private equity flows into real estate",
        ],
        "keywords": [
            "repo rate", "home loan", "EMI", "housing sales", "property price",
            "real estate", "office space", "commercial real estate",
            "DLF", "Godrej Properties", "Prestige", "PMAY",
            "affordable housing", "PropEquity", "Anarock",
        ],
    },

    "TELECOMMUNICATION": {
        "signals": [
            "ARPU (average revenue per user) trends",
            "5G rollout progress and spectrum utilisation",
            "Tariff hike cycle",
            "TRAI regulation and interconnect charges",
            "Subscriber additions and churn",
            "Spectrum auction costs and debt servicing",
        ],
        "keywords": [
            "ARPU", "5G", "telecom", "spectrum", "tariff hike", "TRAI",
            "subscriber", "Jio", "Airtel", "Vi", "Vodafone Idea",
            "broadband", "AGR", "spectrum auction",
        ],
    },

    "CHEMICALS": {
        "signals": [
            "China+1 supply chain diversification benefiting Indian chemical exporters",
            "Crude oil price (feedstock for petrochemicals)",
            "Specialty chemical demand from agrochemical and pharma end-users",
            "Anti-dumping cases on Chinese chemical imports",
            "Domestic agrochemical season and crop protection demand",
            "European chemical industry destocking cycle",
        ],
        "keywords": [
            "specialty chemical", "China+1", "agrochemical", "pesticide",
            "anti-dumping chemical", "chemical export", "feedstock",
            "PI Industries", "SRF", "Aarti Industries", "Deepak Nitrite",
            "Navin Fluorine", "European chemical", "destocking",
        ],
    },

    "TEXTILES": {
        "signals": [
            "Cotton prices (Shankar-6 spot rates)",
            "Bangladesh and Vietnam competition in garment exports",
            "US and EU apparel import demand",
            "PLI scheme for man-made fibres (MMF) and technical textiles",
            "Rupee-dollar rate for exporters",
            "Chinese yarn dumping",
        ],
        "keywords": [
            "cotton price", "yarn price", "textile export", "garment",
            "Bangladesh", "PLI textile", "man-made fibre", "MMF",
            "Page Industries", "Trident", "Welspun", "apparel",
        ],
    },

    "DIVERSIFIED": {
        "signals": [
            "Broad macroeconomic growth (GDP, IIP)",
            "Group-level debt or restructuring news",
            "Conglomerate restructuring / demerger announcements",
            "FII/DII flows and market sentiment",
        ],
        "keywords": [
            "GDP", "IIP", "conglomerate", "demerger", "restructuring",
            "FII", "DII", "Adani", "Tata", "Mahindra", "Aditya Birla",
        ],
    },
}

# Fallback for unknown sectors
_DEFAULT_SIGNALS = {
    "signals": [
        "India GDP and industrial production (IIP)",
        "FII / DII equity flows",
        "RBI monetary policy stance",
        "Global risk-on / risk-off sentiment",
    ],
    "keywords": ["GDP", "IIP", "FII", "DII", "RBI", "interest rate", "inflation"],
}

# ── LLM prompt ────────────────────────────────────────────────────────────────

_TRIAGE_SYSTEM = (
    "You are a sharp equity analyst at a top Indian fund house. "
    "You analyse specific stocks using sector-relevant news signals. "
    "Be concise, data-grounded, and avoid generic commentary. "
    "Respond ONLY with valid JSON."
)

_TRIAGE_PROMPT = """You are triaging news for a specific stock to decide if recent developments
are a POSITIVE, NEUTRAL, or NEGATIVE setup for a 6–18 month investment.

STOCK: {company_name} ({ticker})
SECTOR: {sector}

KEY SIGNALS FOR THIS SECTOR (what actually drives this stock's earnings):
{signals_text}

RECENT NEWS RELEVANT TO THIS STOCK AND ITS SECTOR:
{news_text}

STOCK FUNDAMENTALS (for context):
  Market Cap: ₹{market_cap_cr:.0f} Cr  |  P/E: {pe}  |  ROE: {roe}%  |  Rev Growth: {rev_growth}%

Based ONLY on the news above (not general knowledge), provide:
1. sentiment: "POSITIVE", "NEUTRAL", or "NEGATIVE"
2. confidence: 0.0–1.0 (how much relevant news was available; 0.3 if very little)
3. key_catalyst: The single most important positive signal (or "none")
4. key_risk: The single most important negative signal (or "none")
5. signal_hits: Which sector signals from the list above were mentioned in the news
6. summary: One sentence — what does the news say about this stock's near-term outlook?

Respond ONLY with valid JSON:
{{
  "sentiment": "POSITIVE",
  "confidence": 0.7,
  "key_catalyst": "RBI rate cut cycle starting — reduces HDFC Bank's cost of funds",
  "key_risk": "Rising GNPA in SME segment noted in Q3 commentary",
  "signal_hits": ["RBI repo rate decisions", "Gross NPA / GNPA ratios"],
  "summary": "Rate cut tailwind and strong credit growth outweigh marginal NPA uptick."
}}"""


# ── Core functions ────────────────────────────────────────────────────────────

def _get_sector_config(sector: str) -> dict:
    """Match a yfinance sector string to our signal map, with fuzzy fallback."""
    sector_upper = (sector or "").upper()

    # Direct match
    if sector_upper in SECTOR_SIGNALS:
        return SECTOR_SIGNALS[sector_upper]

    # Fuzzy keyword match
    sector_lower = sector_upper.lower()
    keyword_map = {
        "financial":    "FINANCIAL SERVICES",
        "bank":         "FINANCIAL SERVICES",
        "technology":   "INFORMATION TECHNOLOGY",
        "software":     "INFORMATION TECHNOLOGY",
        "health":       "HEALTHCARE",
        "pharma":       "HEALTHCARE",
        "auto":         "AUTOMOBILE AND AUTO COMPONENTS",
        "capital good": "CAPITAL GOODS",
        "industrial":   "CAPITAL GOODS",
        "construct":    "CONSTRUCTION",
        "cement":       "CONSTRUCTION MATERIALS",
        "consumer stap":"FAST MOVING CONSUMER GOODS",
        "fmcg":         "FAST MOVING CONSUMER GOODS",
        "consumer disc":"CONSUMER DURABLES",
        "metal":        "METALS & MINING",
        "material":     "METALS & MINING",
        "energy":       "OIL GAS & CONSUMABLE FUELS",
        "oil":          "OIL GAS & CONSUMABLE FUELS",
        "power":        "POWER",
        "util":         "POWER",
        "real estate":  "REALTY",
        "telecom":      "TELECOMMUNICATION",
        "communic":     "TELECOMMUNICATION",
        "chemical":     "CHEMICALS",
        "textile":      "TEXTILES",
    }
    for kw, nse_sector in keyword_map.items():
        if kw in sector_lower:
            return SECTOR_SIGNALS.get(nse_sector, _DEFAULT_SIGNALS)

    return _DEFAULT_SIGNALS


def fetch_yfinance_news(ticker: str, max_items: int = 8) -> list[dict]:
    """
    Pull recent news items directly from Yahoo Finance for a specific ticker.
    Returns list of {title, publisher, summary} dicts.
    """
    items = []
    try:
        news = yf.Ticker(ticker).news or []
        for item in news[:max_items]:
            title = item.get("title", "")
            publisher = item.get("publisher", "")
            # Summary is in content->summary for newer yfinance versions
            summary = ""
            content = item.get("content") or {}
            if isinstance(content, dict):
                summary = content.get("summary", "")
            if not summary:
                summary = item.get("summary", "")
            if title:
                items.append({
                    "title":     title,
                    "publisher": publisher,
                    "summary":   summary[:200] if summary else "",
                    "source":    "yfinance",
                })
    except Exception as e:
        print(f"  [Triage] yfinance news fetch failed for {ticker}: {e}")
    return items


def search_db_for_signals(keywords: list[str], hours_back: int = 48, limit: int = 15) -> list[dict]:
    """
    Search the existing news DB for articles matching sector signal keywords.
    Returns a deduplicated list of relevant articles sorted by recency.
    """
    if not keywords:
        return []

    try:
        conn = get_conn()
        c    = conn.cursor()

        # Build OR condition for all keywords
        conditions = " OR ".join(
            ["LOWER(title) LIKE %s OR LOWER(summary) LIKE %s"] * len(keywords)
        )
        params = []
        for kw in keywords:
            kw_lower = f"%{kw.lower()}%"
            params.extend([kw_lower, kw_lower])

        rows = c.execute(f"""
            SELECT DISTINCT source, title, summary
            FROM news
            WHERE ({conditions})
            ORDER BY fetched_at DESC
            LIMIT %s
        """, params + [limit]).fetchall()
        conn.close()

        return [
            {
                "title":     r["title"],
                "publisher": r["source"],
                "summary":   (r["summary"] or "")[:200],
                "source":    "db",
            }
            for r in rows
        ]
    except Exception as e:
        print(f"  [Triage] DB signal search error: {e}")
        return []


def triage_stock(
    stock: dict,
    provider: str = "gemini",
    pause: float = 0.5,
) -> dict:
    """
    Run full news triage for a single screened stock.

    stock : dict from screener (must have ticker, company_name, sector, etc.)
    Returns the stock dict enriched with triage fields.
    """
    ticker       = stock.get("ticker", "")
    company_name = stock.get("company_name", ticker)
    sector       = stock.get("sector", "")

    sector_cfg = _get_sector_config(sector)
    signals    = sector_cfg["signals"]
    keywords   = sector_cfg["keywords"]

    # Add company name and ticker symbol as extra search keywords
    nse_symbol = ticker.replace(".NS", "").replace(".BO", "")
    extra_kws  = [company_name.split()[0], nse_symbol] if company_name else [nse_symbol]
    all_keywords = list(dict.fromkeys(extra_kws + keywords))  # deduplicated, company first

    # ── Fetch news from both sources ──────────────────────────────────────────
    time.sleep(pause)
    yf_news = fetch_yfinance_news(ticker)
    db_news = search_db_for_signals(all_keywords)

    # Merge and deduplicate by title
    seen_titles: set[str] = set()
    all_news: list[dict] = []
    for item in yf_news + db_news:
        title_key = item["title"].lower()[:60]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            all_news.append(item)

    all_news = all_news[:15]  # cap context window impact

    # ── Build LLM prompt ──────────────────────────────────────────────────────
    signals_text = "\n".join(f"  • {s}" for s in signals)

    if all_news:
        news_lines = []
        for n in all_news:
            src     = n.get("publisher") or n.get("source", "")
            summary = n.get("summary", "")
            line    = f"  [{src}] {n['title']}"
            if summary:
                line += f" — {summary}"
            news_lines.append(line)
        news_text = "\n".join(news_lines)
    else:
        news_text = "  (no relevant news found in last 48 hours)"

    prompt = _TRIAGE_PROMPT.format(
        company_name = company_name,
        ticker       = ticker,
        sector       = sector,
        signals_text = signals_text,
        news_text    = news_text,
        market_cap_cr= stock.get("market_cap_cr", 0),
        pe           = stock.get("pe") or "N/A",
        roe          = stock.get("roe") or "N/A",
        rev_growth   = stock.get("revenue_growth") or "N/A",
    )

    # ── LLM call ──────────────────────────────────────────────────────────────
    defaults = {
        "sentiment":    "NEUTRAL",
        "confidence":   0.3,
        "key_catalyst": "none",
        "key_risk":     "none",
        "signal_hits":  [],
        "summary":      "Insufficient news to form a view.",
    }

    try:
        raw = call_llm(
            prompt  = prompt,
            system  = _TRIAGE_SYSTEM,
            provider= provider,
            mode    = "fast",
            max_tokens = 400,
        )
        result = json.loads(_strip_fences(raw))
        triage = {**defaults, **result}
    except Exception as e:
        print(f"  [Triage] LLM call failed for {ticker}: {e}")
        triage = defaults

    # ── Enrich stock dict ─────────────────────────────────────────────────────
    stock.update({
        "triage_sentiment":    triage.get("sentiment", "NEUTRAL"),
        "triage_confidence":   triage.get("confidence", 0.3),
        "triage_catalyst":     triage.get("key_catalyst", "none"),
        "triage_risk":         triage.get("key_risk", "none"),
        "triage_signal_hits":  triage.get("signal_hits", []),
        "triage_summary":      triage.get("summary", ""),
        "triage_news_count":   len(all_news),
    })
    return stock


def triage_batch(
    stocks: list[dict],
    provider: str = "gemini",
    top_n: int = 15,
    pause: float = 0.5,
) -> list[dict]:
    """
    Run triage on the top_n stocks from a screened list.
    Returns the same list with triage fields added, sorted by:
      1. Sentiment (POSITIVE > NEUTRAL > NEGATIVE)
      2. Triage confidence
      3. Market cap (as tiebreaker)
    """
    # Only triage the top_n by market cap — avoids excessive LLM calls
    to_triage = stocks[:top_n]
    rest      = stocks[top_n:]

    print(f"  [Triage] Running sector-aware news triage on {len(to_triage)} stocks...")

    for i, stock in enumerate(to_triage):
        ticker = stock.get("ticker", "?")
        sector = stock.get("sector", "Unknown")
        print(f"    [{i+1}/{len(to_triage)}] {ticker} ({sector})")
        triage_stock(stock, provider=provider, pause=pause)

    # Mark untriaged stocks as neutral
    for stock in rest:
        stock.update({
            "triage_sentiment":  "NEUTRAL",
            "triage_confidence": 0.0,
            "triage_catalyst":   "not triaged",
            "triage_risk":       "not triaged",
            "triage_signal_hits": [],
            "triage_summary":    "Not triaged (outside top_n).",
            "triage_news_count": 0,
        })

    all_stocks = to_triage + rest

    # Sort: POSITIVE first, then confidence, then market cap
    sentiment_order = {"POSITIVE": 0, "NEUTRAL": 1, "NEGATIVE": 2}
    all_stocks.sort(key=lambda s: (
        sentiment_order.get(s.get("triage_sentiment", "NEUTRAL"), 1),
        -(s.get("triage_confidence", 0)),
        -(s.get("market_cap_cr", 0)),
    ))

    pos  = sum(1 for s in to_triage if s.get("triage_sentiment") == "POSITIVE")
    neg  = sum(1 for s in to_triage if s.get("triage_sentiment") == "NEGATIVE")
    neut = len(to_triage) - pos - neg
    print(f"  [Triage] Done — POSITIVE: {pos}, NEUTRAL: {neut}, NEGATIVE: {neg}")

    return all_stocks


def format_for_llm(stock: dict) -> str:
    """
    One-line triage summary for inclusion in auto-trader LLM decision prompt.
    e.g.: "HDFCBANK.NS | FinSvc | P/E 18 | ROE 17% | [POSITIVE 0.8] Rate cut tailwind; Risk: NPA uptick"
    """
    ticker    = stock.get("ticker", "")
    sector    = (stock.get("sector") or "")[:12]
    pe        = stock.get("pe") or "N/A"
    roe       = stock.get("roe") or "N/A"
    rev_g     = stock.get("revenue_growth") or "N/A"
    sentiment = stock.get("triage_sentiment", "NEUTRAL")
    conf      = stock.get("triage_confidence", 0)
    catalyst  = stock.get("triage_catalyst", "none")
    risk      = stock.get("triage_risk", "none")
    summary   = stock.get("triage_summary", "")

    return (
        f"  {ticker} | {sector} | "
        f"P/E {pe} | ROE {roe}% | RevG {rev_g}% | "
        f"[{sentiment} {conf:.0%}] {summary} "
        f"Catalyst: {catalyst} | Risk: {risk}"
    )

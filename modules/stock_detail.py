"""Fetch stock price history, fundamentals, and generate AI thesis via Claude or Gemini."""
import json
import re
import requests
import yfinance as yf
import pandas as pd
from modules.llm_client import call_llm, _strip_fences


def get_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Return OHLCV DataFrame for a ticker (e.g. RELIANCE.NS)."""
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception as e:
        print(f"  Warning: Could not fetch price history for {ticker}: {e}")
        return pd.DataFrame()


def get_stock_info(ticker: str) -> dict:
    """Return detailed info dict for a ticker."""
    try:
        info = yf.Ticker(ticker).info
        return info
    except Exception:
        return {}


def generate_stock_thesis(
    ticker: str,
    company_name: str,
    theme: str,
    info: dict,
    provider: str = "claude",
) -> str:
    """Generate 2-paragraph investment thesis linking the stock to the macro theme."""
    fundamentals_summary = {
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap_cr": round((info.get("marketCap") or 0) / 1e7, 0),
        "pe": info.get("trailingPE"),
        "roe": info.get("returnOnEquity"),
        "revenue_growth_pct": info.get("revenueGrowth"),
        "earnings_growth_pct": info.get("earningsGrowth"),
        "debt_equity": info.get("debtToEquity"),
        "business_summary": (info.get("longBusinessSummary") or "")[:400],
    }

    prompt = f"""You are a top-down equity analyst focused on Indian markets (NSE/BSE).

Theme identified from macro news: "{theme}"

Stock: {company_name} ({ticker})
Fundamentals:
{json.dumps(fundamentals_summary, indent=2)}

Write exactly 2 short paragraphs (3-4 sentences each):
1. How this company is positioned to benefit from the macro theme, with specific business reasons
2. Key risks to this thesis and what to monitor

Be specific and crisp. No generic advice. Focus on the Indian market context."""

    try:
        return call_llm(prompt, provider=provider, mode="deep", max_tokens=500)
    except Exception as e:
        return f"Could not generate thesis: {e}"


_PERPLEXITY_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def fetch_perplexity_analysis(symbol: str) -> str:
    """
    Fetch AI-generated analysis from Perplexity Finance for an NSE stock.
    Tries NSE ticker format first (SYMBOL.NS), then plain symbol.
    Extracts meaningful text from the page via __NEXT_DATA__ JSON or HTML stripping.
    Returns a clean analysis string or empty string if unavailable.
    """
    urls = [
        f"https://www.perplexity.ai/finance/{symbol}.NS/analysis",
        f"https://www.perplexity.ai/finance/{symbol}/analysis",
    ]

    for url in urls:
        try:
            resp = requests.get(url, headers=_PERPLEXITY_HEADERS, timeout=10)
            if resp.status_code != 200:
                continue

            html = resp.text

            # ── Try __NEXT_DATA__ JSON (Next.js SSR payload) ──────────────────
            m = re.search(
                r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL
            )
            if m:
                try:
                    nd = json.loads(m.group(1))
                    props = nd.get("props", {}).get("pageProps", {})
                    # Perplexity may nest analysis under various keys
                    for key in ("analysis", "summary", "description", "content", "text"):
                        val = props.get(key)
                        if val and isinstance(val, str) and len(val) > 80:
                            return val[:700]
                    # If props itself is textual, stringify it
                    props_str = json.dumps(props)
                    if len(props_str) > 200:
                        # Extract first sizeable text value
                        texts = re.findall(r'"([^"]{80,})"', props_str)
                        if texts:
                            return texts[0][:700]
                except Exception:
                    pass

            # ── Fallback: strip HTML and locate the analysis paragraph ────────
            text = re.sub(
                r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE
            )
            text = re.sub(
                r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE
            )
            text = re.sub(r"<[^>]+>", " ", text)
            text = " ".join(text.split())

            # Seek the first financial keyword to skip navigation boilerplate
            for kw in ("revenue", "earnings", "profit", "analyst", "quarter", "growth", "margin"):
                idx = text.lower().find(kw)
                if idx > 150:
                    snippet = text[max(0, idx - 80): idx + 600].strip()
                    if len(snippet) > 100:
                        return snippet

            # Last resort: mid-page slice (avoids header/footer noise)
            if len(text) > 600:
                return text[300:900].strip()

        except Exception as e:
            print(f"  Perplexity fetch failed for {symbol} ({url}): {e}")
            continue

    return ""


def fetch_stock_web_context(ticker: str, screener_data: dict) -> str:
    """
    Enrich a stock with live web data: recent news headlines, analyst consensus,
    price target, and 52-week momentum — all sourced from Yahoo Finance via yfinance.
    Returns a formatted string for inclusion in the LLM prompt.
    """
    lines = []
    try:
        t = yf.Ticker(ticker)

        # ── Recent news (Yahoo Finance via yfinance) ──────────────────────────
        news = t.news or []
        headlines = [
            f"  - [{n.get('publisher', '?')}] {n.get('title', '')}"
            for n in news[:4]
            if n.get("title")
        ]
        if headlines:
            lines.append("Recent news:\n" + "\n".join(headlines))

        # ── Analyst consensus + price target ─────────────────────────────────
        info = t.info
        rec = (info.get("recommendationKey") or "").replace("_", " ").title()
        target = info.get("targetMeanPrice")
        n_analysts = info.get("numberOfAnalystOpinions")
        curr = screener_data.get("current_price") or info.get("currentPrice")

        if rec or target:
            upside = (
                f" | Upside: {((target - curr) / curr * 100):+.1f}%"
                if (target and curr and curr > 0)
                else ""
            )
            lines.append(
                f"Analyst view: {rec or 'N/A'}"
                + (f" | Target ₹{target:.0f}" if target else "")
                + upside
                + (f" | {n_analysts} analysts covering" if n_analysts else "")
            )

        # ── 52-week price momentum ────────────────────────────────────────────
        try:
            yc = t.fast_info.year_change
            if yc is not None:
                lines.append(f"52-week return: {yc * 100:+.1f}%")
        except Exception:
            pass

        # ── Institutional ownership ───────────────────────────────────────────
        inst = info.get("heldPercentInstitutions")
        if inst:
            lines.append(f"Institutional holding: {inst * 100:.1f}%")

    except Exception as e:
        print(f"  Warning: Could not fetch yfinance context for {ticker}: {e}")

    # ── Perplexity Finance AI analysis (best-effort) ──────────────────────────
    symbol = ticker.replace(".NS", "").replace(".BO", "")
    perplexity_text = fetch_perplexity_analysis(symbol)
    if perplexity_text:
        lines.append(f"Perplexity Finance analysis:\n  {perplexity_text.replace(chr(10), ' ').strip()[:600]}")

    return "\n".join(lines)


def summarize_top_picks(
    stocks: list[dict],
    theme_name: str,
    provider: str = "claude",
) -> list[dict]:
    """
    Given a screened stock list, ask the LLM to highlight the 2-3 most compelling picks.
    Enriches top 6 stocks with live Yahoo Finance data (news, analyst views, momentum)
    before sending to the LLM for a more informed selection.
    Returns list of {ticker, company_name, reason} dicts.
    """
    if not stocks:
        return []

    # ── Fetch live web context for top 6 stocks by market cap ────────────────
    top_for_web = sorted(stocks, key=lambda s: s.get("market_cap_cr", 0), reverse=True)[:6]
    web_contexts: dict[str, str] = {}
    for s in top_for_web:
        print(f"  Fetching web context for {s['ticker']}...")
        ctx = fetch_stock_web_context(s["ticker"], s)
        if ctx:
            web_contexts[s["ticker"]] = ctx

    # ── Build stock summaries with web context embedded ───────────────────────
    stock_blocks = []
    for s in stocks:
        block = (
            f"• {s['company_name']} ({s['ticker']})\n"
            f"  Sector={s.get('sector', '?')} | "
            f"MCap=₹{s.get('market_cap_cr', 0):.0f}Cr | "
            f"P/E={s.get('pe') or 'N/A'} | "
            f"ROE={s.get('roe', 0):.1f}% | "
            f"RevGrowth={s.get('revenue_growth', 0):.1f}% | "
            f"D/E={s.get('debt_equity', 0):.2f}"
        )
        if s["ticker"] in web_contexts:
            block += "\n  " + web_contexts[s["ticker"]].replace("\n", "\n  ")
        stock_blocks.append(block)

    stock_section = "\n\n".join(stock_blocks)

    prompt = f"""You are an equity analyst focused on Indian markets (NSE/BSE).
The following stocks were screened for the macro theme: "{theme_name}".
For the top candidates, live data has been gathered from multiple sources:
- Yahoo Finance: recent news headlines, analyst consensus, price targets, 52-week momentum, institutional ownership
- Perplexity Finance: AI-generated company analysis and market context (where available)

{stock_section}

Identify the 2-3 most compelling picks. Prioritize stocks with:
- Strong revenue/earnings growth
- Reasonable valuation (low P/E vs peers)
- High ROE with manageable debt
- Direct business fit to the theme
- Positive analyst consensus or strong price momentum where available
- Any notable catalyst mentioned in the news or Perplexity analysis

For each pick write 2-3 sentences that cite SPECIFIC numbers and data points from the information above.
Do not make generic statements — reference actual figures (P/E, ROE, analyst target, news event, etc.).

Respond ONLY with valid JSON:
{{
  "picks": [
    {{
      "ticker": "TICKER.NS",
      "company_name": "Full Company Name",
      "reason": "Specific 2-3 sentence reason with actual data points."
    }}
  ]
}}"""

    try:
        raw = call_llm(prompt, provider=provider, mode="fast", max_tokens=800)
        raw = _strip_fences(raw)
        data = json.loads(raw)
        return data.get("picks", [])
    except Exception as e:
        print(f"  Warning: Could not generate top picks: {e}")
        return []


def get_key_metrics(info: dict) -> dict:
    """Extract display-ready key metrics from yfinance info."""
    def pct(val):
        if val is None:
            return "N/A"
        return f"{val * 100:.1f}%"

    def num(val, decimals=2):
        if val is None:
            return "N/A"
        return f"{val:.{decimals}f}"

    market_cap = info.get("marketCap")
    if market_cap:
        cap_cr = market_cap / 1e7
        if cap_cr >= 10000:
            cap_str = f"₹{cap_cr/1000:.1f}K Cr"
        else:
            cap_str = f"₹{cap_cr:.0f} Cr"
    else:
        cap_str = "N/A"

    return {
        "Market Cap": cap_str,
        "Current Price": f"₹{info.get('currentPrice') or info.get('regularMarketPrice') or 'N/A'}",
        "52W High": f"₹{info.get('fiftyTwoWeekHigh', 'N/A')}",
        "52W Low": f"₹{info.get('fiftyTwoWeekLow', 'N/A')}",
        "P/E (TTM)": num(info.get("trailingPE")),
        "P/B": num(info.get("priceToBook")),
        "ROE": pct(info.get("returnOnEquity")),
        "Debt/Equity": num((info.get("debtToEquity") or 0) / 100),
        "Revenue Growth": pct(info.get("revenueGrowth")),
        "EPS Growth": pct(info.get("earningsGrowth")),
        "EPS (TTM)": num(info.get("trailingEps")),
        "Dividend Yield": pct(info.get("dividendYield")),
        "Sector": info.get("sector", "N/A"),
        "Industry": info.get("industry", "N/A"),
    }

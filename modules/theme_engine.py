"""Extract macro investment themes from news headlines using the configured LLM."""
import json
from datetime import date
from modules.db import get_conn
from modules.news_fetcher import get_recent_headlines
from modules.llm_client import call_llm, _strip_fences
from config import NSE_SECTORS

SYSTEM_PROMPT = """You are a macro investment analyst specializing in Indian equity markets (NSE/BSE).
Your job is to identify actionable investment themes from recent Indian financial news.
Focus on themes that could drive stock returns over the next 6-18 months.
Always respond with valid JSON only."""

THEME_EXTRACTION_PROMPT = """Analyze these recent Indian financial news headlines and extract 5-8 distinct macro investment themes.

For each theme:
1. Give it a clear, specific name (e.g., "Defence sector indigenisation", "Electric vehicle adoption", "Data center infrastructure boom")
2. Rate your confidence 0.0-1.0 based on news evidence strength
3. List 2-4 specific news headlines/facts supporting it
4. Map it to 1-3 NSE sectors most likely to benefit

NSE Sectors available: {sectors}

News Headlines (last 48 hours):
{headlines}

Respond ONLY with this JSON structure:
{{
  "themes": [
    {{
      "theme": "Theme name",
      "confidence": 0.85,
      "evidence": ["headline 1", "headline 2"],
      "sectors": ["CAPITAL GOODS", "POWER"]
    }}
  ]
}}"""


def extract_themes(force_refresh: bool = False, provider: str = "claude") -> list[dict]:
    """
    Extract themes from today's news. Returns cached result if already done today.
    provider: "claude" | "gemini"
    """
    today = date.today().isoformat()
    conn = get_conn()
    c = conn.cursor()

    # Return cached themes for today (cache is provider-agnostic — same news, same themes)
    rows = c.execute(
        "SELECT * FROM themes WHERE session_date = %s ORDER BY confidence DESC", (today,)
    ).fetchall()
    if rows:
        conn.close()
        return [
            {
                "id": r["id"],
                "theme": r["theme"],
                "confidence": r["confidence"],
                "evidence": json.loads(r["evidence"]),
                "sectors": json.loads(r["sectors"]),
            }
            for r in rows
        ]

    # No cache — only proceed if explicitly requested
    if not force_refresh:
        conn.close()
        return []

    headlines = get_recent_headlines(hours_back=48, limit=150)
    if not headlines:
        conn.close()
        return []

    headline_text = "\n".join(
        f"- [{h['source']}] {h['title']}" + (f": {h['summary'][:100]}" if h["summary"] else "")
        for h in headlines[:120]
    )

    prompt = THEME_EXTRACTION_PROMPT.format(
        sectors=", ".join(NSE_SECTORS),
        headlines=headline_text,
    )

    provider_name = "Gemini 2.5 Flash Lite" if provider == "gemini" else "Claude Haiku"
    print(f"  Calling {provider_name} for theme extraction...")
    raw = call_llm(prompt, system=SYSTEM_PROMPT, provider=provider, mode="fast", max_tokens=2000)
    raw = _strip_fences(raw)

    data = json.loads(raw)
    themes = data.get("themes", [])

    c.execute("DELETE FROM themes WHERE session_date = %s", (today,))
    theme_records = []
    for t in themes:
        c.execute("""
            INSERT INTO themes (session_date, theme, confidence, evidence, sectors)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (today, t["theme"], t["confidence"],
              json.dumps(t["evidence"]), json.dumps(t["sectors"])))
        t["id"] = c.fetchone()["id"]
        theme_records.append(t)

    conn.commit()
    conn.close()
    print(f"  Extracted {len(theme_records)} themes")
    return theme_records


if __name__ == "__main__":
    from modules.db import init_db
    from modules.news_fetcher import fetch_all_feeds
    import sys
    provider = sys.argv[1] if len(sys.argv) > 1 else "claude"
    init_db()
    print("Fetching news...")
    fetch_all_feeds()
    print(f"Extracting themes with {provider}...")
    themes = extract_themes(force_refresh=True, provider=provider)
    for t in themes:
        print(f"\n  [{t['confidence']:.0%}] {t['theme']}")
        print(f"    Sectors: {', '.join(t['sectors'])}")
        for ev in t["evidence"]:
            print(f"    - {ev}")

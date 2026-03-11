"""Fetch Indian financial news from free RSS feeds and store in SQLite."""
import re
import sqlite3
import feedparser
import requests
from datetime import datetime, timedelta
from modules.db import get_conn
from config import RSS_FEEDS

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def fetch_all_feeds(hours_back: int = 48) -> int:
    """Fetch and store news from all configured RSS feeds. Returns count of new articles."""
    conn = get_conn()
    c = conn.cursor()
    new_count = 0

    for feed_cfg in RSS_FEEDS:
        try:
            # Use requests with timeout, then hand content to feedparser
            resp = requests.get(feed_cfg["url"], headers=HEADERS, timeout=8)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except Exception as e:
            print(f"  Warning: Could not fetch {feed_cfg['name']}: {e}")
            continue

        for entry in feed.entries:
            title = entry.get("title", "").strip()
            summary = entry.get("summary", entry.get("description", "")).strip()
            link = entry.get("link", "")
            pub = entry.get("published", entry.get("updated", ""))

            if not title:
                continue

            # Strip HTML from summary
            summary = re.sub(r"<[^>]+>", " ", summary).strip()
            summary = " ".join(summary.split())[:500]

            try:
                c.execute("""
                    INSERT OR IGNORE INTO news (source, title, summary, link, published_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (feed_cfg["name"], title, summary, link, pub))
                if c.rowcount > 0:
                    new_count += 1
            except sqlite3.Error:
                pass

    conn.commit()
    conn.close()
    return new_count


def get_recent_headlines(hours_back: int = 48, limit: int = 200) -> list[dict]:
    """Return recent headlines from DB for LLM processing."""
    conn = get_conn()
    c = conn.cursor()
    # Use SQLite-compatible format (space separator, not 'T') so string comparison works
    cutoff = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y-%m-%d %H:%M:%S")

    rows = c.execute("""
        SELECT source, title, summary
        FROM news
        WHERE fetched_at >= ?
        ORDER BY fetched_at DESC
        LIMIT ?
    """, (cutoff, limit)).fetchall()

    conn.close()
    return [{"source": r["source"], "title": r["title"], "summary": r["summary"]} for r in rows]


if __name__ == "__main__":
    from modules.db import init_db
    init_db()
    print("Fetching news from RSS feeds...")
    n = fetch_all_feeds()
    print(f"  Stored {n} new articles")
    headlines = get_recent_headlines()
    print(f"  Total recent headlines available: {len(headlines)}")
    for h in headlines[:5]:
        print(f"  [{h['source']}] {h['title'][:80]}")

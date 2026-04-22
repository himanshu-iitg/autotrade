"""
Technical analysis scoring layer — sits on top of the fundamental screener.

For each stock that passes fundamental filters, this module computes a
technical score (0–100) based on:
  - RSI (14-day): avoids overbought stocks
  - 52-week range position: prefers stocks recovering from lows
  - Price momentum: 1-month and 3-month returns
  - Moving average trend: price vs 50-day and 200-day SMA
  - Volume trend: recent volume vs 20-day average

A combined score helps the LLM make better buy decisions by surfacing
stocks with improving technicals, not just good fundamentals.
"""
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def _safe(val, default=None):
    try:
        f = float(val)
        return None if (f != f) else f  # NaN → None
    except (TypeError, ValueError):
        return default


def compute_rsi(closes: pd.Series, period: int = 14) -> float | None:
    """Compute RSI for the given close price series."""
    if len(closes) < period + 1:
        return None
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi_series = 100 - (100 / (1 + rs))
    last = rsi_series.dropna()
    return round(float(last.iloc[-1]), 1) if not last.empty else None


def get_technical_score(ticker: str, pause: float = 0.2) -> dict:
    """
    Fetch 1 year of daily OHLCV data and compute a technical score.

    Returns a dict with:
      - rsi: 14-day RSI (None if not enough data)
      - week52_pos: position in 52-week range (0 = at low, 100 = at high)
      - mom_1m: 1-month price return %
      - mom_3m: 3-month price return %
      - above_50d_sma: True/False
      - above_200d_sma: True/False
      - vol_ratio: recent 5-day avg volume / 20-day avg volume
      - tech_score: composite 0–100 score
      - tech_signal: "BUY" / "NEUTRAL" / "AVOID"
    """
    result = {
        "rsi": None,
        "week52_pos": None,
        "mom_1m": None,
        "mom_3m": None,
        "above_50d_sma": None,
        "above_200d_sma": None,
        "vol_ratio": None,
        "tech_score": 50,       # default neutral
        "tech_signal": "NEUTRAL",
    }

    try:
        time.sleep(pause)
        hist = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if hist.empty or len(hist) < 30:
            return result

        closes = hist["Close"].squeeze()
        volumes = hist["Volume"].squeeze()
        current = float(closes.iloc[-1])
        score = 0.0
        weights = 0.0

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi = compute_rsi(closes)
        result["rsi"] = rsi
        if rsi is not None:
            # RSI 40–65: sweet spot — not oversold panic, not overbought froth
            if 40 <= rsi <= 65:
                score += 25
            elif 30 <= rsi < 40 or 65 < rsi <= 75:
                score += 12
            elif rsi < 30:
                score += 5   # severely oversold — could be value or falling knife
            else:
                score += 0   # >75: overbought, avoid
            weights += 25

        # ── 52-week range position ────────────────────────────────────────────
        high_52 = float(closes.max())
        low_52  = float(closes.min())
        if high_52 > low_52:
            pos = (current - low_52) / (high_52 - low_52) * 100
            result["week52_pos"] = round(pos, 1)
            # Prefer stocks in 20–70% of their range (recovering, not already at highs)
            if 20 <= pos <= 70:
                score += 20
            elif 70 < pos <= 85:
                score += 10
            elif pos < 20:
                score += 8   # near 52-week low — could be value or downtrend
            else:
                score += 2   # near 52-week high — limited upside for new entry
            weights += 20

        # ── Price momentum ────────────────────────────────────────────────────
        if len(closes) >= 22:
            mom_1m = (current / float(closes.iloc[-22]) - 1) * 100
            result["mom_1m"] = round(mom_1m, 2)
            # Mild positive momentum is best (strong trend, not parabolic)
            if 0 < mom_1m <= 15:
                score += 15
            elif 15 < mom_1m <= 30:
                score += 8
            elif -10 <= mom_1m <= 0:
                score += 10   # slight dip — possible entry
            elif mom_1m < -10:
                score += 2
            else:
                score += 0   # >30% in 1 month: momentum too extended
            weights += 15

        if len(closes) >= 66:
            mom_3m = (current / float(closes.iloc[-66]) - 1) * 100
            result["mom_3m"] = round(mom_3m, 2)
            # Positive 3-month trend preferred
            if 5 < mom_3m <= 40:
                score += 15
            elif 0 < mom_3m <= 5:
                score += 10
            elif -15 <= mom_3m <= 0:
                score += 7
            else:
                score += 2
            weights += 15

        # ── Moving averages ───────────────────────────────────────────────────
        if len(closes) >= 50:
            sma50 = float(closes.rolling(50).mean().iloc[-1])
            above_50 = current > sma50
            result["above_50d_sma"] = above_50
            score += 10 if above_50 else 2
            weights += 10

        if len(closes) >= 200:
            sma200 = float(closes.rolling(200).mean().iloc[-1])
            above_200 = current > sma200
            result["above_200d_sma"] = above_200
            score += 10 if above_200 else 2
            weights += 10

        # ── Volume trend ──────────────────────────────────────────────────────
        if len(volumes) >= 20:
            vol_20d = float(volumes.iloc[-20:].mean())
            vol_5d  = float(volumes.iloc[-5:].mean())
            if vol_20d > 0:
                vol_ratio = vol_5d / vol_20d
                result["vol_ratio"] = round(vol_ratio, 2)
                # Rising volume with rising price = accumulation = bullish
                if vol_ratio > 1.2:
                    score += 5
                elif vol_ratio > 0.8:
                    score += 3
                else:
                    score += 0   # declining volume = distribution / low interest
                weights += 5

        # ── Composite score (scale to 0–100) ──────────────────────────────────
        if weights > 0:
            tech_score = round((score / weights) * 100)
            result["tech_score"] = max(0, min(100, tech_score))

        # Signal
        ts = result["tech_score"]
        if ts >= 65:
            result["tech_signal"] = "BUY"
        elif ts >= 40:
            result["tech_signal"] = "NEUTRAL"
        else:
            result["tech_signal"] = "AVOID"

    except Exception as e:
        print(f"  [Technical] Error for {ticker}: {e}")

    return result


def enrich_with_technicals(stocks: list[dict], pause: float = 0.25) -> list[dict]:
    """
    Add technical score fields to a list of screened stock dicts.
    Modifies dicts in-place and returns the same list, sorted by
    a combined fundamental + technical score.
    """
    print(f"  [Technical] Computing indicators for {len(stocks)} stocks...")

    for stock in stocks:
        ticker = stock.get("ticker", "")
        if not ticker:
            continue
        tech = get_technical_score(ticker, pause=pause)
        stock.update({
            "rsi":            tech["rsi"],
            "week52_pos":     tech["week52_pos"],
            "mom_1m":         tech["mom_1m"],
            "mom_3m":         tech["mom_3m"],
            "above_50d_sma":  tech["above_50d_sma"],
            "above_200d_sma": tech["above_200d_sma"],
            "vol_ratio":      tech["vol_ratio"],
            "tech_score":     tech["tech_score"],
            "tech_signal":    tech["tech_signal"],
        })

    # Sort by combined score: fundamental rank (market_cap proxy) + technical
    # We blend equal weight: tech_score contributes 40%, market cap rank 60%
    n = len(stocks)
    if n > 1:
        # Rank by market cap (higher = better rank)
        sorted_by_mcap = sorted(stocks, key=lambda s: s.get("market_cap_cr", 0), reverse=True)
        mcap_rank = {s["ticker"]: (n - i) / n * 100 for i, s in enumerate(sorted_by_mcap)}
        for s in stocks:
            s["combined_score"] = round(
                0.6 * mcap_rank.get(s["ticker"], 50) + 0.4 * s.get("tech_score", 50), 1
            )
        stocks.sort(key=lambda s: s.get("combined_score", 0), reverse=True)

    print(f"  [Technical] Done. Top pick by combined score: "
          f"{stocks[0]['ticker'] if stocks else 'N/A'}")
    return stocks


def format_tech_summary(stock: dict) -> str:
    """Return a one-line technical summary string for LLM prompts."""
    parts = []
    if stock.get("rsi") is not None:
        parts.append(f"RSI={stock['rsi']:.0f}")
    if stock.get("week52_pos") is not None:
        parts.append(f"52w-pos={stock['week52_pos']:.0f}%")
    if stock.get("mom_1m") is not None:
        parts.append(f"mom1m={stock['mom_1m']:+.1f}%")
    if stock.get("mom_3m") is not None:
        parts.append(f"mom3m={stock['mom_3m']:+.1f}%")
    if stock.get("above_200d_sma") is not None:
        parts.append("above200SMA" if stock["above_200d_sma"] else "below200SMA")
    signal = stock.get("tech_signal", "")
    score  = stock.get("tech_score", "")
    return f"[{signal} {score}/100] " + " | ".join(parts)

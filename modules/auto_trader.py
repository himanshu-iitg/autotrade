"""
Automatic paper trader — runs the full pipeline end-to-end and makes
LLM-driven buy/sell decisions without human intervention.

Capital: ₹5,00,000  |  Max positions: 5  |  Stop-loss: -15%
"""
import json
import math
import yfinance as yf
from datetime import date
from modules.db import get_conn
from modules.news_fetcher import fetch_all_feeds
from modules.theme_engine import extract_themes
from modules.screener import screen_stocks_for_theme
from modules.llm_client import call_llm, _strip_fences

CAPITAL = 500_000        # ₹5,00,000 paper capital
MAX_POSITIONS = 5        # Hard cap on simultaneous holdings
STOP_LOSS_PCT = 15.0     # Trigger sell if position down > 15%
MIN_BUY_CASH = 50_000    # Don't open a position with < ₹50k available
MAX_PER_POSITION = 100_000  # Max ₹1,00,000 allocated per stock

DECISION_SYSTEM = (
    "You are a long-term equity portfolio manager for Indian markets (NSE). "
    "Your goal is capital appreciation over 6–18 months. Avoid frequent churn — "
    "only sell if fundamentals have clearly deteriorated or a materially better "
    "opportunity exists. Respond ONLY with valid JSON."
)


class AutoTrader:
    def __init__(self, portfolio_id: int | None = None):
        self.portfolio_id = portfolio_id

    # ── Portfolio setup ───────────────────────────────────────────────────────

    def ensure_portfolio(self, capital: float = CAPITAL) -> int:
        """Return active portfolio ID, creating one if none exists."""
        conn = get_conn()
        c = conn.cursor()
        row = c.execute(
            "SELECT id FROM auto_portfolio WHERE status = 'active' LIMIT 1"
        ).fetchone()
        if row:
            self.portfolio_id = row["id"]
            conn.close()
            return self.portfolio_id
        c.execute(
            "INSERT INTO auto_portfolio (name, capital, cash_remaining, status) VALUES (%s, %s, %s, 'active') RETURNING id",
            ("Auto Trader v1", capital, capital),
        )
        self.portfolio_id = c.fetchone()["id"]
        conn.commit()
        conn.close()
        return self.portfolio_id

    # ── Read state ────────────────────────────────────────────────────────────

    def get_portfolio_state(self) -> dict:
        """Return current positions with live prices/P&L and cash balance."""
        conn = get_conn()
        c = conn.cursor()
        port = dict(c.execute(
            "SELECT * FROM auto_portfolio WHERE id = %s", (self.portfolio_id,)
        ).fetchone())
        positions_raw = c.execute(
            "SELECT * FROM auto_positions WHERE portfolio_id = %s AND status = 'open'",
            (self.portfolio_id,),
        ).fetchall()
        conn.close()

        positions = []
        total_current_value = 0.0

        for p in positions_raw:
            p = dict(p)
            try:
                fi = yf.Ticker(p["ticker"]).fast_info
                current_price = fi.last_price or fi.regular_market_price or p["entry_price"]
            except Exception:
                current_price = p["entry_price"]

            cost = p["shares"] * p["entry_price"]
            current_val = p["shares"] * current_price
            pnl = current_val - cost
            pnl_pct = (pnl / cost * 100) if cost else 0
            days_held = (date.today() - date.fromisoformat(p["entry_date"])).days

            total_current_value += current_val
            positions.append({
                **p,
                "current_price": round(current_price, 2),
                "current_value": round(current_val, 2),
                "cost_basis": round(cost, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "days_held": days_held,
            })

        total_value = total_current_value + port["cash_remaining"]
        total_pnl = total_value - port["capital"]
        total_pnl_pct = (total_pnl / port["capital"] * 100) if port["capital"] else 0

        try:
            nifty = yf.Ticker("^NSEI").fast_info.last_price
        except Exception:
            nifty = None

        return {
            "portfolio": port,
            "positions": positions,
            "cash_remaining": port["cash_remaining"],
            "total_value": round(total_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "nifty_current": nifty,
        }

    def get_trade_history(self) -> list[dict]:
        """All auto trades ordered newest first."""
        conn = get_conn()
        c = conn.cursor()
        rows = c.execute(
            "SELECT * FROM auto_trades WHERE portfolio_id = %s ORDER BY trade_date DESC, id DESC",
            (self.portfolio_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_equity_history(self) -> list[dict]:
        """Run history for equity curve (one portfolio_value per completed run)."""
        conn = get_conn()
        c = conn.cursor()
        rows = c.execute(
            """SELECT run_date, portfolio_value, llm_summary
               FROM auto_runs
               WHERE portfolio_id = %s AND portfolio_value IS NOT NULL
               ORDER BY run_date, id""",
            (self.portfolio_id,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_last_run(self) -> dict | None:
        """Most recent run record."""
        conn = get_conn()
        c = conn.cursor()
        row = c.execute(
            "SELECT * FROM auto_runs WHERE portfolio_id = %s ORDER BY id DESC LIMIT 1",
            (self.portfolio_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run_pipeline(self) -> dict:
        """
        Execute the full auto-trading pipeline:
          1. Fetch news
          2. Extract themes (Claude Haiku)
          3. Screen stocks for top 2 themes
          4. Apply hard stop-losses
          5. Ask Claude Sonnet for buy/sell decisions
          6. Execute trades
          7. Save run log
        Returns summary dict with buys/sells/stop_losses/errors lists.
        """
        pid = self.ensure_portfolio()
        today = date.today().isoformat()
        log: dict = {"buys": [], "sells": [], "stop_losses": [], "summary": "", "errors": []}

        # Create run record (will be updated at the end)
        conn = get_conn()
        c = conn.cursor()
        c.execute(
            "INSERT INTO auto_runs (portfolio_id, run_date) VALUES (%s, %s) RETURNING id",
            (pid, today),
        )
        run_id = c.fetchone()["id"]
        conn.commit()
        conn.close()

        try:
            # ── 1. Fetch news ─────────────────────────────────────────────────
            print("[AutoTrader] Fetching news...")
            n_articles = fetch_all_feeds()
            print(f"  {n_articles} new articles fetched")

            # ── 2. Extract themes ─────────────────────────────────────────────
            print("[AutoTrader] Extracting themes...")
            themes = extract_themes(force_refresh=True, provider="gemini")
            if not themes:
                raise RuntimeError("No themes extracted — news feeds may be empty")
            print(f"  {len(themes)} themes found")

            # ── 3. Screen top 2 themes ────────────────────────────────────────
            print("[AutoTrader] Screening stocks for top themes...")
            all_candidates: list[dict] = []
            top_themes = sorted(themes, key=lambda t: t["confidence"], reverse=True)[:2]

            for theme in top_themes:
                stocks = screen_stocks_for_theme(
                    theme_id=theme["id"],
                    theme_name=theme["theme"],
                    sectors=theme["sectors"],
                    force=True,
                    max_stocks=20,
                )
                for s in stocks:
                    s["theme"] = theme["theme"]
                all_candidates.extend(stocks)
                print(f"  '{theme['theme']}': {len(stocks)} stocks")

            # Deduplicate by ticker (keep first/highest-confidence theme)
            seen: set[str] = set()
            unique_candidates: list[dict] = []
            for s in all_candidates:
                if s["ticker"] not in seen:
                    seen.add(s["ticker"])
                    unique_candidates.append(s)

            # ── 4. Hard stop-losses ───────────────────────────────────────────
            print("[AutoTrader] Checking stop-losses...")
            state = self.get_portfolio_state()
            stop_loss_tickers: set[str] = set()

            for pos in state["positions"]:
                if pos["pnl_pct"] <= -STOP_LOSS_PCT:
                    print(f"  STOP LOSS: {pos['ticker']} at {pos['pnl_pct']:.1f}%")
                    self._sell_position(
                        pos, pos["current_price"], run_id,
                        f"Hard stop-loss triggered at {pos['pnl_pct']:.1f}%",
                        action="stop_loss",
                    )
                    stop_loss_tickers.add(pos["ticker"])
                    log["stop_losses"].append(pos["ticker"])

            # Refresh state after stop-losses
            state = self.get_portfolio_state()

            # ── 5. LLM decisions ──────────────────────────────────────────────
            print("[AutoTrader] Asking Claude for buy/sell decisions...")
            open_positions = [p for p in state["positions"] if p["ticker"] not in stop_loss_tickers]
            decisions = self._make_decisions(unique_candidates, open_positions, state["cash_remaining"])
            log["summary"] = decisions.get("summary", "")

            # ── 6a. Sells ─────────────────────────────────────────────────────
            for sell in decisions.get("sells", []):
                ticker = sell.get("ticker", "")
                pos = next((p for p in open_positions if p["ticker"] == ticker), None)
                if pos:
                    print(f"  SELL: {ticker}")
                    self._sell_position(pos, pos["current_price"], run_id, sell.get("reason", ""))
                    log["sells"].append(ticker)

            # Refresh state
            state = self.get_portfolio_state()
            open_count = len(state["positions"])
            cash = state["cash_remaining"]

            # Tickers already held (don't double-buy)
            held_tickers = {p["ticker"] for p in state["positions"]}

            # ── 6b. Buys ──────────────────────────────────────────────────────
            for buy in decisions.get("buys", []):
                ticker = buy.get("ticker", "")

                if open_count >= MAX_POSITIONS:
                    print(f"  SKIP BUY {ticker}: max {MAX_POSITIONS} positions reached")
                    break
                if cash < MIN_BUY_CASH:
                    print(f"  SKIP BUY {ticker}: insufficient cash (₹{cash:,.0f})")
                    break
                if ticker in held_tickers:
                    print(f"  SKIP BUY {ticker}: already held")
                    continue

                candidate = next((s for s in unique_candidates if s["ticker"] == ticker), None)
                if not candidate:
                    continue

                # Live price
                try:
                    fi = yf.Ticker(ticker).fast_info
                    price = fi.last_price or fi.regular_market_price or candidate["current_price"]
                except Exception:
                    price = candidate["current_price"]

                if not price or price <= 0:
                    continue

                # Equal-weight among remaining empty slots
                slots_remaining = max(MAX_POSITIONS - open_count, 1)
                budget = min(cash / slots_remaining, MAX_PER_POSITION)
                shares = math.floor(budget / price)

                if shares < 1:
                    print(f"  SKIP BUY {ticker}: price ₹{price:.0f} too high for budget ₹{budget:.0f}")
                    continue

                cost = shares * price
                self._buy_position(
                    ticker=ticker,
                    company_name=candidate.get("company_name", ""),
                    shares=shares,
                    price=price,
                    run_id=run_id,
                    reason=buy.get("reason", ""),
                )
                log["buys"].append(ticker)
                held_tickers.add(ticker)
                cash -= cost
                open_count += 1
                print(f"  BUY: {ticker} x{shares} @ ₹{price:.2f} (₹{cost:,.0f})")

            # ── 7. Save run log ───────────────────────────────────────────────
            final_state = self.get_portfolio_state()
            conn = get_conn()
            c = conn.cursor()
            c.execute("""
                UPDATE auto_runs SET
                    themes_found = %s,
                    stocks_screened = %s,
                    buys_made = %s,
                    sells_made = %s,
                    stop_losses_triggered = %s,
                    portfolio_value = %s,
                    llm_summary = %s,
                    completed_at = %s
                WHERE id = %s
            """, (
                len(themes), len(unique_candidates),
                len(log["buys"]), len(log["sells"]), len(log["stop_losses"]),
                final_state["total_value"], log["summary"],
                __import__('datetime').datetime.now().isoformat(), run_id,
            ))
            conn.commit()
            conn.close()

            print(
                f"[AutoTrader] Done. "
                f"Portfolio: ₹{final_state['total_value']:,.0f} | "
                f"P&L: {final_state['total_pnl_pct']:+.2f}%"
            )

        except Exception as e:
            log["errors"].append(str(e))
            print(f"[AutoTrader] ERROR: {e}")
            conn = get_conn()
            c = conn.cursor()
            c.execute(
                "UPDATE auto_runs SET llm_summary = %s, completed_at = %s WHERE id = %s",
                (f"ERROR: {e}", __import__('datetime').datetime.now().isoformat(), run_id),
            )
            conn.commit()
            conn.close()

        return log

    # ── LLM decision-making ───────────────────────────────────────────────────

    def _make_decisions(
        self,
        candidates: list[dict],
        positions: list[dict],
        cash: float,
    ) -> dict:
        """Call Claude Sonnet to decide which stocks to buy, sell, or hold."""
        pos_lines = []
        for p in positions:
            pos_lines.append(
                f"  {p['ticker']} | {p['company_name'][:25]} | "
                f"entry ₹{p['entry_price']:.0f} | now ₹{p['current_price']:.0f} | "
                f"{p['pnl_pct']:+.1f}% | {p['days_held']}d held"
            )
        pos_text = "\n".join(pos_lines) if pos_lines else "  (no open positions)"

        # Top 12 candidates by market cap for the prompt
        top_candidates = sorted(candidates, key=lambda x: x.get("market_cap_cr", 0), reverse=True)[:12]
        cand_lines = []
        for s in top_candidates:
            cand_lines.append(
                f"  {s['ticker']} | {s['company_name'][:25]} | {s.get('theme','')[:30]} | "
                f"P/E {s.get('pe') or 'N/A'} | ROE {s.get('roe') or 'N/A'}% | "
                f"RevGrowth {s.get('revenue_growth') or 'N/A'}% | MCap ₹{s.get('market_cap_cr',0):.0f}Cr"
            )
        cand_text = "\n".join(cand_lines) if cand_lines else "  (no candidates found)"

        open_slots = MAX_POSITIONS - len(positions)

        prompt = f"""PORTFOLIO (₹{CAPITAL:,.0f} base capital):
Cash available: ₹{cash:,.0f}
Open positions: {len(positions)}/{MAX_POSITIONS} (can add {open_slots} more)

CURRENT HOLDINGS:
{pos_text}

TODAY'S SCREENER RESULTS (top candidates by market cap):
{cand_text}

DECISION RULES:
- Long-term focus (6–18 months): do NOT sell just because a stock is down short-term
- Sell only if: clearly no longer aligns with any theme AND held > 30 days, OR fundamentals significantly deteriorated
- Buy only if open_slots > 0 and cash >= ₹{MIN_BUY_CASH:,.0f}
- Do NOT recommend buying stocks already in holdings
- Prefer large-cap, quality stocks (high ROE, revenue growth, manageable P/E)

Return ONLY valid JSON (no markdown fences):
{{
  "sells": [{{"ticker": "X.NS", "reason": "brief specific reason"}}],
  "buys": [{{"ticker": "X.NS", "reason": "brief specific reason"}}],
  "holds": [{{"ticker": "X.NS", "reason": "brief specific reason"}}],
  "summary": "2–3 sentence portfolio rationale"
}}"""

        try:
            response = call_llm(
                prompt=prompt,
                system=DECISION_SYSTEM,
                provider="gemini",
                mode="deep",
                max_tokens=800,
            )
            return json.loads(_strip_fences(response))
        except Exception as e:
            print(f"  LLM decision error: {e}")
            return {"sells": [], "buys": [], "holds": [], "summary": f"LLM error: {e}"}

    # ── Trade execution helpers ───────────────────────────────────────────────

    def _buy_position(
        self,
        ticker: str,
        company_name: str,
        shares: float,
        price: float,
        run_id: int,
        reason: str,
    ):
        """Insert position + trade record, deduct cash from portfolio."""
        cost = shares * price
        conn = get_conn()
        c = conn.cursor()
        c.execute("""
            INSERT INTO auto_positions
            (portfolio_id, ticker, company_name, shares, entry_price, entry_date, reason, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'open')
        """, (self.portfolio_id, ticker, company_name, shares, price, date.today().isoformat(), reason))

        c.execute("""
            INSERT INTO auto_trades
            (portfolio_id, run_id, ticker, action, shares, price, trade_date, reason)
            VALUES (%s, %s, %s, 'buy', %s, %s, %s, %s)
        """, (self.portfolio_id, run_id, ticker, shares, price, date.today().isoformat(), reason))

        c.execute(
            "UPDATE auto_portfolio SET cash_remaining = cash_remaining - %s WHERE id = %s",
            (cost, self.portfolio_id),
        )
        conn.commit()
        conn.close()

    def _sell_position(
        self,
        pos: dict,
        price: float,
        run_id: int,
        reason: str,
        action: str = "sell",
    ):
        """Close a position, insert trade record, return cash to portfolio."""
        proceeds = pos["shares"] * price
        close_status = "stopped" if action == "stop_loss" else "closed"
        conn = get_conn()
        c = conn.cursor()
        c.execute("""
            UPDATE auto_positions
            SET status = %s, sell_price = %s, sell_date = %s
            WHERE id = %s
        """, (close_status, price, date.today().isoformat(), pos["id"]))

        c.execute("""
            INSERT INTO auto_trades
            (portfolio_id, run_id, ticker, action, shares, price, trade_date, reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (self.portfolio_id, run_id, pos["ticker"], action,
              pos["shares"], price, date.today().isoformat(), reason))

        c.execute(
            "UPDATE auto_portfolio SET cash_remaining = cash_remaining + %s WHERE id = %s",
            (proceeds, self.portfolio_id),
        )
        conn.commit()
        conn.close()

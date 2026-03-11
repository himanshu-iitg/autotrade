"""Paper trading module: strategy versioning, virtual portfolio, P&L tracking."""
import yfinance as yf
from datetime import date, datetime
from modules.db import get_conn


# ─── Strategy CRUD ────────────────────────────────────────────────────────────

def create_strategy(name: str, theme: str, virtual_capital: float, notes: str = "") -> int:
    """Create a new paper trading strategy. Returns strategy ID."""
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO strategies (name, theme, virtual_capital, notes, status)
        VALUES (?, ?, ?, ?, 'active')
    """, (name, theme, virtual_capital, notes))
    strategy_id = c.lastrowid
    conn.commit()
    conn.close()
    return strategy_id


def get_all_strategies() -> list[dict]:
    conn = get_conn()
    c = conn.cursor()
    rows = c.execute("SELECT * FROM strategies ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_strategy(strategy_id: int) -> dict | None:
    conn = get_conn()
    c = conn.cursor()
    row = c.execute("SELECT * FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def close_strategy(strategy_id: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "UPDATE strategies SET status = 'closed', closed_at = datetime('now') WHERE id = ?",
        (strategy_id,)
    )
    conn.commit()
    conn.close()


# ─── Trade CRUD ───────────────────────────────────────────────────────────────

def add_trade(strategy_id: int, ticker: str, company_name: str,
              shares: float, buy_price: float) -> int:
    """Record a paper buy trade. Returns trade ID."""
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO trades (strategy_id, ticker, company_name, shares, buy_price, buy_date)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (strategy_id, ticker, company_name, shares, buy_price, date.today().isoformat()))
    trade_id = c.lastrowid
    conn.commit()
    conn.close()
    return trade_id


def close_trade(trade_id: int, sell_price: float):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        UPDATE trades SET sell_price = ?, sell_date = ? WHERE id = ?
    """, (sell_price, date.today().isoformat(), trade_id))
    conn.commit()
    conn.close()


def get_open_trades(strategy_id: int) -> list[dict]:
    conn = get_conn()
    c = conn.cursor()
    rows = c.execute(
        "SELECT * FROM trades WHERE strategy_id = ? AND sell_date IS NULL",
        (strategy_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_trades(strategy_id: int) -> list[dict]:
    conn = get_conn()
    c = conn.cursor()
    rows = c.execute("SELECT * FROM trades WHERE strategy_id = ?", (strategy_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── P&L Calculation ─────────────────────────────────────────────────────────

def compute_portfolio_value(strategy_id: int) -> dict:
    """
    Fetch current prices for all open trades, compute P&L.
    Returns portfolio summary dict.
    """
    strategy = get_strategy(strategy_id)
    if not strategy:
        return {}

    trades = get_open_trades(strategy_id)
    total_invested = sum(t["shares"] * t["buy_price"] for t in trades)
    cash_remaining = strategy["virtual_capital"] - total_invested

    positions = []
    current_portfolio_value = 0.0

    for trade in trades:
        ticker = trade["ticker"]
        try:
            info = yf.Ticker(ticker).fast_info
            current_price = info.last_price or info.regular_market_price or 0
        except Exception:
            current_price = trade["buy_price"]  # fallback to buy price

        current_value = trade["shares"] * current_price
        cost_basis = trade["shares"] * trade["buy_price"]
        pnl = current_value - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0

        current_portfolio_value += current_value
        positions.append({
            "trade_id": trade["id"],
            "ticker": ticker,
            "company_name": trade["company_name"],
            "shares": trade["shares"],
            "buy_price": trade["buy_price"],
            "current_price": round(current_price, 2),
            "cost_basis": round(cost_basis, 2),
            "current_value": round(current_value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
        })

    total_value = current_portfolio_value + cash_remaining
    total_pnl = total_value - strategy["virtual_capital"]
    total_pnl_pct = (total_pnl / strategy["virtual_capital"] * 100) if strategy["virtual_capital"] else 0

    # Fetch Nifty 50 for benchmark comparison
    nifty_value = _get_nifty_value()

    return {
        "strategy": strategy,
        "positions": positions,
        "cash_remaining": round(cash_remaining, 2),
        "invested_value": round(total_invested, 2),
        "current_portfolio_value": round(current_portfolio_value, 2),
        "total_value": round(total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "nifty_current": nifty_value,
    }


def save_daily_snapshot(strategy_id: int, portfolio_value: float, cash_remaining: float):
    """Save today's portfolio snapshot. Upserts."""
    nifty_value = _get_nifty_value()
    today = date.today().isoformat()
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO daily_snapshots (strategy_id, snapshot_date, portfolio_value, cash_remaining, nifty_value)
        VALUES (?, ?, ?, ?, ?)
    """, (strategy_id, today, portfolio_value, cash_remaining, nifty_value))
    conn.commit()
    conn.close()


def get_daily_snapshots(strategy_id: int) -> list[dict]:
    conn = get_conn()
    c = conn.cursor()
    rows = c.execute(
        "SELECT * FROM daily_snapshots WHERE strategy_id = ? ORDER BY snapshot_date",
        (strategy_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_all_snapshots():
    """Update daily snapshots for all active strategies. Call once per session."""
    conn = get_conn()
    c = conn.cursor()
    active = c.execute("SELECT id FROM strategies WHERE status = 'active'").fetchall()
    conn.close()

    for row in active:
        sid = row["id"]
        summary = compute_portfolio_value(sid)
        if summary:
            save_daily_snapshot(sid, summary["total_value"], summary["cash_remaining"])
            print(f"  Strategy {sid}: Rs.{summary['total_value']:,.0f} | P&L: {summary['total_pnl_pct']:+.2f}%")


# ─── Comparison ───────────────────────────────────────────────────────────────

def compare_strategies() -> list[dict]:
    """Return summary row for each strategy for comparison table."""
    strategies = get_all_strategies()
    rows = []
    for s in strategies:
        snapshots = get_daily_snapshots(s["id"])
        if snapshots:
            latest = snapshots[-1]
            first = snapshots[0]
            port_return_pct = (latest["portfolio_value"] - s["virtual_capital"]) / s["virtual_capital"] * 100
            nifty_start = first.get("nifty_value") or 1
            nifty_end = latest.get("nifty_value") or nifty_start
            nifty_return_pct = (nifty_end - nifty_start) / nifty_start * 100
            alpha = port_return_pct - nifty_return_pct
        else:
            port_return_pct = nifty_return_pct = alpha = 0
            latest = {}

        rows.append({
            "id": s["id"],
            "name": s["name"],
            "theme": s["theme"],
            "status": s["status"],
            "virtual_capital": s["virtual_capital"],
            "current_value": latest.get("portfolio_value", s["virtual_capital"]),
            "return_pct": round(port_return_pct, 2),
            "nifty_return_pct": round(nifty_return_pct, 2),
            "alpha_pct": round(alpha, 2),
            "created_at": s["created_at"],
        })
    return rows


def _get_nifty_value() -> float | None:
    """Fetch current Nifty 50 index value."""
    try:
        info = yf.Ticker("^NSEI").fast_info
        return info.last_price
    except Exception:
        return None

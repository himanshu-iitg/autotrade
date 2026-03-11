"""PostgreSQL database initialization and helpers (Supabase)."""
import psycopg2
import psycopg2.extras
from config import DATABASE_URL


class _Cursor:
    """Thin cursor wrapper so execute() returns self, matching sqlite3 behaviour."""
    def __init__(self, cur):
        self._cur = cur

    def execute(self, sql, params=None):
        self._cur.execute(sql, params)
        return self

    def fetchall(self):
        return self._cur.fetchall()

    def fetchone(self):
        return self._cur.fetchone()

    @property
    def rowcount(self):
        return self._cur.rowcount


class _Conn:
    """Thin connection wrapper that returns RealDictCursor by default."""
    def __init__(self, conn):
        self._conn = conn

    def cursor(self):
        return _Cursor(self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor))

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._conn.close()


def get_conn() -> _Conn:
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to .streamlit/secrets.toml (local) "
            "or Streamlit Cloud app Settings → Secrets."
        )
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    return _Conn(conn)


def init_db():
    conn = get_conn()
    c = conn.cursor()

    # News headlines cache
    c.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id SERIAL PRIMARY KEY,
            source TEXT,
            title TEXT UNIQUE,
            summary TEXT,
            link TEXT,
            published_at TEXT,
            fetched_at TEXT DEFAULT TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS')
        )
    """)

    # LLM-extracted themes (one session = one batch)
    c.execute("""
        CREATE TABLE IF NOT EXISTS themes (
            id SERIAL PRIMARY KEY,
            session_date TEXT,
            theme TEXT,
            confidence REAL,
            evidence TEXT,
            sectors TEXT,
            created_at TEXT DEFAULT TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS')
        )
    """)

    # Screened stocks per theme
    c.execute("""
        CREATE TABLE IF NOT EXISTS screened_stocks (
            id SERIAL PRIMARY KEY,
            session_date TEXT,
            theme_id INTEGER,
            ticker TEXT,
            company_name TEXT,
            sector TEXT,
            market_cap_cr REAL,
            pe REAL,
            pb REAL,
            roe REAL,
            debt_equity REAL,
            revenue_growth REAL,
            eps_growth REAL,
            current_price REAL,
            screened_at TEXT DEFAULT TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS'),
            FOREIGN KEY (theme_id) REFERENCES themes(id)
        )
    """)

    # Paper trading: strategy versions
    c.execute("""
        CREATE TABLE IF NOT EXISTS strategies (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            theme TEXT,
            virtual_capital REAL NOT NULL,
            notes TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS'),
            closed_at TEXT
        )
    """)

    # Paper trading: individual trades per strategy
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            strategy_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            company_name TEXT,
            shares REAL NOT NULL,
            buy_price REAL NOT NULL,
            buy_date TEXT NOT NULL,
            sell_price REAL,
            sell_date TEXT,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        )
    """)

    # Paper trading: daily P&L snapshots
    c.execute("""
        CREATE TABLE IF NOT EXISTS daily_snapshots (
            id SERIAL PRIMARY KEY,
            strategy_id INTEGER NOT NULL,
            snapshot_date TEXT NOT NULL,
            portfolio_value REAL NOT NULL,
            cash_remaining REAL NOT NULL,
            nifty_value REAL,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id),
            UNIQUE(strategy_id, snapshot_date)
        )
    """)

    # Unique index ensures ON CONFLICT correctly deduplicates screened stocks
    c.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_screened_unique
        ON screened_stocks(session_date, theme_id, ticker)
    """)

    # ── Auto Trader tables ────────────────────────────────────────────────────

    # Single active auto portfolio (capital + cash tracking)
    c.execute("""
        CREATE TABLE IF NOT EXISTS auto_portfolio (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            capital REAL NOT NULL DEFAULT 500000,
            cash_remaining REAL NOT NULL,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT TO_CHAR(CURRENT_DATE, 'YYYY-MM-DD')
        )
    """)

    # Open/closed positions held by the auto portfolio
    c.execute("""
        CREATE TABLE IF NOT EXISTS auto_positions (
            id SERIAL PRIMARY KEY,
            portfolio_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            company_name TEXT,
            shares REAL NOT NULL,
            entry_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            sell_price REAL,
            sell_date TEXT,
            reason TEXT,
            status TEXT DEFAULT 'open',
            FOREIGN KEY(portfolio_id) REFERENCES auto_portfolio(id)
        )
    """)

    # Every buy/sell/stop_loss action ever taken
    c.execute("""
        CREATE TABLE IF NOT EXISTS auto_trades (
            id SERIAL PRIMARY KEY,
            portfolio_id INTEGER NOT NULL,
            run_id INTEGER,
            ticker TEXT,
            action TEXT,
            shares REAL,
            price REAL,
            trade_date TEXT,
            reason TEXT,
            FOREIGN KEY(portfolio_id) REFERENCES auto_portfolio(id)
        )
    """)

    # One row per pipeline run — used for equity curve
    c.execute("""
        CREATE TABLE IF NOT EXISTS auto_runs (
            id SERIAL PRIMARY KEY,
            portfolio_id INTEGER NOT NULL,
            run_date TEXT,
            themes_found INTEGER DEFAULT 0,
            stocks_screened INTEGER DEFAULT 0,
            buys_made INTEGER DEFAULT 0,
            sells_made INTEGER DEFAULT 0,
            stop_losses_triggered INTEGER DEFAULT 0,
            portfolio_value REAL,
            llm_summary TEXT,
            completed_at TEXT,
            FOREIGN KEY(portfolio_id) REFERENCES auto_portfolio(id)
        )
    """)

    conn.commit()
    conn.close()

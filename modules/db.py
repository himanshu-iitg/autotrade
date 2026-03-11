"""SQLite database initialization and helpers."""
import sqlite3
import os
from config import DB_PATH


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    # News headlines cache
    c.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            title TEXT UNIQUE,
            summary TEXT,
            link TEXT,
            published_at TEXT,
            fetched_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # LLM-extracted themes (one session = one batch)
    c.execute("""
        CREATE TABLE IF NOT EXISTS themes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date TEXT,
            theme TEXT,
            confidence REAL,
            evidence TEXT,
            sectors TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # Screened stocks per theme
    c.execute("""
        CREATE TABLE IF NOT EXISTS screened_stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            screened_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (theme_id) REFERENCES themes(id)
        )
    """)

    # Paper trading: strategy versions
    c.execute("""
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            theme TEXT,
            virtual_capital REAL NOT NULL,
            notes TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT (datetime('now')),
            closed_at TEXT
        )
    """)

    # Paper trading: individual trades per strategy
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL,
            snapshot_date TEXT NOT NULL,
            portfolio_value REAL NOT NULL,
            cash_remaining REAL NOT NULL,
            nifty_value REAL,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id),
            UNIQUE(strategy_id, snapshot_date)
        )
    """)

    # Unique index ensures INSERT OR IGNORE correctly deduplicates screened stocks
    c.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_screened_unique
        ON screened_stocks(session_date, theme_id, ticker)
    """)

    # ── Auto Trader tables ────────────────────────────────────────────────────

    # Single active auto portfolio (capital + cash tracking)
    c.execute("""
        CREATE TABLE IF NOT EXISTS auto_portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            capital REAL NOT NULL DEFAULT 500000,
            cash_remaining REAL NOT NULL,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT (date('now'))
        )
    """)

    # Open/closed positions held by the auto portfolio
    c.execute("""
        CREATE TABLE IF NOT EXISTS auto_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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

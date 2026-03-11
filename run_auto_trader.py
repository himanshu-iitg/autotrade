#!/usr/bin/env python3
"""
Standalone auto-trader script — runs the full pipeline without Streamlit.
Scheduled via Windows Task Scheduler at 11:30 AM IST on weekdays.

Usage:
    python run_auto_trader.py
"""
import sys
import os

# Fix Windows console encoding so ₹ and Unicode chars don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date
from modules.db import init_db
from modules.auto_trader import AutoTrader
from config import NSE_MARKET_HOLIDAYS


def is_market_holiday() -> bool:
    return date.today().isoformat() in NSE_MARKET_HOLIDAYS


def main():
    print(f"=== NSE Auto Trader | {date.today()} ===\n")

    if is_market_holiday():
        print(f"Today ({date.today()}) is an NSE market holiday. Skipping run.")
        sys.exit(0)

    init_db()

    trader = AutoTrader()
    log = trader.run_pipeline()

    print("\n=== Run Summary ===")
    print(f"  Buys        : {log['buys'] or 'none'}")
    print(f"  Sells       : {log['sells'] or 'none'}")
    print(f"  Stop-losses : {log['stop_losses'] or 'none'}")
    if log["summary"]:
        print(f"  Rationale   : {log['summary']}")

    if log.get("errors"):
        print(f"\nERRORS: {log['errors']}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

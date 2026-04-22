"""
Zerodha Kite Connect — real order execution module.

Two auth modes:
  A) AUTO-LOGIN (recommended for GitHub Actions / scheduled runs)
     Set KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET in .env
     The module will log in automatically each day using TOTP.

  B) MANUAL TOKEN (safe fallback)
     Run `python -m modules.zerodha_trader --login` in your terminal each morning
     after opening https://kite.trade/connect/login?api_key=YOUR_KEY&v=3
     and paste the request_token when prompted.

Products used: CNC (delivery/NRML) — long-term holding, not intraday.

Docs: https://kite.trade/docs/connect/v3/
"""
import os
import re
import json
import time
import requests
from datetime import date
from config import _secret

# ── Credentials (from .env / Streamlit secrets) ───────────────────────────────
KITE_API_KEY     = _secret("KITE_API_KEY")
KITE_API_SECRET  = _secret("KITE_API_SECRET")
KITE_USER_ID     = _secret("KITE_USER_ID")       # Zerodha login ID (e.g. AB1234)
KITE_PASSWORD    = _secret("KITE_PASSWORD")       # Zerodha web password
KITE_TOTP_SECRET = _secret("KITE_TOTP_SECRET")   # 32-char base32 from Zerodha TOTP setup

TOKEN_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "db", "kite_token.json"
)


# ── Token persistence ─────────────────────────────────────────────────────────

def _save_token(access_token: str):
    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    with open(TOKEN_FILE, "w") as f:
        json.dump({"access_token": access_token, "date": date.today().isoformat()}, f)
    print(f"[Zerodha] Access token saved for {date.today()}")


def _load_token() -> str | None:
    """Return today's cached access token, or None if missing/stale."""
    try:
        with open(TOKEN_FILE) as f:
            data = json.load(f)
        if data.get("date") == date.today().isoformat():
            return data.get("access_token")
    except Exception:
        pass
    return None


# ── Auto-login via TOTP ───────────────────────────────────────────────────────

def auto_login() -> str | None:
    """
    Fully automated Zerodha login using TOTP.
    Returns the access_token (also saves it to disk).
    Returns None if credentials are not configured.

    IMPORTANT: This method simulates web-browser login.
    Zerodha does not officially support this — use at your own discretion.
    If Zerodha changes their login page, this may need to be updated.
    """
    if not all([KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET, KITE_API_KEY, KITE_API_SECRET]):
        print("[Zerodha] Auto-login credentials not fully configured. Falling back to manual token.")
        return None

    try:
        import pyotp
    except ImportError:
        print("[Zerodha] pyotp not installed. Run: pip install pyotp")
        return None

    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})

        # Step 1: Submit user_id + password
        r1 = session.post(
            "https://kite.zerodha.com/api/login",
            data={"user_id": KITE_USER_ID, "password": KITE_PASSWORD},
            timeout=15,
        )
        r1_data = r1.json()
        if r1_data.get("status") != "success":
            raise RuntimeError(f"Login step 1 failed: {r1_data.get('message', r1.text[:200])}")

        request_id = r1_data["data"]["request_id"]

        # Step 2: Submit TOTP
        totp_value = pyotp.TOTP(KITE_TOTP_SECRET).now()
        r2 = session.post(
            "https://kite.zerodha.com/api/twofa",
            data={
                "user_id": KITE_USER_ID,
                "request_id": request_id,
                "twofa_value": totp_value,
                "twofa_type": "totp",
                "skip_session": "true",
            },
            timeout=15,
        )
        r2_data = r2.json()
        if r2_data.get("status") != "success":
            raise RuntimeError(f"TOTP step failed: {r2_data.get('message', r2.text[:200])}")

        # Step 3: Hit the Kite Connect login URL to get request_token from redirect
        login_url = f"https://kite.trade/connect/login?api_key={KITE_API_KEY}&v=3"
        r3 = session.get(login_url, allow_redirects=False, timeout=15)
        redirect = r3.headers.get("Location", "")
        match = re.search(r"request_token=([A-Za-z0-9]+)", redirect)
        if not match:
            raise RuntimeError(
                f"Could not extract request_token from redirect URL: {redirect[:300]}"
            )
        request_token = match.group(1)

        # Step 4: Generate session → access_token
        import hashlib
        checksum = hashlib.sha256(
            f"{KITE_API_KEY}{request_token}{KITE_API_SECRET}".encode()
        ).hexdigest()
        r4 = session.post(
            "https://api.kite.trade/session/token",
            data={
                "api_key": KITE_API_KEY,
                "request_token": request_token,
                "checksum": checksum,
            },
            headers={"X-Kite-Version": "3"},
            timeout=15,
        )
        r4_data = r4.json()
        if r4_data.get("status") != "success":
            raise RuntimeError(f"Session generation failed: {r4_data.get('message', r4.text[:200])}")

        access_token = r4_data["data"]["access_token"]
        _save_token(access_token)
        print(f"[Zerodha] Auto-login successful for user {KITE_USER_ID}")
        return access_token

    except Exception as e:
        print(f"[Zerodha] Auto-login error: {e}")
        return None


def manual_login():
    """
    Interactive helper for manual token entry.
    Open the Kite Connect login URL in your browser, log in,
    then copy the request_token from the redirect URL and paste it here.
    """
    if not KITE_API_KEY:
        print("ERROR: KITE_API_KEY not set in .env")
        return

    login_url = f"https://kite.trade/connect/login?api_key={KITE_API_KEY}&v=3"
    print(f"\n1. Open this URL in your browser:\n   {login_url}")
    print("2. Log in to Zerodha")
    print("3. After login, you'll be redirected to your redirect URL")
    print("4. Copy the 'request_token' value from that URL\n")

    request_token = input("Paste request_token here: ").strip()
    if not request_token:
        print("No token entered. Aborting.")
        return

    try:
        import hashlib
        session = requests.Session()
        checksum = hashlib.sha256(
            f"{KITE_API_KEY}{request_token}{KITE_API_SECRET}".encode()
        ).hexdigest()
        r = session.post(
            "https://api.kite.trade/session/token",
            data={
                "api_key": KITE_API_KEY,
                "request_token": request_token,
                "checksum": checksum,
            },
            headers={"X-Kite-Version": "3"},
            timeout=15,
        )
        data = r.json()
        if data.get("status") != "success":
            print(f"ERROR: {data.get('message', r.text[:300])}")
            return
        access_token = data["data"]["access_token"]
        _save_token(access_token)
        print(f"\nSuccess! Access token saved. Valid for today ({date.today()}).")
    except Exception as e:
        print(f"ERROR: {e}")


# ── Authenticated API calls (raw HTTP, no kiteconnect package needed) ─────────

BASE_URL = "https://api.kite.trade"


def _headers() -> dict:
    token = _load_token() or auto_login()
    if not token:
        raise RuntimeError(
            "Zerodha access token not available.\n"
            "Option A (automated): Set KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET in .env\n"
            "Option B (manual):    Run `python -m modules.zerodha_trader --login` each morning"
        )
    return {
        "X-Kite-Version": "3",
        "Authorization": f"token {KITE_API_KEY}:{token}",
        "Content-Type": "application/x-www-form-urlencoded",
    }


def _get(path: str, params: dict | None = None) -> dict:
    r = requests.get(f"{BASE_URL}{path}", headers=_headers(), params=params, timeout=15)
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Kite API error [{path}]: {data.get('message', r.text[:200])}")
    return data["data"]


def _post(path: str, payload: dict) -> dict:
    r = requests.post(f"{BASE_URL}{path}", headers=_headers(), data=payload, timeout=15)
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Kite API error [{path}]: {data.get('message', r.text[:200])}")
    return data["data"]


def _delete(path: str) -> dict:
    r = requests.delete(f"{BASE_URL}{path}", headers=_headers(), timeout=15)
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Kite API error [{path}]: {data.get('message', r.text[:200])}")
    return data["data"]


# ── Portfolio & market data ───────────────────────────────────────────────────

def get_funds() -> dict:
    """Return available equity cash balance."""
    data = _get("/user/margins/equity")
    return {
        "available_cash": data.get("available", {}).get("live_balance", 0),
        "used_margin":    data.get("utilised", {}).get("debits", 0),
        "net":            data.get("net", 0),
    }


def get_holdings() -> list[dict]:
    """Long-term delivery holdings (NRML/CNC already settled)."""
    return _get("/portfolio/holdings")


def get_positions() -> dict:
    """Intraday + positional open positions."""
    return _get("/portfolio/positions")


def get_ltp(tickers: list[str]) -> dict[str, float]:
    """
    Get last traded price for NSE tickers.
    tickers: list of yfinance-style tickers e.g. ['RELIANCE.NS', 'TCS.NS']
    Returns dict: {'RELIANCE.NS': 2500.0, ...}
    """
    if not tickers:
        return {}
    nse_symbols = [f"NSE:{t.replace('.NS','').replace('.BO','')}" for t in tickers]
    params = {"i": nse_symbols}
    # LTP endpoint uses multi-value 'i' param
    r = requests.get(
        f"{BASE_URL}/quote/ltp",
        headers=_headers(),
        params=[("i", s) for s in nse_symbols],
        timeout=15,
    )
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(f"LTP error: {data.get('message', r.text[:200])}")
    result = {}
    for ticker, nse_sym in zip(tickers, nse_symbols):
        result[ticker] = data["data"].get(nse_sym, {}).get("last_price", 0.0)
    return result


# ── Order placement ───────────────────────────────────────────────────────────

def place_buy_order(
    ticker: str,
    shares: int,
    order_type: str = "MARKET",
    limit_price: float = 0.0,
    tag: str = "AutoTrader",
) -> str:
    """
    Place a CNC (delivery) BUY order on NSE.
    ticker: yfinance-style e.g. 'RELIANCE.NS'
    Returns Kite order_id.
    """
    if shares < 1:
        raise ValueError(f"shares must be >= 1, got {shares}")

    nse_symbol = ticker.replace(".NS", "").replace(".BO", "").upper()
    payload = {
        "tradingsymbol": nse_symbol,
        "exchange": "NSE",
        "transaction_type": "BUY",
        "quantity": str(shares),
        "product": "CNC",
        "order_type": order_type,   # MARKET or LIMIT
        "validity": "DAY",
        "tag": tag[:20],
    }
    if order_type == "LIMIT" and limit_price > 0:
        payload["price"] = str(round(limit_price, 2))

    data = _post("/orders/regular", payload)
    order_id = data.get("order_id", "")
    print(f"[Zerodha] BUY {nse_symbol} x{shares} @ {order_type} — order_id: {order_id}")
    return order_id


def place_sell_order(
    ticker: str,
    shares: int,
    order_type: str = "MARKET",
    limit_price: float = 0.0,
    tag: str = "AutoTrader",
) -> str:
    """
    Place a CNC (delivery) SELL order on NSE.
    Returns Kite order_id.
    """
    if shares < 1:
        raise ValueError(f"shares must be >= 1, got {shares}")

    nse_symbol = ticker.replace(".NS", "").replace(".BO", "").upper()
    payload = {
        "tradingsymbol": nse_symbol,
        "exchange": "NSE",
        "transaction_type": "SELL",
        "quantity": str(shares),
        "product": "CNC",
        "order_type": order_type,
        "validity": "DAY",
        "tag": tag[:20],
    }
    if order_type == "LIMIT" and limit_price > 0:
        payload["price"] = str(round(limit_price, 2))

    data = _post("/orders/regular", payload)
    order_id = data.get("order_id", "")
    print(f"[Zerodha] SELL {nse_symbol} x{shares} @ {order_type} — order_id: {order_id}")
    return order_id


def cancel_order(order_id: str) -> bool:
    """Cancel a pending order. Returns True on success."""
    try:
        _delete(f"/orders/regular/{order_id}")
        print(f"[Zerodha] Cancelled order {order_id}")
        return True
    except Exception as e:
        print(f"[Zerodha] Cancel failed for {order_id}: {e}")
        return False


def get_order_status(order_id: str) -> dict:
    """Get status of a specific order."""
    orders = _get("/orders")
    for o in orders:
        if str(o.get("order_id")) == str(order_id):
            return o
    return {}


# ── Utility: sync Zerodha holdings into auto_positions ───────────────────────

def sync_holdings_to_db(portfolio_id: int):
    """
    Pull real holdings from Zerodha and upsert them into auto_positions.
    Useful on first run or after manual trades outside the bot.
    """
    from modules.db import get_conn

    holdings = get_holdings()
    if not holdings:
        print("[Zerodha] No holdings to sync.")
        return

    conn = get_conn()
    c = conn.cursor()

    for h in holdings:
        ticker = h["tradingsymbol"] + ".NS"
        shares = h["quantity"]
        avg_price = h["average_price"]
        company_name = h.get("exchange_token", ticker)

        if shares <= 0:
            continue

        # Check if already tracked
        existing = c.execute(
            "SELECT id FROM auto_positions WHERE portfolio_id = %s AND ticker = %s AND status = 'open'",
            (portfolio_id, ticker),
        ).fetchone()

        if not existing:
            c.execute("""
                INSERT INTO auto_positions
                (portfolio_id, ticker, company_name, shares, entry_price, entry_date, reason, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'Synced from Zerodha holdings', 'open')
            """, (portfolio_id, ticker, company_name, shares, avg_price, date.today().isoformat()))
            print(f"[Zerodha Sync] Imported {ticker}: {shares} shares @ ₹{avg_price:.2f}")

    conn.commit()
    conn.close()
    print(f"[Zerodha Sync] Done. {len(holdings)} holdings checked.")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--login" in sys.argv:
        manual_login()

    elif "--auto-login" in sys.argv:
        token = auto_login()
        if token:
            print(f"Token (first 20 chars): {token[:20]}...")
        else:
            print("Auto-login failed. Check your credentials in .env")

    elif "--funds" in sys.argv:
        try:
            funds = get_funds()
            print(f"Available cash : ₹{funds['available_cash']:,.2f}")
            print(f"Used margin    : ₹{funds['used_margin']:,.2f}")
            print(f"Net            : ₹{funds['net']:,.2f}")
        except Exception as e:
            print(f"Error: {e}")

    elif "--holdings" in sys.argv:
        try:
            holdings = get_holdings()
            print(f"\n{len(holdings)} holdings:\n")
            for h in holdings:
                pnl = (h.get("last_price", 0) - h["average_price"]) * h["quantity"]
                print(f"  {h['tradingsymbol']:15s} {h['quantity']:5} shares  "
                      f"avg ₹{h['average_price']:>8.2f}  "
                      f"ltp ₹{h.get('last_price',0):>8.2f}  "
                      f"P&L ₹{pnl:>+10.2f}")
        except Exception as e:
            print(f"Error: {e}")

    else:
        print("Usage:")
        print("  python -m modules.zerodha_trader --login        # Manual token entry")
        print("  python -m modules.zerodha_trader --auto-login   # TOTP auto-login")
        print("  python -m modules.zerodha_trader --funds        # Check balance")
        print("  python -m modules.zerodha_trader --holdings     # View holdings")

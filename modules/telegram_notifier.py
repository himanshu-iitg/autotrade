"""
Telegram notification module — sends trade alerts and daily summaries.

Setup (one-time):
1. Open Telegram, search for @BotFather
2. Send /newbot, follow the prompts → you get a BOT_TOKEN
3. Start a chat with your new bot, then get your CHAT_ID:
   Open: https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
   Find "chat":{"id":XXXXXXXX} in the response
4. Add to your .env:
   TELEGRAM_BOT_TOKEN=123456789:ABCdef...
   TELEGRAM_CHAT_ID=987654321

If these are not set, all notify_* calls are silent no-ops.
"""
import requests
from config import _secret

TELEGRAM_BOT_TOKEN = _secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _secret("TELEGRAM_CHAT_ID")

_BASE = "https://api.telegram.org/bot{token}/sendMessage"


def _enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _send(text: str, parse_mode: str = "HTML") -> bool:
    """Send a message. Returns True on success, False on failure."""
    if not _enabled():
        return False
    try:
        r = requests.post(
            _BASE.format(token=TELEGRAM_BOT_TOKEN),
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": "true",
            },
            timeout=10,
        )
        ok = r.json().get("ok", False)
        if not ok:
            print(f"[Telegram] Send failed: {r.text[:200]}")
        return ok
    except Exception as e:
        print(f"[Telegram] Error: {e}")
        return False


# ── Public notification helpers ───────────────────────────────────────────────

def notify_buy(ticker: str, company_name: str, shares: int | float,
               price: float, reason: str, is_live: bool = False) -> bool:
    mode = "🟢 LIVE BUY" if is_live else "📄 PAPER BUY"
    cost = shares * price
    text = (
        f"<b>{mode}</b>\n"
        f"📈 <b>{ticker}</b> — {company_name}\n"
        f"Shares : {shares:.0f}\n"
        f"Price  : ₹{price:,.2f}\n"
        f"Cost   : ₹{cost:,.0f}\n"
        f"Reason : {reason}"
    )
    return _send(text)


def notify_sell(ticker: str, company_name: str, shares: int | float,
                price: float, pnl: float, pnl_pct: float,
                reason: str, is_live: bool = False) -> bool:
    mode = "🔴 LIVE SELL" if is_live else "📄 PAPER SELL"
    emoji = "📈" if pnl >= 0 else "📉"
    text = (
        f"<b>{mode}</b>\n"
        f"{emoji} <b>{ticker}</b> — {company_name}\n"
        f"Shares : {shares:.0f}\n"
        f"Price  : ₹{price:,.2f}\n"
        f"P&amp;L  : ₹{pnl:+,.0f} ({pnl_pct:+.2f}%)\n"
        f"Reason : {reason}"
    )
    return _send(text)


def notify_stop_loss(ticker: str, company_name: str, shares: int | float,
                     price: float, pnl_pct: float, is_live: bool = False) -> bool:
    mode = "🚨 LIVE STOP-LOSS" if is_live else "🚨 PAPER STOP-LOSS"
    text = (
        f"<b>{mode} TRIGGERED</b>\n"
        f"💀 <b>{ticker}</b> — {company_name}\n"
        f"Shares  : {shares:.0f}\n"
        f"Exit ₹  : ₹{price:,.2f}\n"
        f"Loss    : {pnl_pct:.2f}%"
    )
    return _send(text)


def notify_daily_summary(
    portfolio_value: float,
    cash_remaining: float,
    total_pnl_pct: float,
    nifty_pct: float | None,
    buys: list[str],
    sells: list[str],
    stop_losses: list[str],
    llm_summary: str,
    is_live: bool = False,
) -> bool:
    mode = "🤖 LIVE AUTO-TRADER" if is_live else "📄 PAPER AUTO-TRADER"
    alpha = ""
    if nifty_pct is not None:
        alpha_val = total_pnl_pct - nifty_pct
        alpha = f"\nAlpha vs Nifty : {alpha_val:+.2f}%"

    buys_str   = ", ".join(buys)   or "none"
    sells_str  = ", ".join(sells)  or "none"
    stops_str  = ", ".join(stop_losses) or "none"

    text = (
        f"<b>{mode} — Daily Run</b>\n"
        f"\n"
        f"💼 Portfolio : ₹{portfolio_value:,.0f}\n"
        f"💵 Cash      : ₹{cash_remaining:,.0f}\n"
        f"📊 P&amp;L    : {total_pnl_pct:+.2f}%{alpha}\n"
        f"\n"
        f"✅ Bought    : {buys_str}\n"
        f"❌ Sold      : {sells_str}\n"
        f"🛑 Stop-loss : {stops_str}\n"
        f"\n"
        f"🧠 <i>{llm_summary}</i>"
    )
    return _send(text)


def notify_error(context: str, error: str) -> bool:
    text = (
        f"⚠️ <b>Auto-Trader Error</b>\n"
        f"Context : {context}\n"
        f"Error   : {error[:300]}"
    )
    return _send(text)


def send_test_message() -> bool:
    """Quick test — call from CLI to verify setup."""
    return _send("✅ <b>Telegram test</b> — your NSE Auto Trader bot is connected!")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _enabled():
        print("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in .env")
    else:
        ok = send_test_message()
        print("Message sent!" if ok else "Failed to send message. Check your token and chat ID.")

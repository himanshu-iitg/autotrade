# Upgrade Plan — NSE Auto Trader
_Saved for later — paused to work on existing screener first_

## Files Already Written
- `modules/zerodha_trader.py` ✅ — Zerodha Kite Connect real order execution
- `modules/technical.py` ✅ — RSI, momentum, 52-week scoring layer

## Files Still To Write
- `modules/telegram_notifier.py` — Trade alerts via Telegram bot
- Update `config.py` — Add KITE_*, TELEGRAM_* secrets
- Update `requirements.txt` — Add kiteconnect, pyotp, requests
- Update `.env.example` — Add all new keys with instructions
- Update `modules/auto_trader.py` — Wire in LIVE_TRADING flag, Zerodha execution, Telegram alerts
- `SETUP_GUIDE.md` — Step-by-step instructions for the full upgrade

## Zerodha Setup Steps (for later)
1. Go to https://developers.kite.trade/ → Create app → get API Key + Secret
2. Set redirect URL to `https://127.0.0.1` (for local dev)
3. Add KITE_API_KEY, KITE_API_SECRET to .env
4. For auto-login: add KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET
   - TOTP secret = the 32-char base32 code shown when setting up Zerodha TOTP authenticator
5. Test: `python -m modules.zerodha_trader --funds`
6. Set LIVE_TRADING=true in .env only when confident in paper trading results

## Telegram Setup Steps (for later)
1. Open Telegram → @BotFather → /newbot → get BOT_TOKEN
2. Start chat with bot, open https://api.telegram.org/bot<TOKEN>/getUpdates → get CHAT_ID
3. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env
4. Test: `python -m modules.telegram_notifier`

## Key Design Decisions
- LIVE_TRADING flag in .env: false = paper mode (default), true = real Zerodha orders
- All new modules are optional/non-breaking — if keys not set, they silently skip
- Technical scoring is an additive layer, not a replacement for fundamentals
- CNC product only (delivery) — no intraday, no MIS, no F&O

"""
NSE Stock Advisor — Streamlit app.

Flow: Screen → Triage → Suggestions → Act
"""
import sys, json, hashlib, math
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from modules.db import init_db
from modules.news_fetcher import fetch_all_feeds
from modules.screener import screen_nifty500, get_recently_seen_tickers
from modules.news_triage import triage_stock, _get_sector_config
from modules.stock_detail import (
    get_price_history, get_stock_info,
    generate_stock_thesis, get_key_metrics, summarize_top_picks,
)
from modules.paper_trader import (
    create_strategy, get_all_strategies, close_strategy,
    add_trade, close_trade, get_open_trades, get_all_trades,
    compute_portfolio_value, save_daily_snapshot, update_all_snapshots,
    get_daily_snapshots, compare_strategies,
)
from modules.auto_trader import AutoTrader
from modules.llm_client import provider_label
from config import SCREENER_DEFAULTS, ANTHROPIC_API_KEY, GEMINI_API_KEY

# ── Screening Presets ─────────────────────────────────────────────────────────

SCREENING_PRESETS = {
    "Value Hunt": {
        "icon": "💰",
        "tagline": "Profitable quality, trading cheap",
        "description": "High ROE, low P/E, low debt. Best when markets are fairly valued or overheated — forces you to find genuinely cheap quality.",
        "filters": {"min_market_cap_cr": 2000, "max_pe": 20, "min_pe": 5, "max_debt_equity": 1.0, "min_roe_pct": 12},
        "sectors": None,
    },
    "Growth Compounder": {
        "icon": "🚀",
        "tagline": "Pay up for high-quality growth",
        "description": "Strong revenue growth + high ROE. Willing to pay a higher P/E for businesses that keep compounding. Best in bull markets.",
        "filters": {"min_market_cap_cr": 1000, "max_pe": 55, "max_debt_equity": 1.5, "min_roe_pct": 18, "min_revenue_growth_pct": 10},
        "sectors": None,
    },
    "Quality Defensive": {
        "icon": "🛡️",
        "tagline": "Large-cap safety in uncertain markets",
        "description": "Only large caps, very low debt, strong ROE. For when you want to stay invested but reduce risk — consolidation or correction phases.",
        "filters": {"min_market_cap_cr": 10000, "max_pe": 40, "max_debt_equity": 0.5, "min_roe_pct": 15},
        "sectors": None,
    },
    "Mid Cap Opportunity": {
        "icon": "📈",
        "tagline": "Higher risk, higher potential",
        "description": "Smaller but profitable companies with room to run. Best in risk-on markets with strong domestic liquidity.",
        "filters": {"min_market_cap_cr": 500, "max_pe": 45, "max_debt_equity": 1.5, "min_roe_pct": 10},
        "sectors": None,
    },
    "Capex Cycle": {
        "icon": "⚙️",
        "tagline": "Riding India's investment boom",
        "description": "Capital goods, infra, power — sectors that win when government or private capex is rising. PLI, railways, defence.",
        "filters": {"min_market_cap_cr": 1000, "max_pe": 60, "max_debt_equity": 2.0, "min_revenue_growth_pct": 8},
        "sectors": ["CAPITAL GOODS", "CONSTRUCTION", "POWER", "CONSTRUCTION MATERIALS"],
    },
    "Export Winners": {
        "icon": "🌍",
        "tagline": "USD earners — play the weak rupee",
        "description": "IT, pharma, chemicals — revenue in dollars, costs in rupees. Best when INR is under pressure or global demand is recovering.",
        "filters": {"min_market_cap_cr": 1000, "max_pe": 35, "max_debt_equity": 0.8, "min_roe_pct": 14},
        "sectors": ["INFORMATION TECHNOLOGY", "HEALTHCARE", "CHEMICALS"],
    },
    "Financials Focus": {
        "icon": "🏦",
        "tagline": "Banks & NBFCs — play the rate cycle",
        "description": "Rate-sensitive plays. Best when RBI is cutting rates or credit growth is accelerating. Watch repo rate and NPA trends.",
        "filters": {"min_market_cap_cr": 2000, "max_pe": 25, "max_debt_equity": 10.0, "min_roe_pct": 12},
        "sectors": ["FINANCIAL SERVICES"],
    },
    "Consumption India": {
        "icon": "🛒",
        "tagline": "Domestic demand — urban + rural",
        "description": "FMCG, auto, consumer durables. Best after a good monsoon, rural income boost, or urban wage growth cycle.",
        "filters": {"min_market_cap_cr": 2000, "max_pe": 50, "max_debt_equity": 1.0, "min_roe_pct": 12},
        "sectors": ["FAST MOVING CONSUMER GOODS", "AUTOMOBILE AND AUTO COMPONENTS", "CONSUMER DURABLES", "CONSUMER SERVICES"],
    },
}

PRESET_NAMES = list(SCREENING_PRESETS.keys())

_DAY_RATIONALE = {
    0: "Start of week — good time for a growth-oriented screen before markets pick up momentum.",
    1: "Tuesday — mid-week, useful to hunt for value before mid-week volatility.",
    2: "Wednesday — mid-week, consider defensive or export plays.",
    3: "Thursday — pre-weekend, good to look at sector-specific opportunities.",
    4: "Friday — end of week review, quality defensive or mid-cap discovery works well.",
    5: "Weekend — great time to explore mid-cap opportunities without time pressure.",
    6: "Sunday — plan your week ahead with a growth or value screen.",
}


def get_today_preset() -> str:
    """Rotate preset daily — deterministic by date, different every day."""
    from datetime import date
    h = int(hashlib.md5(date.today().isoformat().encode()).hexdigest(), 16)
    return PRESET_NAMES[h % len(PRESET_NAMES)]


def sentiment_badge(sentiment: str) -> str:
    return {"POSITIVE": "🟢", "NEUTRAL": "🟡", "NEGATIVE": "🔴"}.get(sentiment, "⚪")


# ── App setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NSE Stock Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_db()

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("📊 NSE Stock Advisor")
page = st.sidebar.radio(
    "Navigate",
    ["🔍 Screen", "📰 Triage", "💡 Suggestions", "📊 Portfolio", "⚙️ Auto Trader", "📡 News Feed"],
    index=0,
)
st.sidebar.markdown("---")

st.sidebar.markdown("**LLM Provider**")
provider_options = []
if GEMINI_API_KEY:    provider_options.append("gemini")
if ANTHROPIC_API_KEY: provider_options.append("claude")
if not provider_options: provider_options = ["gemini"]

selected_provider = st.sidebar.radio(
    "Use for triage & thesis:",
    options=provider_options,
    format_func=lambda p: {"claude": "Claude Sonnet — paid, best quality",
                            "gemini": "Gemini 2.5 Flash Lite — free tier"}.get(p, p),
    key="llm_provider",
)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh News DB", type="primary"):
    with st.spinner("Fetching…"):
        n = fetch_all_feeds()
        st.sidebar.success(f"{n} new articles")
if st.sidebar.button("📸 Update Snapshots"):
    with st.spinner("Updating…"):
        update_all_snapshots()
        st.sidebar.success("Done")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Screen":
    from datetime import date

    st.title("🔍 Screen")
    st.caption("Choose a screening strategy → run → proceed to triage")

    today_preset_name = get_today_preset()
    today_preset      = SCREENING_PRESETS[today_preset_name]
    dow               = date.today().weekday()

    # ── Today's suggestion card ───────────────────────────────────────────────
    st.markdown("### Today's Suggested Screen")
    col_icon, col_info, col_btn = st.columns([1, 6, 2])
    with col_icon:
        st.markdown(f"<div style='font-size:3rem;text-align:center;padding-top:8px'>"
                    f"{today_preset['icon']}</div>", unsafe_allow_html=True)
    with col_info:
        st.markdown(f"**{today_preset_name}** — {today_preset['tagline']}")
        st.caption(today_preset["description"])
        st.caption(f"💡 {_DAY_RATIONALE[dow]}")
    with col_btn:
        st.write("")
        st.write("")
        if st.button(f"Use this →", type="primary", key="use_today"):
            st.session_state["selected_preset"] = today_preset_name
            st.rerun()

    st.markdown("---")

    # ── Preset grid ───────────────────────────────────────────────────────────
    st.markdown("### Or choose a different strategy")
    selected_preset = st.session_state.get("selected_preset", today_preset_name)

    grid_cols = st.columns(4)
    for i, (name, preset) in enumerate(SCREENING_PRESETS.items()):
        with grid_cols[i % 4]:
            active   = (name == selected_preset)
            bg       = "#1a3a5c" if active else "#0f1f2e"
            border   = "#00d4ff" if active else "#2a4a6a"
            st.markdown(
                f"""<div style='border:1px solid {border};border-radius:8px;
                    padding:10px 12px;background:{bg};margin-bottom:6px;min-height:88px'>
                    <span style='font-size:1.3rem'>{preset['icon']}</span>
                    <span style='font-weight:700;color:#e0e0e0;margin-left:6px'>{name}</span><br>
                    <span style='font-size:0.75rem;color:#aaa'>{preset['tagline']}</span>
                </div>""",
                unsafe_allow_html=True,
            )
            lbl = "✓ Active" if active else "Select"
            if st.button(lbl, key=f"preset_{name}",
                         type="primary" if active else "secondary",
                         use_container_width=True):
                st.session_state["selected_preset"] = name
                st.rerun()

    st.markdown("---")

    # ── Active preset + filter customisation ──────────────────────────────────
    preset  = SCREENING_PRESETS[selected_preset]
    filters = dict(preset["filters"])
    sectors = list(preset.get("sectors") or [])

    st.markdown(f"### {preset['icon']} {selected_preset}")
    st.markdown(f"*{preset['description']}*")

    with st.expander("⚙️ Customise Filters", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            filters["min_market_cap_cr"] = st.number_input(
                "Min MCap (₹Cr)", value=float(filters.get("min_market_cap_cr", 500)), step=500.0)
        with c2:
            filters["max_pe"] = st.number_input(
                "Max P/E", value=float(filters.get("max_pe", 60)), step=5.0)
        with c3:
            filters["max_debt_equity"] = st.number_input(
                "Max D/E", value=float(filters.get("max_debt_equity", 2.0)), step=0.25)
        with c4:
            filters["min_roe_pct"] = st.number_input(
                "Min ROE %", value=float(filters.get("min_roe_pct", 0)), step=2.0)
        c5, c6 = st.columns(2)
        with c5:
            filters["min_revenue_growth_pct"] = st.number_input(
                "Min Rev Growth %",
                value=float(filters.get("min_revenue_growth_pct", -100)), step=5.0)
        with c6:
            max_results_ui = st.slider("Max results", 10, 40, 20)

        from config import NSE_SECTORS
        sector_override = st.multiselect(
            "Sector filter (blank = follow preset default)",
            NSE_SECTORS, default=sectors)
        if sector_override:
            sectors = sector_override

    col_d, col_e, col_f = st.columns(3)
    with col_d:
        discovery_mode = st.toggle("🎲 Discovery Mode",
            help="Shuffle Nifty 500 order — surfaces different stocks each run")
    with col_e:
        exclude_recent = st.toggle("🚫 Exclude Recently Seen",
            help="Skip tickers that appeared in any screen in the last 3 days")
    with col_f:
        force_refresh  = st.toggle("♻️ Force Re-screen",
            help="Clear today's cache and re-fetch from yfinance")

    st.markdown("")
    if st.button("▶ Run Screen", type="primary"):
        exclude_set = get_recently_seen_tickers(days_back=3) if exclude_recent else None
        if exclude_set:
            st.info(f"Excluding {len(exclude_set)} recently seen tickers")

        with st.spinner(f"Screening Nifty 500 — **{selected_preset}** filters…"):
            stocks = screen_nifty500(
                sectors     = sectors if sectors else None,
                filters     = filters,
                max_results = max_results_ui,
                force       = force_refresh or discovery_mode,
                shuffle     = discovery_mode,
                exclude_tickers = exclude_set,
            )

        if not stocks:
            st.warning("No stocks passed. Try relaxing filters or a different preset.")
        else:
            st.session_state["screened_stocks"] = stocks
            st.session_state["active_preset"]   = selected_preset
            st.session_state["triaged_stocks"]  = []
            st.success(f"✅ {len(stocks)} stocks passed **{selected_preset}** filters")

    # ── Results table ─────────────────────────────────────────────────────────
    stocks = st.session_state.get("screened_stocks", [])
    if stocks:
        active = st.session_state.get("active_preset", "")
        st.markdown(f"#### Results — {active} ({len(stocks)} stocks)")
        df = pd.DataFrame(stocks)
        display_cols = {
            "ticker": "Ticker", "company_name": "Company", "sector": "Sector",
            "market_cap_cr": "MCap ₹Cr", "current_price": "Price ₹",
            "pe": "P/E", "roe": "ROE %", "debt_equity": "D/E",
            "revenue_growth": "Rev Gth %", "eps_growth": "EPS Gth %",
        }
        st.dataframe(
            df[[c for c in display_cols if c in df.columns]].rename(columns=display_cols),
            use_container_width=True, hide_index=True,
        )
        st.markdown("---")
        col_next, col_info = st.columns([2, 5])
        with col_next:
            if st.button("📰 Proceed to Triage →", type="primary", use_container_width=True):
                st.session_state["_nav"] = "📰 Triage"
                st.rerun()
        with col_info:
            st.caption("Triage analyses sector-specific signals per stock "
                       "(repo rate for banks, FDA for pharma, etc.) before AI suggestions.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: TRIAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📰 Triage":
    st.title("📰 Triage")
    st.caption("Sector-aware news analysis — the right signals for each stock, not generic headlines")

    provider = st.session_state.get("llm_provider", "gemini")
    stocks   = st.session_state.get("screened_stocks", [])

    if not stocks:
        st.info("No screened stocks yet. Go to **🔍 Screen** first.")
        st.stop()

    preset = st.session_state.get("active_preset", "")
    st.markdown(f"**{len(stocks)} stocks** from *{preset}* ready for triage")

    # Show what signals will be checked
    sectors_present = sorted({s.get("sector", "") for s in stocks if s.get("sector")})
    if sectors_present:
        with st.expander("💡 Signals being monitored per sector", expanded=False):
            for sec in sectors_present:
                cfg     = _get_sector_config(sec)
                signals = " · ".join(cfg["signals"][:4])
                st.markdown(f"**{sec}**: {signals}")

    col_n, col_rn = st.columns([3, 2])
    with col_n:
        top_n = st.slider("Triage top N stocks", 5, len(stocks), min(12, len(stocks)))
    with col_rn:
        refresh_news = st.checkbox("Refresh news DB first", value=False)

    if st.button("▶ Run Triage", type="primary"):
        if refresh_news:
            with st.spinner("Refreshing news DB…"):
                fetch_all_feeds()

        progress_bar = st.progress(0, text="Starting…")
        status_box   = st.empty()
        triaged      = []

        to_triage = [dict(s) for s in stocks[:top_n]]
        rest      = [dict(s) for s in stocks[top_n:]]

        for i, stock in enumerate(to_triage):
            ticker = stock.get("ticker", "?")
            sector = stock.get("sector", "")
            pct    = int(i / len(to_triage) * 100)
            progress_bar.progress(pct, text=f"Triaging {ticker} ({sector})…")
            status_box.caption(
                f"Checking sector signals for **{stock.get('company_name', ticker)}**…")
            enriched = triage_stock(stock, provider=provider, pause=0.4)
            triaged.append(enriched)

        progress_bar.progress(100, text="Triage complete ✅")
        status_box.empty()

        for s in rest:
            s.update({
                "triage_sentiment": "NEUTRAL", "triage_confidence": 0.0,
                "triage_catalyst": "not triaged", "triage_risk": "not triaged",
                "triage_signal_hits": [], "triage_summary": "Not triaged.",
                "triage_news_count": 0,
            })

        sentiment_order = {"POSITIVE": 0, "NEUTRAL": 1, "NEGATIVE": 2}
        triaged.sort(key=lambda s: (
            sentiment_order.get(s.get("triage_sentiment", "NEUTRAL"), 1),
            -(s.get("triage_confidence", 0)),
            -(s.get("market_cap_cr", 0)),
        ))

        all_stocks = triaged + rest
        st.session_state["triaged_stocks"] = all_stocks

        pos  = sum(1 for s in triaged if s.get("triage_sentiment") == "POSITIVE")
        neg  = sum(1 for s in triaged if s.get("triage_sentiment") == "NEGATIVE")
        neut = len(triaged) - pos - neg
        st.success(f"Done — 🟢 {pos} Positive · 🟡 {neut} Neutral · 🔴 {neg} Negative")

    # ── Triage results ────────────────────────────────────────────────────────
    triaged_stocks = st.session_state.get("triaged_stocks", [])
    if triaged_stocks:
        st.markdown("---")

        cp, cn, cne = st.columns(3)
        pos  = sum(1 for s in triaged_stocks if s.get("triage_sentiment") == "POSITIVE")
        neg  = sum(1 for s in triaged_stocks if s.get("triage_sentiment") == "NEGATIVE")
        neut = len(triaged_stocks) - pos - neg
        cp.metric("🟢 Positive", pos)
        cn.metric("🟡 Neutral", neut)
        cne.metric("🔴 Negative", neg)

        st.markdown("")
        rows = []
        for s in triaged_stocks:
            sent = s.get("triage_sentiment", "NEUTRAL")
            rows.append({
                "":           sentiment_badge(sent),
                "Ticker":     s.get("ticker", ""),
                "Company":    (s.get("company_name") or "")[:28],
                "Sector":     (s.get("sector") or "")[:18],
                "Sentiment":  sent,
                "Confidence": f"{s.get('triage_confidence', 0):.0%}",
                "MCap ₹Cr":   s.get("market_cap_cr", 0),
                "P/E":        s.get("pe"),
                "ROE %":      s.get("roe"),
                "Catalyst":   (s.get("triage_catalyst") or "")[:55],
                "Risk":       (s.get("triage_risk") or "")[:55],
                "Summary":    (s.get("triage_summary") or "")[:80],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        col_next, col_info = st.columns([2, 5])
        with col_next:
            if st.button("💡 Proceed to Suggestions →", type="primary", use_container_width=True):
                st.session_state["_nav"] = "💡 Suggestions"
                st.rerun()
        with col_info:
            st.caption("Suggestions will rank POSITIVE stocks and generate AI investment theses.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: SUGGESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Suggestions":
    st.title("💡 Suggestions")
    st.caption("AI-ranked picks from triage results — deep dive and add to portfolio")

    provider       = st.session_state.get("llm_provider", "gemini")
    triaged_stocks = st.session_state.get("triaged_stocks", [])
    preset         = st.session_state.get("active_preset", "")

    if not triaged_stocks:
        st.info("Run **📰 Triage** first to get suggestions.")
        st.stop()

    # ── AI Top Picks ──────────────────────────────────────────────────────────
    st.markdown("### AI Top Picks")
    positive  = [s for s in triaged_stocks if s.get("triage_sentiment") == "POSITIVE"]
    neutral   = [s for s in triaged_stocks if s.get("triage_sentiment") == "NEUTRAL"]
    pool      = (positive + neutral)[:10]

    picks_key = f"top_picks_{preset}"
    cb, cl = st.columns([2, 4])
    with cb:
        gen_picks = st.button("🤖 Generate Top Picks", type="primary",
                              help=f"AI reviews {len(pool)} triaged candidates")
    with cl:
        if picks_key in st.session_state:
            st.caption(f"Picks for **{preset}**")

    if gen_picks:
        model_lbl = "Gemini 2.5 Flash Lite" if provider == "gemini" else "Claude Sonnet"
        with st.spinner(f"Analysing with {model_lbl}…"):
            try:
                picks = summarize_top_picks(pool, preset, provider=provider)
                st.session_state[picks_key] = picks
            except RuntimeError as e:
                st.error(f"LLM Error: {e}")

    if picks_key in st.session_state:
        picks = st.session_state[picks_key]
        if picks:
            pick_cols = st.columns(len(picks))
            for col, pick in zip(pick_cols, picks):
                sym        = pick["ticker"].replace(".NS", "").replace(".BO", "")
                ti         = next((s for s in triaged_stocks if s["ticker"] == pick["ticker"]), {})
                sent       = ti.get("triage_sentiment", "NEUTRAL")
                catalyst   = ti.get("triage_catalyst", "")
                risk       = ti.get("triage_risk", "")
                conf       = ti.get("triage_confidence", 0)
                yahoo_url  = f"https://finance.yahoo.com/quote/{pick['ticker']}/"
                scr_url    = f"https://www.screener.in/company/{sym}/"
                perp_url   = f"https://www.perplexity.ai/finance/{sym}.NS/analysis"
                with col:
                    st.markdown(
                        f"""<div style="border:1px solid #3a5a7a;border-radius:8px;
                            padding:16px;background:#0f2033;margin-bottom:8px">
                            <p style="font-size:1.05em;font-weight:700;color:#00d4ff;margin:0 0 2px 0">
                              {sentiment_badge(sent)} {pick['company_name']}</p>
                            <p style="font-size:0.78em;color:#888;margin:0 0 8px 0">
                              {pick['ticker']} · {sent} {conf:.0%}</p>
                            <p style="font-size:0.85em;color:#ddd;margin:0 0 6px 0">
                              {pick['reason']}</p>
                            <p style="font-size:0.75em;color:#4ade80;margin:0 0 2px 0">
                              🟢 {catalyst or '–'}</p>
                            <p style="font-size:0.75em;color:#f97316;margin:0 0 12px 0">
                              ⚠️ {risk or '–'}</p>
                            <p style="font-size:0.72em;margin:0;line-height:2">
                              <a href="{perp_url}" target="_blank" style="color:#a78bfa;margin-right:8px">Perplexity ↗</a>
                              <a href="{yahoo_url}" target="_blank" style="color:#4da6ff;margin-right:8px">Yahoo ↗</a>
                              <a href="{scr_url}"  target="_blank" style="color:#4da6ff">Screener ↗</a>
                            </p></div>""",
                        unsafe_allow_html=True,
                    )

    st.markdown("---")

    # ── Deep Dive ─────────────────────────────────────────────────────────────
    st.markdown("### Stock Deep Dive")

    ticker_opts = [
        f"{s['ticker']} — {s.get('company_name','')}  {sentiment_badge(s.get('triage_sentiment','NEUTRAL'))}"
        for s in triaged_stocks
    ]
    selected     = st.selectbox("Select stock:", ticker_opts)
    ticker       = selected.split(" — ")[0].strip()
    company_name = selected.split(" — ")[1].split("  ")[0].strip()

    ti          = next((s for s in triaged_stocks if s["ticker"] == ticker), {})
    sentiment   = ti.get("triage_sentiment", "NEUTRAL")
    catalyst    = ti.get("triage_catalyst", "")
    risk        = ti.get("triage_risk", "")
    triage_sum  = ti.get("triage_summary", "")
    sig_hits    = ti.get("triage_signal_hits", [])
    conf        = ti.get("triage_confidence", 0)

    border_color = {"POSITIVE": "#22c55e", "NEGATIVE": "#ef4444"}.get(sentiment, "#eab308")
    st.markdown(
        f"""<div style="border-left:4px solid {border_color};padding:10px 16px;
            background:#111827;border-radius:4px;margin-bottom:16px">
            <b>{sentiment_badge(sentiment)} {sentiment}</b> ({conf:.0%} confidence)
            &nbsp;·&nbsp; {triage_sum}<br>
            <span style="color:#4ade80;font-size:0.8em">Catalyst: {catalyst or '–'}</span>
            &nbsp;&nbsp;
            <span style="color:#f97316;font-size:0.8em">Risk: {risk or '–'}</span>
        </div>""",
        unsafe_allow_html=True,
    )
    if sig_hits:
        st.caption(f"Signals found: {' · '.join(sig_hits)}")

    period = st.select_slider("Chart period", ["3mo","6mo","1y","2y","5y"], value="1y")

    col_chart, col_metrics = st.columns([3, 1])
    with st.spinner(f"Loading {ticker}…"):
        hist    = get_price_history(ticker, period=period)
        info    = get_stock_info(ticker)
        metrics = get_key_metrics(info)

    with col_chart:
        if not hist.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=hist["Date"], open=hist["Open"], high=hist["High"],
                low=hist["Low"], close=hist["Close"], name=ticker,
            )])
            fig.update_layout(
                title=f"{company_name} ({ticker})",
                xaxis_title="Date", yaxis_title="Price (₹)",
                height=420, xaxis_rangeslider_visible=False,
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Price history unavailable.")

    with col_metrics:
        st.markdown("**Key Metrics**")
        for k, v in metrics.items():
            st.markdown(f"**{k}:** {v}")

    st.markdown("---")
    thesis_lbl = "Gemini 2.5 Flash Lite" if provider == "gemini" else "Claude Sonnet"
    if st.button(f"📝 Generate Investment Thesis ({thesis_lbl})", type="primary"):
        enriched_theme = f"{preset} · Triage: {sentiment} — {triage_sum}"
        with st.spinner("Generating thesis…"):
            try:
                thesis = generate_stock_thesis(ticker, company_name, enriched_theme, info, provider=provider)
                st.session_state[f"thesis_{ticker}"] = thesis
            except RuntimeError as e:
                st.error(f"LLM Error: {e}")

    if f"thesis_{ticker}" in st.session_state:
        st.markdown("#### Investment Thesis")
        st.markdown(st.session_state[f"thesis_{ticker}"])

    st.markdown("---")
    st.markdown("#### Add to Paper Portfolio")
    strategies = [s for s in get_all_strategies() if s["status"] == "active"]
    if strategies:
        strat_names = {s["id"]: s["name"] for s in strategies}
        sel_sid = st.selectbox("Strategy:", list(strat_names.keys()),
                               format_func=lambda x: strat_names[x])
        try:
            cur_price = float(
                metrics.get("Current Price","₹0").replace("₹","").replace(",","") or 0)
        except Exception:
            cur_price = 0.0
        ca, cb = st.columns(2)
        with ca:
            alloc = st.number_input("Allocate (₹):", value=10000.0, step=1000.0)
        with cb:
            buy_px = st.number_input("Buy price (₹):", value=cur_price, step=0.5)
        shares = alloc / buy_px if buy_px > 0 else 0
        st.caption(f"= {shares:.2f} shares @ ₹{buy_px:.2f}")
        if st.button("➕ Add to Portfolio"):
            add_trade(sel_sid, ticker, company_name, round(shares, 4), buy_px)
            st.success(f"Added {shares:.2f} shares of {ticker} to '{strat_names[sel_sid]}'")
    else:
        st.info("Create a strategy first (📊 Portfolio).")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Portfolio":
    st.title("📊 Portfolio")
    tab_create, tab_view, tab_trades, tab_compare = st.tabs(
        ["Create Strategy", "Portfolio View", "Manage Trades", "Compare Strategies"])

    with tab_create:
        st.markdown("### New Strategy")
        with st.form("new_strategy"):
            sname    = st.text_input("Name", placeholder="v1 — Value Hunt May 2026")
            stheme   = st.text_input("Thesis / Preset", placeholder="Value Hunt screen")
            scapital = st.number_input("Virtual Capital (₹)", value=100000.0, step=10000.0)
            snotes   = st.text_area("Notes (optional)")
            if st.form_submit_button("Create Strategy", type="primary") and sname:
                sid = create_strategy(sname, stheme, scapital, snotes)
                st.success(f"Created **{sname}** (ID {sid}) with ₹{scapital:,.0f}")

        st.markdown("---")
        st.markdown("### All Strategies")
        for s in get_all_strategies():
            ca, cb = st.columns([4, 1])
            with ca:
                icon = "✅" if s["status"] == "active" else "🔒"
                st.markdown(f"{icon} **{s['name']}** · {s['theme']} · ₹{s['virtual_capital']:,.0f}")
                st.caption(f"Created: {s['created_at'][:10]}")
            with cb:
                if s["status"] == "active":
                    if st.button("Close", key=f"close_{s['id']}"):
                        close_strategy(s["id"])
                        st.rerun()

    with tab_view:
        strategies = [s for s in get_all_strategies() if s["status"] == "active"]
        if not strategies:
            st.info("No active strategies. Create one first.")
        else:
            sel = st.selectbox("Strategy:", [s["id"] for s in strategies],
                               format_func=lambda x: next(s["name"] for s in strategies if s["id"] == x))
            with st.spinner("Computing P&L…"):
                summary = compute_portfolio_value(sel)
            if summary:
                strat = summary["strategy"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Capital", f"₹{strat['virtual_capital']:,.0f}")
                c2.metric("Value", f"₹{summary['total_value']:,.0f}", f"{summary['total_pnl_pct']:+.2f}%")
                c3.metric("Cash", f"₹{summary['cash_remaining']:,.0f}")
                nifty = summary.get("nifty_current")
                c4.metric("Nifty 50", f"{nifty:,.0f}" if nifty else "N/A")

                if summary["positions"]:
                    pos_df = pd.DataFrame(summary["positions"])
                    st.dataframe(pos_df[[c for c in [
                        "ticker","company_name","shares","buy_price","current_price",
                        "cost_basis","current_value","pnl","pnl_pct"
                    ] if c in pos_df.columns]].rename(columns={
                        "ticker":"Ticker","company_name":"Company","shares":"Shares",
                        "buy_price":"Buy ₹","current_price":"Curr ₹","cost_basis":"Cost ₹",
                        "current_value":"Value ₹","pnl":"P&L ₹","pnl_pct":"P&L %",
                    }), use_container_width=True, hide_index=True)

                    snaps = get_daily_snapshots(sel)
                    if len(snaps) > 1:
                        snap_df = pd.DataFrame(snaps)
                        initial = strat["virtual_capital"]
                        snap_df["return_pct"] = (snap_df["portfolio_value"] - initial) / initial * 100
                        if snap_df["nifty_value"].notna().any():
                            ns = snap_df["nifty_value"].dropna().iloc[0]
                            snap_df["nifty_return_pct"] = (snap_df["nifty_value"] - ns) / ns * 100
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=snap_df["snapshot_date"],
                            y=snap_df["return_pct"], name="Portfolio %",
                            line=dict(color="cyan", width=2)))
                        if "nifty_return_pct" in snap_df.columns:
                            fig.add_trace(go.Scatter(x=snap_df["snapshot_date"],
                                y=snap_df["nifty_return_pct"], name="Nifty 50 %",
                                line=dict(color="orange", width=2, dash="dash")))
                        fig.update_layout(title="Portfolio vs Nifty 50 (%)",
                            yaxis_title="Return %", height=340, template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                    if st.button("💾 Save Snapshot"):
                        save_daily_snapshot(sel, summary["total_value"], summary["cash_remaining"])
                        st.success("Snapshot saved")
                else:
                    st.info("No positions yet. Add from 💡 Suggestions.")

    with tab_trades:
        for s in get_all_strategies():
            pass
        strategies_all = get_all_strategies()
        if strategies_all:
            sel = st.selectbox("Strategy:", [s["id"] for s in strategies_all],
                               format_func=lambda x: next(s["name"] for s in strategies_all if s["id"] == x),
                               key="trades_strat")
            for t in get_all_trades(sel):
                status = "Open" if not t["sell_date"] else f"Closed @ ₹{t['sell_price']}"
                ca, cb = st.columns([4, 1])
                with ca:
                    st.markdown(f"**{t['ticker']}** {t['company_name']} | "
                                f"{t['shares']:.2f} @ ₹{t['buy_price']:.2f} | {status}")
                with cb:
                    if not t["sell_date"]:
                        sp = st.number_input("Sell ₹", key=f"sell_{t['id']}", value=t["buy_price"])
                        if st.button("Sell", key=f"do_sell_{t['id']}"):
                            close_trade(t["id"], sp)
                            st.rerun()

    with tab_compare:
        st.markdown("### Strategy Comparison")
        comparison = compare_strategies()
        if not comparison:
            st.info("No strategy data yet.")
        else:
            df = pd.DataFrame(comparison)
            st.dataframe(df[[c for c in [
                "name","theme","status","virtual_capital","current_value",
                "return_pct","nifty_return_pct","alpha_pct","created_at"
            ] if c in df.columns]].rename(columns={
                "name":"Strategy","theme":"Theme","status":"Status",
                "virtual_capital":"Capital ₹","current_value":"Value ₹",
                "return_pct":"Return %","nifty_return_pct":"Nifty %",
                "alpha_pct":"Alpha %","created_at":"Created",
            }), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: AUTO TRADER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Auto Trader":
    st.title("⚙️ Auto Trader")
    st.caption(
        "Fully automated — Screen Nifty 500 → Sector-aware triage → LLM decisions. "
        "Capital: ₹5,00,000 · Max 5 positions · Stop-loss: −15%"
    )

    trader = AutoTrader()
    trader.ensure_portfolio()

    col_run, col_last = st.columns([2, 4])
    with col_run:
        run_now = st.button("▶ Run Pipeline Now", type="primary")
    with col_last:
        last = trader.get_last_run()
        if last and last.get("completed_at"):
            st.caption(
                f"Last run: {last['completed_at'][:16]} — "
                f"{last['buys_made']} buys · {last['sells_made']} sells · "
                f"{last['stop_losses_triggered']} stop-losses · "
                f"{last.get('themes_found', 0)} POSITIVE triage results"
            )
        else:
            st.caption("No completed runs yet.")

    if run_now:
        with st.spinner("Running: Screen → Triage → Decide… (~3–4 min)"):
            log = trader.run_pipeline()
        if log.get("errors"):
            st.error(f"Pipeline error: {log['errors'][0]}")
        else:
            parts = []
            if log["buys"]:        parts.append(f"Bought: {', '.join(log['buys'])}")
            if log["sells"]:       parts.append(f"Sold: {', '.join(log['sells'])}")
            if log["stop_losses"]: parts.append(f"Stop-loss: {', '.join(log['stop_losses'])}")
            if not parts:          parts.append("No trades (held or no signals)")
            st.success(" · ".join(parts))
            if log.get("summary"):
                st.info(log["summary"])
        st.rerun()

    st.markdown("---")
    st.markdown("### Portfolio Overview")
    with st.spinner("Fetching live prices…"):
        state = trader.get_portfolio_state()
    port = state["portfolio"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Capital",   f"₹{port['capital']:,.0f}")
    c2.metric("Value",     f"₹{state['total_value']:,.0f}", f"{state['total_pnl_pct']:+.2f}%")
    c3.metric("Cash",      f"₹{state['cash_remaining']:,.0f}")
    c4.metric("Positions", f"{len(state['positions'])}/5")
    nifty = state.get("nifty_current")
    c5.metric("Nifty 50",  f"{nifty:,.0f}" if nifty else "N/A")

    if state["positions"]:
        st.markdown("### Open Positions")
        pos_df = pd.DataFrame(state["positions"])
        st.dataframe(pos_df[[c for c in [
            "ticker","company_name","shares","entry_price","current_price",
            "cost_basis","current_value","pnl","pnl_pct","days_held","reason"
        ] if c in pos_df.columns]].rename(columns={
            "ticker":"Ticker","company_name":"Company","shares":"Shares",
            "entry_price":"Entry ₹","current_price":"Curr ₹","cost_basis":"Cost ₹",
            "current_value":"Value ₹","pnl":"P&L ₹","pnl_pct":"P&L %",
            "days_held":"Days","reason":"Reason",
        }), use_container_width=True, hide_index=True)

    last = trader.get_last_run()
    if last and last.get("llm_summary") and not str(last["llm_summary"]).startswith("ERROR"):
        st.markdown("### Last Run — LLM Rationale")
        st.info(last["llm_summary"])

    th = trader.get_trade_history()
    if th:
        st.markdown("### Trade History")
        th_df = pd.DataFrame(th)
        cols_th = [c for c in [
            "trade_date","ticker","company_name","action","shares",
            "price","total_value","reason",
        ] if c in th_df.columns]
        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1:
            tickers_in_history = sorted(th_df["ticker"].unique().tolist()) if "ticker" in th_df.columns else []
            selected_tickers = st.multiselect(
                "Filter by ticker", tickers_in_history,
                placeholder="All tickers"
            )
        with filter_col2:
            action_filter = st.selectbox("Action", ["All", "BUY", "SELL"])
        if selected_tickers:
            th_df = th_df[th_df["ticker"].isin(selected_tickers)]
        if action_filter != "All" and "action" in th_df.columns:
            th_df = th_df[th_df["action"] == action_filter]

        st.dataframe(th_df[cols_th].rename(columns={
            "trade_date": "Date", "ticker": "Ticker", "company_name": "Company",
            "action": "Action", "shares": "Shares", "price": "Price ₹",
            "total_value": "Value ₹", "reason": "Reason",
        }), use_container_width=True, hide_index=True)

        # Equity curve
        if "trade_date" in th_df.columns and "total_value" in th_df.columns:
            st.markdown("### Equity Curve")
            th_df_sorted = th_df.sort_values("trade_date")
            th_df_sorted["cumulative"] = th_df_sorted.apply(
                lambda row: row["total_value"] if row.get("action") == "BUY" else -row["total_value"],
                axis=1,
            ).cumsum() + port["capital"]
            st.line_chart(th_df_sorted.set_index("trade_date")["cumulative"])


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — News Feed
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📡 News Feed":
    st.markdown("## 📡 News Feed")
    st.markdown("Raw news from your database — search, filter, and explore what's moving the market.")

    # ── Controls ──────────────────────────────────────────────────────────────
    nf_col1, nf_col2, nf_col3 = st.columns([3, 2, 1])
    with nf_col1:
        news_query = st.text_input("🔍 Search keywords", placeholder="repo rate, FDA, EV, USD…")
    with nf_col2:
        sector_opts = [
            "All Sectors",
            "FINANCIAL SERVICES", "INFORMATION TECHNOLOGY", "HEALTHCARE",
            "AUTOMOBILE AND AUTO COMPONENTS", "CAPITAL GOODS", "CONSTRUCTION",
            "CHEMICALS", "CONSUMER DURABLES", "FAST MOVING CONSUMER GOODS",
            "OIL GAS & CONSUMABLE FUELS", "POWER", "METALS & MINING",
            "TELECOMMUNICATION", "REALTY", "TEXTILES",
            "MEDIA ENTERTAINMENT & PUBLICATION",
        ]
        news_sector = st.selectbox("Sector filter", sector_opts)
    with nf_col3:
        news_hours = st.selectbox("Time window", [6, 12, 24, 48, 72], index=2)
        news_hours = int(news_hours)

    st.markdown("---")

    # ── Fetch news from DB ────────────────────────────────────────────────────
    try:
        from modules.db import get_conn
        from datetime import datetime, timedelta

        conn = get_conn()
        cur  = conn.cursor()

        since_dt = (datetime.utcnow() - timedelta(hours=news_hours)).isoformat()

        base_sql  = "SELECT * FROM news_articles WHERE published_at >= %s"
        params    = [since_dt]

        if news_query.strip():
            keywords = [k.strip() for k in news_query.replace(",", " ").split() if k.strip()]
            if keywords:
                kw_clause = " OR ".join(["(title ILIKE %s OR summary ILIKE %s)" for _ in keywords])
                base_sql += f" AND ({kw_clause})"
                for kw in keywords:
                    params.extend([f"%{kw}%", f"%{kw}%"])

        if news_sector != "All Sectors":
            from modules.news_triage import SECTOR_SIGNALS
            sector_kws = SECTOR_SIGNALS.get(news_sector, {}).get("keywords", [])
            if sector_kws:
                sk_clause = " OR ".join(["(title ILIKE %s OR summary ILIKE %s)" for _ in sector_kws])
                base_sql += f" AND ({sk_clause})"
                for sk in sector_kws:
                    params.extend([f"%{sk}%", f"%{sk}%"])

        base_sql += " ORDER BY published_at DESC LIMIT 200"
        rows = cur.execute(base_sql, params).fetchall()
        conn.close()

        if rows:
            news_list = [dict(r) for r in rows]
            st.markdown(f"**{len(news_list)} articles** found in last **{news_hours}h**")

            # ── Summary pills ────────────────────────────────────────────────
            sentiment_counts = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
            for item in news_list:
                s = (item.get("sentiment") or "NEUTRAL").upper()
                if s in sentiment_counts:
                    sentiment_counts[s] += 1
                else:
                    sentiment_counts["NEUTRAL"] += 1

            pill_c1, pill_c2, pill_c3, pill_c4 = st.columns(4)
            pill_c1.metric("Total",    len(news_list))
            pill_c2.metric("🟢 Positive", sentiment_counts["POSITIVE"])
            pill_c3.metric("🟡 Neutral",  sentiment_counts["NEUTRAL"])
            pill_c4.metric("🔴 Negative", sentiment_counts["NEGATIVE"])

            st.markdown("---")

            # ── Article cards ────────────────────────────────────────────────
            for item in news_list[:50]:
                pub  = item.get("published_at", "")[:16].replace("T", " ")
                sent = (item.get("sentiment") or "").upper()
                badge = {"POSITIVE": "🟢", "NEGATIVE": "🔴"}.get(sent, "🟡")
                src   = item.get("source") or item.get("feed_name") or ""
                title = item.get("title") or "(no title)"
                url   = item.get("url") or item.get("link") or ""
                summary = item.get("summary") or ""

                with st.expander(f"{badge} {title}  —  {pub}  [{src}]"):
                    if summary:
                        st.markdown(summary[:600] + ("…" if len(summary) > 600 else ""))
                    if url:
                        st.markdown(f"[Read full article ↗]({url})")
                    tickers_mentioned = item.get("tickers_mentioned") or ""
                    if tickers_mentioned:
                        st.caption(f"Tickers mentioned: {tickers_mentioned}")

            if len(news_list) > 50:
                st.info(f"Showing top 50 of {len(news_list)} articles. Refine your search to narrow results.")
        else:
            st.info(f"No articles found in the last {news_hours}h with the current filters. "
                    "Try expanding the time window or clearing keyword filters.")

    except Exception as e:
        st.error(f"Could not load news feed: {e}")
        st.caption("Make sure your database is connected and the news_articles table exists. "
                   "Run the Auto Trader pipeline once to populate it.")

    # ── Manual refresh button ─────────────────────────────────────────────────
    st.markdown("---")
    nf_btn_col1, nf_btn_col2 = st.columns([1, 3])
    with nf_btn_col1:
        if st.button("🔄 Refresh News DB", use_container_width=True):
            with st.spinner("Fetching latest news from all feeds…"):
                try:
                    from modules.news_fetcher import refresh_news_db
                    count = refresh_news_db()
                    st.success(f"Fetched {count} new articles.")
                except Exception as e:
                    st.warning(f"News refresh failed: {e}")
    with nf_btn_col2:
        st.caption("News is automatically refreshed when you run the Auto Trader pipeline. "
                   "Use this button to pull the latest articles manually.")

    # ── Sector heatmap ────────────────────────────────────────────────────────
    with st.expander("📊 Sector News Volume (last 24h)", expanded=False):
        try:
            from modules.news_triage import SECTOR_SIGNALS
            from modules.db import get_conn as _gc
            from datetime import datetime as _dt, timedelta as _td

            _conn  = _gc()
            _cur   = _conn.cursor()
            _since = (_dt.utcnow() - _td(hours=24)).isoformat()
            _rows  = _cur.execute(
                "SELECT title, summary FROM news_articles WHERE published_at >= %s", (_since,)
            ).fetchall()
            _conn.close()

            if _rows:
                sector_hits = {}
                all_text = " ".join(
                    f"{r['title']} {r.get('summary','')}" for r in _rows
                ).lower()
                for sector, sig in SECTOR_SIGNALS.items():
                    hits = sum(all_text.count(kw.lower()) for kw in sig["keywords"])
                    if hits > 0:
                        sector_hits[sector] = hits

                if sector_hits:
                    heat_df = pd.DataFrame(
                        sorted(sector_hits.items(), key=lambda x: x[1], reverse=True),
                        columns=["Sector", "Keyword Hits"]
                    )
                    st.bar_chart(heat_df.set_index("Sector")["Keyword Hits"])
                else:
                    st.info("No sector keyword hits in recent news.")
            else:
                st.info("No articles in the last 24h to analyse.")
        except Exception as _e:
            st.caption(f"Heatmap unavailable: {_e}")

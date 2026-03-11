"""Main Streamlit app — Top-Down NSE/BSE Stock Analysis."""
import sys
import json
import streamlit as st

# Fix Windows cp1252 console so Unicode chars (₹ etc.) don't crash print() calls
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from modules.db import init_db
from modules.news_fetcher import fetch_all_feeds, get_recent_headlines
from modules.theme_engine import extract_themes
from modules.screener import screen_stocks_for_theme
from modules.stock_detail import get_price_history, get_stock_info, generate_stock_thesis, get_key_metrics, summarize_top_picks
from modules.paper_trader import (
    create_strategy, get_all_strategies, close_strategy,
    add_trade, close_trade, get_open_trades, get_all_trades,
    compute_portfolio_value, save_daily_snapshot, update_all_snapshots,
    get_daily_snapshots, compare_strategies, get_strategy
)
from modules.auto_trader import AutoTrader
from modules.llm_client import provider_label
from config import SCREENER_DEFAULTS, ANTHROPIC_API_KEY, GEMINI_API_KEY

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Top-Down Investor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Initialize DB ────────────────────────────────────────────────────────────
init_db()

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
st.sidebar.title("NSE Top-Down Investor")
page = st.sidebar.radio(
    "Navigate",
    ["Today's Themes", "News Feed", "Screen Stocks", "Stock Deep Dive", "Paper Trading", "Strategy Comparison", "Auto Trader"],
    index=0,
)

st.sidebar.markdown("---")

# ─── LLM Provider Selector ────────────────────────────────────────────────────
st.sidebar.markdown("**LLM Provider**")

# Determine which providers are configured
provider_options = []
if GEMINI_API_KEY:
    provider_options.append("gemini")
if ANTHROPIC_API_KEY:
    provider_options.append("claude")
if not provider_options:
    provider_options = ["gemini"]  # show even if unconfigured, will fail gracefully

provider_labels = {
    "claude": "Claude (Anthropic) — paid, best quality",
    "gemini": "Gemini 2.5 Flash Lite (Google) — free tier",
}

selected_provider = st.sidebar.radio(
    "Use for theme extraction & thesis:",
    options=provider_options,
    format_func=lambda p: provider_labels.get(p, p),
    key="llm_provider",
)

if selected_provider == "gemini" and not GEMINI_API_KEY:
    st.sidebar.warning("GEMINI_API_KEY not set in .env")
if selected_provider == "claude" and not ANTHROPIC_API_KEY:
    st.sidebar.warning("ANTHROPIC_API_KEY not set in .env")

st.sidebar.markdown("---")
if st.sidebar.button("Refresh News + Update P&L", type="primary"):
    with st.spinner("Fetching latest news..."):
        n = fetch_all_feeds()
        st.sidebar.success(f"Fetched {n} new articles")
    with st.spinner("Updating portfolio snapshots..."):
        update_all_snapshots()
        st.sidebar.success("Snapshots updated")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: TODAY'S THEMES
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Today's Themes":
    st.title("Today's Macro Investment Themes")
    st.caption("Extracted from Indian financial news (ET, Moneycontrol, Livemint, NSE) using Claude Haiku")

    col1, col2 = st.columns([2, 1])

    with col1:
        refresh = st.button("Extract Themes from Today's News", type="primary")

    with col2:
        force = st.checkbox("Force re-extract (ignore cache)")

    provider = st.session_state.get("llm_provider", "claude")
    model_label = "Gemini 2.5 Flash Lite" if provider == "gemini" else "Claude Haiku"

    if refresh:
        with st.spinner("Fetching news feeds..."):
            n = fetch_all_feeds()
            st.info(f"Fetched {n} new articles")

        with st.spinner(f"Analyzing with {model_label}..."):
            try:
                # Always extract on button click; force=True also clears existing cache
                themes = extract_themes(force_refresh=True, provider=provider)
                if themes:
                    st.success(f"Found {len(themes)} investment themes")
                else:
                    st.warning("No headlines found. Click 'Refresh News' first.")
            except RuntimeError as e:
                st.error(f"**LLM Error ({provider}):** {e}")
                themes = []

    # Load cached themes — never calls LLM, returns [] if nothing extracted yet
    themes = extract_themes(force_refresh=False, provider=provider)

    if not themes:
        st.info(f"Click **'Extract Themes from Today's News'** to begin.  \nSelected model: **{model_label}**")
        st.stop()

    st.markdown("---")
    for i, t in enumerate(themes):
        conf_color = "green" if t["confidence"] > 0.75 else "orange" if t["confidence"] > 0.5 else "red"
        with st.expander(
            f"**{t['theme']}** — Confidence: :{conf_color}[{t['confidence']:.0%}]",
            expanded=(i < 3)
        ):
            col_ev, col_sec = st.columns([3, 2])
            with col_ev:
                st.markdown("**Supporting Evidence:**")
                for ev in t["evidence"]:
                    st.markdown(f"- {ev}")
            with col_sec:
                st.markdown("**Sectors to explore:**")
                for sec in t["sectors"]:
                    st.markdown(f"- {sec}")

    # Show recent headlines count
    headlines = get_recent_headlines(hours_back=48)
    st.sidebar.info(f"Headlines in DB: {len(headlines)}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: NEWS FEED
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "News Feed":
    st.title("News Feed")
    st.caption("All headlines fetched from Indian financial RSS feeds (last 48 hours)")

    from modules.db import get_conn
    conn = get_conn()
    c = conn.cursor()

    # Filter controls
    col_src, col_search, col_limit = st.columns([2, 3, 1])
    with col_src:
        sources = [r[0] for r in c.execute("SELECT DISTINCT source FROM news ORDER BY source").fetchall()]
        selected_sources = st.multiselect("Filter by source:", sources, default=sources)
    with col_search:
        search_term = st.text_input("Search headlines:", placeholder="e.g. RBI, SEBI, inflation")
    with col_limit:
        limit = st.selectbox("Show:", [50, 100, 200, 500], index=1)

    conn.close()

    if st.button("Refresh News Now", type="primary"):
        with st.spinner("Fetching RSS feeds..."):
            n = fetch_all_feeds()
            st.success(f"Fetched {n} new articles")
        st.rerun()

    # Query
    from modules.db import get_conn as _gc
    conn2 = _gc()
    conn2.row_factory = __import__('sqlite3').Row
    c2 = conn2.cursor()

    placeholders = ",".join("?" * len(selected_sources)) if selected_sources else "''"
    query = f"""
        SELECT source, title, summary, link, published_at, fetched_at
        FROM news
        WHERE source IN ({placeholders})
        ORDER BY fetched_at DESC
        LIMIT ?
    """
    params = list(selected_sources) + [limit]
    rows = c2.execute(query, params).fetchall() if selected_sources else []
    conn2.close()

    # Apply search filter
    if search_term:
        term = search_term.lower()
        rows = [r for r in rows if term in r["title"].lower() or term in (r["summary"] or "").lower()]

    st.markdown(f"**{len(rows)} articles** {'matching' if search_term else 'from'} selected sources")
    st.markdown("---")

    for r in rows:
        with st.expander(f"[{r['source']}] {r['title']}", expanded=False):
            if r["summary"]:
                st.write(r["summary"][:300] + ("..." if len(r["summary"]) > 300 else ""))
            col_l, col_t = st.columns([3, 1])
            with col_l:
                if r["link"]:
                    st.markdown(f"[Read full article]({r['link']})")
            with col_t:
                ts = r["published_at"] or r["fetched_at"] or ""
                st.caption(ts[:16])


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: SCREEN STOCKS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Screen Stocks":
    st.title("Screen NSE Stocks by Theme")

    themes = extract_themes(force_refresh=False)
    if not themes:
        st.warning("No themes extracted yet. Go to 'Today's Themes' first.")
        st.stop()

    theme_names = [t["theme"] for t in themes]
    selected_name = st.selectbox("Select a theme to screen for:", theme_names)
    selected_theme = next(t for t in themes if t["theme"] == selected_name)

    st.markdown(f"**Sectors:** {', '.join(selected_theme['sectors'])}")

    # Screener filters
    with st.expander("Adjust Screener Filters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_mcap = st.number_input("Min Market Cap (₹ Cr)", value=SCREENER_DEFAULTS["min_market_cap_cr"], step=100)
        with col2:
            max_pe = st.number_input("Max P/E", value=float(SCREENER_DEFAULTS["max_pe"]), step=5.0)
        with col3:
            max_de = st.number_input("Max Debt/Equity", value=float(SCREENER_DEFAULTS["max_debt_equity"]), step=0.1)

    col_btn, col_force = st.columns([3, 2])
    with col_btn:
        screen_btn = st.button("Screen Stocks", type="primary")
    with col_force:
        force_rescreen = st.checkbox(
            "Force re-screen",
            help="Clear today's cached results and re-fetch from Nifty 500 (useful when changing filters)"
        )

    if screen_btn:
        with st.spinner(f"Screening Nifty 500 for '{selected_name}'... (fetching fundamentals via yfinance, please wait)"):
            filters = {"min_market_cap_cr": min_mcap, "max_pe": max_pe, "max_debt_equity": max_de}
            stocks = screen_stocks_for_theme(
                theme_id=selected_theme["id"],
                theme_name=selected_name,
                sectors=selected_theme["sectors"],
                filters=filters,
                force=force_rescreen,
            )

        if not stocks:
            st.warning("No stocks matched the filters. Try relaxing the filters or a different theme.")
        else:
            st.success(f"Found {len(stocks)} stocks matching theme: **{selected_name}**")
            df = pd.DataFrame(stocks)
            display_cols = {
                "ticker": "Ticker",
                "company_name": "Company",
                "sector": "Sector",
                "market_cap_cr": "Mkt Cap (₹Cr)",
                "current_price": "Price (₹)",
                "pe": "P/E",
                "pb": "P/B",
                "roe": "ROE %",
                "debt_equity": "D/E",
                "revenue_growth": "Rev Growth %",
                "eps_growth": "EPS Growth %",
            }
            df_display = df[[c for c in display_cols if c in df.columns]].rename(columns=display_cols)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Store selection in session state for deep dive
            st.session_state["screened_stocks"] = stocks
            st.session_state["current_theme"] = selected_name
            st.info("Go to 'Stock Deep Dive' to analyze individual stocks.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: STOCK DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Stock Deep Dive":
    st.title("Stock Deep Dive")

    stocks = st.session_state.get("screened_stocks", [])
    theme_name = st.session_state.get("current_theme", "")

    if not stocks:
        st.warning("Screen stocks first (Page 2), then come here to analyze.")
        st.stop()

    # ── AI Top Picks ──────────────────────────────────────────────────────────
    st.markdown("### AI Top Picks")
    provider = st.session_state.get("llm_provider", "claude")
    picks_key = f"top_picks_{theme_name}"

    col_picks_btn, col_picks_label = st.columns([2, 3])
    with col_picks_btn:
        gen_picks = st.button(
            "Analyse All & Surface Best Stocks",
            type="primary",
            help=f"Ask AI to review all {len(stocks)} screened stocks and highlight the 2-3 most compelling"
        )
    with col_picks_label:
        if picks_key in st.session_state:
            st.caption(f"Showing picks for theme: **{theme_name}**")

    if gen_picks:
        model_label = "Gemini 2.5 Flash Lite" if provider == "gemini" else "Claude Haiku"
        with st.spinner(
            f"Fetching live data from Yahoo Finance + Perplexity Finance "
            f"(news, analyst views, AI analysis) for top stocks, "
            f"then selecting with {model_label}..."
        ):
            try:
                picks = summarize_top_picks(stocks, theme_name, provider=provider)
                st.session_state[picks_key] = picks
            except RuntimeError as e:
                st.error(f"**LLM Error:** {e}")

    if picks_key in st.session_state:
        picks = st.session_state[picks_key]
        if picks:
            cols = st.columns(len(picks))
            for col, pick in zip(cols, picks):
                symbol = pick["ticker"].replace(".NS", "").replace(".BO", "")
                yahoo_url = f"https://finance.yahoo.com/quote/{pick['ticker']}/"
                google_url = f"https://www.google.com/finance/quote/{symbol}:NSE"
                screener_url = f"https://www.screener.in/company/{symbol}/"
                perplexity_url = f"https://www.perplexity.ai/finance/{symbol}.NS/analysis"
                with col:
                    st.markdown(
                        f"""<div style="border:1px solid #3a5a7a; border-radius:8px; padding:16px; background:#0f2033;">
                        <p style="font-size:1.05em; font-weight:700; color:#00d4ff; margin:0 0 2px 0;">{pick['company_name']}</p>
                        <p style="font-size:0.78em; color:#888; margin:0 0 10px 0;">{pick['ticker']}</p>
                        <p style="font-size:0.87em; line-height:1.55; color:#ddd; margin:0 0 14px 0;">{pick['reason']}</p>
                        <p style="font-size:0.75em; margin:0; line-height:2;">
                          <a href="{perplexity_url}" target="_blank" style="color:#a78bfa; margin-right:10px;">Perplexity ↗</a>
                          <a href="{yahoo_url}" target="_blank" style="color:#4da6ff; margin-right:10px;">Yahoo Finance ↗</a>
                          <a href="{google_url}" target="_blank" style="color:#4da6ff; margin-right:10px;">Google Finance ↗</a>
                          <a href="{screener_url}" target="_blank" style="color:#4da6ff;">Screener.in ↗</a>
                        </p>
                        </div>""",
                        unsafe_allow_html=True
                    )
        else:
            st.warning("Could not generate picks — try again or check LLM provider settings.")

    st.markdown("---")

    ticker_options = [f"{s['ticker']} — {s['company_name']}" for s in stocks]
    selected = st.selectbox("Select stock for deep dive:", ticker_options)
    ticker = selected.split(" — ")[0]
    company_name = selected.split(" — ")[1]

    period = st.select_slider("Chart Period", options=["3mo", "6mo", "1y", "2y", "5y"], value="1y")

    col_chart, col_metrics = st.columns([3, 1])

    with st.spinner(f"Loading data for {ticker}..."):
        hist = get_price_history(ticker, period=period)
        info = get_stock_info(ticker)
        metrics = get_key_metrics(info)

    with col_chart:
        if not hist.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=hist["Date"],
                open=hist["Open"], high=hist["High"],
                low=hist["Low"], close=hist["Close"],
                name=ticker
            )])
            fig.update_layout(
                title=f"{company_name} ({ticker})",
                xaxis_title="Date", yaxis_title="Price (₹)",
                height=450,
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not load price history.")

    with col_metrics:
        st.markdown("**Key Metrics**")
        for k, v in metrics.items():
            st.markdown(f"**{k}:** {v}")

    # AI Thesis
    st.markdown("---")
    thesis_label = "Gemini 2.5 Flash Lite" if provider == "gemini" else "Claude Sonnet"
    if st.button(f"Generate AI Investment Thesis ({thesis_label})", type="primary"):
        with st.spinner(f"Generating thesis with {thesis_label}..."):
            try:
                thesis = generate_stock_thesis(ticker, company_name, theme_name, info, provider=provider)
                st.session_state[f"thesis_{ticker}"] = thesis
            except RuntimeError as e:
                st.error(f"**LLM Error:** {e}")

    if f"thesis_{ticker}" in st.session_state:
        st.markdown("### Investment Thesis")
        st.markdown(st.session_state[f"thesis_{ticker}"])

    # Quick add to paper portfolio
    st.markdown("---")
    st.markdown("### Add to Paper Portfolio")

    strategies = [s for s in get_all_strategies() if s["status"] == "active"]
    if strategies:
        strategy_names = {s["id"]: s["name"] for s in strategies}
        sel_sid = st.selectbox(
            "Strategy to add to:",
            options=list(strategy_names.keys()),
            format_func=lambda x: strategy_names[x]
        )
        try:
            current_price = float(metrics.get("Current Price", "₹0").replace("₹", "").replace(",", "") or 0)
        except (ValueError, TypeError):
            current_price = 0.0
        col_a, col_b = st.columns(2)
        with col_a:
            alloc_amount = st.number_input("Allocate amount (₹):", value=10000.0, step=1000.0)
        with col_b:
            buy_price = st.number_input("Buy price (₹):", value=current_price, step=0.5)
        shares = alloc_amount / buy_price if buy_price > 0 else 0
        st.caption(f"= {shares:.2f} shares @ ₹{buy_price:.2f}")

        if st.button("Add to Portfolio"):
            add_trade(sel_sid, ticker, company_name, round(shares, 4), buy_price)
            st.success(f"Added {shares:.2f} shares of {ticker} to '{strategy_names[sel_sid]}'")
    else:
        st.info("Create a paper trading strategy first (Page 4).")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PAPER TRADING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Paper Trading":
    st.title("Paper Trading — Strategy Manager")

    tab_create, tab_portfolio, tab_trades = st.tabs(["Create Strategy", "Portfolio View", "Manage Trades"])

    # ── Tab: Create Strategy ──────────────────────────────────────────────────
    with tab_create:
        st.markdown("### Create a New Strategy Version")
        with st.form("new_strategy"):
            strat_name = st.text_input("Strategy Name", placeholder="v1 Infrastructure Theme")
            strat_theme = st.text_input("Theme/Thesis", placeholder="Capital goods & infra spending")
            strat_capital = st.number_input("Virtual Capital (₹)", value=100000.0, step=10000.0)
            strat_notes = st.text_area("Notes (optional)")
            submitted = st.form_submit_button("Create Strategy", type="primary")

        if submitted and strat_name:
            sid = create_strategy(strat_name, strat_theme, strat_capital, strat_notes)
            st.success(f"Created strategy **{strat_name}** (ID: {sid}) with ₹{strat_capital:,.0f} virtual capital")

        st.markdown("---")
        st.markdown("### Active Strategies")
        strategies = get_all_strategies()
        if strategies:
            for s in strategies:
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    status_icon = "✅" if s["status"] == "active" else "🔒"
                    st.markdown(f"{status_icon} **{s['name']}** — Theme: {s['theme']} — Capital: ₹{s['virtual_capital']:,.0f}")
                    st.caption(f"Created: {s['created_at'][:10]}")
                with col_b:
                    if s["status"] == "active":
                        if st.button("Close", key=f"close_{s['id']}"):
                            close_strategy(s["id"])
                            st.rerun()
        else:
            st.info("No strategies yet.")

    # ── Tab: Portfolio View ───────────────────────────────────────────────────
    with tab_portfolio:
        strategies = [s for s in get_all_strategies() if s["status"] == "active"]
        if not strategies:
            st.info("No active strategies. Create one first.")
        else:
            sel = st.selectbox(
                "Select strategy:",
                [s["id"] for s in strategies],
                format_func=lambda x: next(s["name"] for s in strategies if s["id"] == x)
            )
            with st.spinner("Computing portfolio value..."):
                summary = compute_portfolio_value(sel)

            if summary:
                strat = summary["strategy"]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Virtual Capital", f"₹{strat['virtual_capital']:,.0f}")
                col2.metric("Portfolio Value", f"₹{summary['total_value']:,.0f}",
                            f"{summary['total_pnl_pct']:+.2f}%")
                col3.metric("Cash Remaining", f"₹{summary['cash_remaining']:,.0f}")
                nifty = summary.get("nifty_current")
                col4.metric("Nifty 50", f"{nifty:,.0f}" if nifty else "N/A")

                if summary["positions"]:
                    st.markdown("### Holdings")
                    pos_df = pd.DataFrame(summary["positions"])
                    display = pos_df[[
                        "ticker", "company_name", "shares", "buy_price",
                        "current_price", "cost_basis", "current_value", "pnl", "pnl_pct"
                    ]].rename(columns={
                        "ticker": "Ticker", "company_name": "Company",
                        "shares": "Shares", "buy_price": "Buy ₹",
                        "current_price": "Curr ₹", "cost_basis": "Cost ₹",
                        "current_value": "Value ₹", "pnl": "P&L ₹", "pnl_pct": "P&L %"
                    })
                    st.dataframe(display, use_container_width=True, hide_index=True)

                    # Equity curve
                    snapshots = get_daily_snapshots(sel)
                    if len(snapshots) > 1:
                        st.markdown("### Equity Curve")
                        snap_df = pd.DataFrame(snapshots)
                        initial = strat["virtual_capital"]
                        snap_df["return_pct"] = (snap_df["portfolio_value"] - initial) / initial * 100
                        if snap_df["nifty_value"].notna().any():
                            nifty_start = snap_df["nifty_value"].dropna().iloc[0]
                            snap_df["nifty_return_pct"] = (snap_df["nifty_value"] - nifty_start) / nifty_start * 100
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=snap_df["snapshot_date"], y=snap_df["return_pct"],
                            name="Portfolio Return %", line=dict(color="cyan", width=2)
                        ))
                        if "nifty_return_pct" in snap_df.columns:
                            fig.add_trace(go.Scatter(
                                x=snap_df["snapshot_date"], y=snap_df["nifty_return_pct"],
                                name="Nifty 50 Return %", line=dict(color="orange", width=2, dash="dash")
                            ))
                        fig.update_layout(
                            title="Portfolio vs Nifty 50 (%)",
                            yaxis_title="Return %", height=350,
                            template="plotly_dark",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    if st.button("Save Today's Snapshot"):
                        save_daily_snapshot(sel, summary["total_value"], summary["cash_remaining"])
                        st.success("Snapshot saved")
                else:
                    st.info("No open positions in this strategy. Add stocks from 'Stock Deep Dive'.")

    # ── Tab: Manage Trades ────────────────────────────────────────────────────
    with tab_trades:
        strategies = get_all_strategies()
        if not strategies:
            st.info("No strategies yet.")
        else:
            sel = st.selectbox(
                "Select strategy:",
                [s["id"] for s in strategies],
                format_func=lambda x: next(s["name"] for s in strategies if s["id"] == x),
                key="trades_strat"
            )
            all_trades = get_all_trades(sel)
            if all_trades:
                for t in all_trades:
                    status = "Open" if not t["sell_date"] else f"Closed @ ₹{t['sell_price']}"
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.markdown(
                            f"**{t['ticker']}** {t['company_name']} | "
                            f"{t['shares']:.2f} shares @ ₹{t['buy_price']:.2f} | {status}"
                        )
                    with col_b:
                        if not t["sell_date"]:
                            sell_price = st.number_input("Sell ₹", key=f"sell_{t['id']}", value=t["buy_price"])
                            if st.button("Sell", key=f"do_sell_{t['id']}"):
                                close_trade(t["id"], sell_price)
                                st.success("Trade closed")
                                st.rerun()
            else:
                st.info("No trades in this strategy.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: STRATEGY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Strategy Comparison":
    import time as _time

    st.title("Strategy Comparison")
    st.caption("Compare all strategy versions by return and alpha vs Nifty 50")

    col_ref, col_ts = st.columns([1, 3])
    with col_ref:
        manual_refresh = st.button("Refresh Live Prices", type="primary")

    # Auto-refresh on first visit, then throttle to every 5 minutes
    _now = _time.time()
    _last_refresh = st.session_state.get("comparison_refreshed_at", 0)
    _needs_refresh = manual_refresh or (_now - _last_refresh > 300)

    if _needs_refresh:
        with st.spinner("Fetching live prices for all active strategies..."):
            update_all_snapshots()
        st.session_state["comparison_refreshed_at"] = _time.time()
        with col_ts:
            st.caption(f"Updated just now")
    else:
        _secs = int(_now - _last_refresh)
        with col_ts:
            st.caption(f"Last updated {_secs}s ago — auto-refreshes every 5 min")

    comparison = compare_strategies()
    if not comparison:
        st.info("No strategies with snapshot data yet. Run paper trading for a few days first.")
        st.stop()

    df = pd.DataFrame(comparison)
    display = df[[
        "name", "theme", "status", "virtual_capital",
        "current_value", "return_pct", "nifty_return_pct", "alpha_pct", "created_at"
    ]].rename(columns={
        "name": "Strategy", "theme": "Theme", "status": "Status",
        "virtual_capital": "Capital (₹)", "current_value": "Value (₹)",
        "return_pct": "Return %", "nifty_return_pct": "Nifty %",
        "alpha_pct": "Alpha %", "created_at": "Created"
    })
    display["Created"] = display["Created"].str[:10]
    st.dataframe(display, use_container_width=True, hide_index=True)

    # Bar chart: return comparison
    if len(comparison) > 1:
        fig = go.Figure()
        fig.add_bar(
            x=[r["name"] for r in comparison],
            y=[r["return_pct"] for r in comparison],
            name="Portfolio Return %", marker_color="cyan"
        )
        fig.add_bar(
            x=[r["name"] for r in comparison],
            y=[r["nifty_return_pct"] for r in comparison],
            name="Nifty 50 Return %", marker_color="orange"
        )
        fig.update_layout(
            barmode="group", title="Strategy Returns vs Nifty 50",
            yaxis_title="Return %", template="plotly_dark", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Equity curves for all strategies
        st.markdown("### Equity Curves — All Strategies")
        fig2 = go.Figure()
        for r in comparison:
            snaps = get_daily_snapshots(r["id"])
            if len(snaps) > 1:
                snap_df = pd.DataFrame(snaps)
                initial = r["virtual_capital"]
                snap_df["return_pct"] = (snap_df["portfolio_value"] - initial) / initial * 100
                fig2.add_trace(go.Scatter(
                    x=snap_df["snapshot_date"], y=snap_df["return_pct"],
                    name=r["name"], mode="lines"
                ))
        if fig2.data:
            fig2.update_layout(
                title="Portfolio Return % Over Time",
                yaxis_title="Return %", template="plotly_dark", height=400
            )
            st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7: AUTO TRADER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Auto Trader":
    st.title("Auto Trader")
    st.caption(
        "Fully automatic portfolio: news → themes → screener → Claude buy/sell decisions. "
        "Runs daily at 11:30 AM IST (weekdays) via Windows Task Scheduler. "
        "Capital: ₹5,00,000 | Max positions: 5 | Stop-loss: −15%"
    )

    trader = AutoTrader()
    trader.ensure_portfolio()

    # ── Run Now ──────────────────────────────────────────────────────────────
    col_run, col_last = st.columns([2, 3])
    with col_run:
        run_now = st.button("Run Pipeline Now", type="primary",
                            help="Fetch news, extract themes, screen stocks, and execute trades")
    with col_last:
        last_run = trader.get_last_run()
        if last_run and last_run.get("completed_at"):
            st.caption(f"Last run: {last_run['completed_at'][:16]} — "
                       f"{last_run['buys_made']} buys, {last_run['sells_made']} sells, "
                       f"{last_run['stop_losses_triggered']} stop-losses")
        else:
            st.caption("No completed runs yet.")

    if run_now:
        with st.spinner("Running full pipeline (news → themes → screen → trade)... this takes 2–3 minutes"):
            log = trader.run_pipeline()
        if log.get("errors"):
            st.error(f"Pipeline error: {log['errors'][0]}")
        else:
            parts = []
            if log["buys"]:
                parts.append(f"Bought: {', '.join(log['buys'])}")
            if log["sells"]:
                parts.append(f"Sold: {', '.join(log['sells'])}")
            if log["stop_losses"]:
                parts.append(f"Stop-loss: {', '.join(log['stop_losses'])}")
            if not parts:
                parts.append("No trades executed (held or no signals)")
            st.success(" | ".join(parts))
            if log["summary"]:
                st.info(log["summary"])
        st.rerun()

    st.markdown("---")

    # ── Portfolio Overview ────────────────────────────────────────────────────
    st.markdown("### Portfolio Overview")
    with st.spinner("Fetching live prices..."):
        state = trader.get_portfolio_state()

    port = state["portfolio"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Capital", f"₹{port['capital']:,.0f}")
    col2.metric("Portfolio Value", f"₹{state['total_value']:,.0f}",
                f"{state['total_pnl_pct']:+.2f}%")
    col3.metric("Cash Available", f"₹{state['cash_remaining']:,.0f}")
    col4.metric("Open Positions", f"{len(state['positions'])}/5")
    nifty = state.get("nifty_current")
    col5.metric("Nifty 50", f"{nifty:,.0f}" if nifty else "N/A")

    # ── Open Positions ────────────────────────────────────────────────────────
    st.markdown("### Open Positions")
    if state["positions"]:
        pos_df = pd.DataFrame(state["positions"])
        display_cols = {
            "ticker": "Ticker",
            "company_name": "Company",
            "shares": "Shares",
            "entry_price": "Entry ₹",
            "current_price": "Current ₹",
            "cost_basis": "Cost ₹",
            "current_value": "Value ₹",
            "pnl": "P&L ₹",
            "pnl_pct": "P&L %",
            "days_held": "Days Held",
            "reason": "Buy Reason",
        }
        pos_display = pos_df[[c for c in display_cols if c in pos_df.columns]].rename(columns=display_cols)
        st.dataframe(pos_display, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions. Run the pipeline to start trading.")

    # ── Last Run Summary ──────────────────────────────────────────────────────
    last_run = trader.get_last_run()
    if last_run and last_run.get("llm_summary") and not last_run["llm_summary"].startswith("ERROR"):
        st.markdown("### Last Run — Claude's Rationale")
        st.markdown(f"> {last_run['llm_summary']}")

    st.markdown("---")

    # ── Trade History ─────────────────────────────────────────────────────────
    st.markdown("### Trade History")
    trade_history = trader.get_trade_history()
    if trade_history:
        th_df = pd.DataFrame(trade_history)

        # Filter by action type
        action_filter = st.multiselect(
            "Filter by action:",
            options=["buy", "sell", "stop_loss"],
            default=["buy", "sell", "stop_loss"],
            key="trade_action_filter",
        )
        if action_filter:
            th_df = th_df[th_df["action"].isin(action_filter)]

        display_th = {
            "trade_date": "Date",
            "action": "Action",
            "ticker": "Ticker",
            "shares": "Shares",
            "price": "Price ₹",
            "reason": "Reason",
        }
        th_display = th_df[[c for c in display_th if c in th_df.columns]].rename(columns=display_th)

        # Color-code action column
        def _action_color(val):
            colors = {"buy": "color: #22c55e", "sell": "color: #f97316", "stop_loss": "color: #ef4444"}
            return colors.get(val, "")

        styled = th_display.style.applymap(_action_color, subset=["Action"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("No trades yet.")

    # ── Equity Curve ──────────────────────────────────────────────────────────
    equity_history = trader.get_equity_history()
    if len(equity_history) > 1:
        st.markdown("### Equity Curve")
        eq_df = pd.DataFrame(equity_history)
        capital = port["capital"]
        eq_df["return_pct"] = (eq_df["portfolio_value"] - capital) / capital * 100

        # Nifty benchmark from run dates
        nifty_data = None
        try:
            import yfinance as yf
            dates = eq_df["run_date"].tolist()
            nifty_hist = yf.download("^NSEI", start=dates[0], end=dates[-1], progress=False)
            if not nifty_hist.empty:
                nifty_hist = nifty_hist["Close"].reset_index()
                nifty_hist.columns = ["Date", "nifty"]
                nifty_hist["Date"] = nifty_hist["Date"].astype(str).str[:10]
                nifty_start = nifty_hist["nifty"].iloc[0]
                nifty_hist["nifty_return_pct"] = (nifty_hist["nifty"] - nifty_start) / nifty_start * 100
                nifty_data = nifty_hist
        except Exception:
            pass

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_df["run_date"], y=eq_df["return_pct"],
            name="Auto Portfolio %", line=dict(color="cyan", width=2),
        ))
        if nifty_data is not None:
            fig.add_trace(go.Scatter(
                x=nifty_data["Date"], y=nifty_data["nifty_return_pct"],
                name="Nifty 50 %", line=dict(color="orange", width=2, dash="dash"),
            ))
        fig.update_layout(
            title="Auto Portfolio vs Nifty 50 (Return %)",
            yaxis_title="Return %", height=350, template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

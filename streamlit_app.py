"""
NSE Stock Analyzer — free dashboard for Streamlit Community Cloud
=================================================================
Keep this file in the SAME GitHub repo (root folder) as nse_analysis_api.py.
It imports every indicator, the signal score, relative strength and the backtest
straight from that file, so there is one source of truth and no separate API server.

Run locally:   streamlit run streamlit_app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import nse_analysis_api as core

st.set_page_config(page_title="NSE Stock Analyzer", page_icon="📈", layout="wide")

BENCHMARKS = {"Nifty 50": "^NSEI", "Nifty Bank": "^NSEBANK", "Nifty IT": "^CNXIT"}
STRATEGY_NAMES = {
    "ma_cross": "Moving-average crossover",
    "macd": "MACD crossover",
    "rsi": "RSI mean-reversion",
}


def to_ticker(sym: str) -> str:
    sym = sym.upper().strip()
    return sym if sym.endswith(".NS") else f"{sym}.NS"


@st.cache_data(ttl=300, show_spinner=False)
def load_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=period, auto_adjust=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(ticker: str) -> dict:
    return core.get_fundamentals(yf.Ticker(ticker))


def as_table(d: dict, key: str = "Metric") -> pd.DataFrame:
    return pd.DataFrame({key: list(d.keys()), "Value": ["n/a" if v is None else str(v) for v in d.values()]})


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("NSE symbol", "RELIANCE", help="e.g. RELIANCE, TCS, HDFCBANK, M&M")
    bench_name = st.selectbox("Benchmark index", list(BENCHMARKS))
    st.caption("Educational tool, not investment advice. Data: Yahoo Finance via yfinance (may be delayed).")

st.title("📈 NSE Stock Analyzer")
if not symbol.strip():
    st.info("Enter an NSE symbol in the sidebar.")
    st.stop()

ticker = to_ticker(symbol)
try:
    hist = load_history(ticker)
except Exception as e:
    st.error(f"Could not fetch data from Yahoo Finance: {e}. If it says 'Too Many Requests', wait a minute and retry.")
    st.stop()
if hist.empty:
    st.error(f"No data found for {ticker}. Check the symbol.")
    st.stop()

tab_overview, tab_chart, tab_rs, tab_bt, tab_watch = st.tabs(
    ["Overview", "Chart", "Relative strength", "Backtest", "Watchlist"]
)

# ── Overview ──────────────────────────────────────────────────────────────
with tab_overview:
    technical, ca_flag = core.get_technical_snapshot(hist)
    signal = core.compute_signal_score(technical)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{ticker} price (₹)", technical["current_price"])
    c2.metric("Signal score", f"{signal['score']:+d} / 5", signal["label"], delta_color="off")
    c3.metric("50/200 MA trend", (technical["ma_trend"] or "n/a").title())

    if ca_flag:
        st.warning("A >15% single-day move was found in the last 2 years. Moving averages and Bollinger figures "
                   "may be distorted by a demerger or similar event. Verify before trusting the signals.")

    st.subheader("Why this score?")
    st.dataframe(pd.DataFrame(signal["components"]), hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Technical indicators")
        st.dataframe(as_table(technical, "Indicator"), hide_index=True)
    with right:
        st.subheader("Fundamentals")
        try:
            fundamentals = load_fundamentals(ticker)
            clean = {k: v for k, v in fundamentals.items() if not k.startswith("_")}
            st.dataframe(as_table(clean), hide_index=True)
            st.caption("Margins, D/E and ROE are not meaningful for banks and NBFCs.")
        except Exception as e:
            st.info(f"Fundamentals unavailable right now: {e}")

# ── Chart ─────────────────────────────────────────────────────────────────
with tab_chart:
    months = st.slider("Months to show", 1, 24, 6)
    close = hist["Close"]
    rsi = core.calculate_rsi(close)
    macd_line, signal_line, macd_hist = core.calculate_macd(close)
    ma50, ma200 = core.calculate_moving_averages(close)
    view = hist.tail(months * 21)
    idx = view.index

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=idx, open=view["Open"], high=view["High"], low=view["Low"],
                                 close=view["Close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=ma50.loc[idx], name="MA50", line=dict(width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=ma200.loc[idx], name="MA200", line=dict(width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=idx, y=rsi.loc[idx], name="RSI 14", line=dict(width=1.3)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in macd_hist.loc[idx].fillna(0)]
    fig.add_trace(go.Bar(x=idx, y=macd_hist.loc[idx], name="MACD hist", marker_color=colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=idx, y=macd_line.loc[idx], name="MACD", line=dict(width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=idx, y=signal_line.loc[idx], name="Signal", line=dict(width=1.2)), row=3, col=1)
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(height=760, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig)

# ── Relative strength ─────────────────────────────────────────────────────
with tab_rs:
    benchmark = BENCHMARKS[bench_name]
    bench_hist = load_history(benchmark)
    if bench_hist.empty:
        st.warning(f"No data for benchmark {benchmark}.")
    else:
        try:
            rs = core.compute_relative_strength(core._close_series(hist), core._close_series(bench_hist))
        except ValueError as e:
            st.warning(str(e))
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"vs {bench_name}", rs["verdict"] or "n/a")
            c2.metric("Periods outperformed", rs["periods_outperformed"])
            c3.metric("Beta (1y)", rs["beta_1y"])
            c4.metric("RS trend", (rs["rs_trend"] or "n/a").title())

            rows = [{"Period": k, "Stock %": v["stock_return_pct"], "Benchmark %": v["benchmark_return_pct"],
                     "Excess %": v["excess_return_pct"]} for k, v in rs["performance"].items() if v]
            st.dataframe(pd.DataFrame(rows), hide_index=True)

            pair = pd.concat([core._close_series(hist).rename(ticker),
                              core._close_series(bench_hist).rename(bench_name)],
                             axis=1, join="inner").dropna().tail(252)
            st.caption("Last 12 months, rebased to 100")
            st.line_chart(pair / pair.iloc[0] * 100)

# ── Backtest ──────────────────────────────────────────────────────────────
with tab_bt:
    c1, c2, c3 = st.columns(3)
    strategy = c1.selectbox("Strategy", list(STRATEGY_NAMES), format_func=STRATEGY_NAMES.get)
    period = c2.selectbox("History", ["2y", "3y", "5y", "10y"], index=2)
    cost = c3.number_input("Cost per side (%)", 0.0, 2.0, 0.1, 0.05)

    fast, slow, rsi_buy, rsi_sell = 50, 200, 30.0, 70.0
    valid = True
    p1, p2, _ = st.columns(3)
    if strategy == "ma_cross":
        fast = p1.number_input("Fast MA", 5, 200, 50)
        slow = p2.number_input("Slow MA", 20, 400, 200)
        valid = fast < slow
    elif strategy == "rsi":
        rsi_buy = p1.number_input("Buy when RSI below", 5.0, 50.0, 30.0)
        rsi_sell = p2.number_input("Sell when RSI above", 50.0, 95.0, 70.0)

    if not valid:
        st.warning("Fast MA must be lower than slow MA.")
    else:
        try:
            bt_hist = load_history(ticker, period)
            res = core.run_backtest(bt_hist["Close"], strategy, fast=fast, slow=slow, rsi_buy=rsi_buy,
                                    rsi_sell=rsi_sell, cost_pct=cost, include_curve=True)
        except ValueError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"Backtest failed: {e}")
        else:
            sr, bh = res["strategy_results"], res["buy_and_hold_results"]
            keys = ["total_return_pct", "cagr_pct", "annual_volatility_pct", "sharpe_ratio", "max_drawdown_pct"]
            st.dataframe(pd.DataFrame({"Strategy": {k: sr[k] for k in keys},
                                       "Buy & hold": {k: bh[k] for k in keys}}))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Completed trades", sr["completed_trades"])
            m2.metric("Win rate %", sr["win_rate_pct"] if sr["win_rate_pct"] is not None else "n/a")
            m3.metric("Time in market %", sr["time_in_market_pct"])
            m4.metric("Beat buy & hold?", "Yes" if res["beat_buy_and_hold"] else "No")

            curve = pd.DataFrame(res["equity_curve_weekly"]).set_index("date")
            st.caption("Growth of ₹1 (weekly)")
            st.line_chart(curve)

            st.write(f"**Current state:** {res['current_signal']}")
            if res["open_position"]:
                st.info(f"Open position since {res['open_position']['entry_date']} "
                        f"({res['open_position']['unrealised_return_pct']}% unrealised)")
            if res["recent_trades"]:
                st.subheader("Recent trades")
                st.dataframe(pd.DataFrame(res["recent_trades"]), hide_index=True)
            st.caption(res["disclaimer"])

# ── Watchlist ─────────────────────────────────────────────────────────────
with tab_watch:
    raw = st.text_area("Symbols (comma-separated, max 10)", "RELIANCE, TCS, INFY, HDFCBANK, ITC")
    if st.button("Scan watchlist"):
        symbols = [s.strip() for s in raw.split(",") if s.strip()][:10]
        rows = []
        with st.spinner("Scanning..."):
            for s in symbols:
                t = to_ticker(s)
                try:
                    h = load_history(t)
                    if h.empty:
                        rows.append({"Symbol": t, "Signal": None, "Label": "No data"})
                        continue
                    tech, flag = core.get_technical_snapshot(h)
                    sig = core.compute_signal_score(tech)
                    rows.append({"Symbol": t, "Price": tech["current_price"], "Signal": sig["score"],
                                 "Label": sig["label"], "RSI": tech["rsi_14"], "MA trend": tech["ma_trend"],
                                 "% of 52w high": tech["pct_of_52w_high"], "Corp. action?": flag})
                except Exception as e:
                    rows.append({"Symbol": t, "Signal": None, "Label": f"Error: {e}"})
        st.session_state["watchlist_rows"] = rows

    if "watchlist_rows" in st.session_state:
        df = pd.DataFrame(st.session_state["watchlist_rows"]).sort_values("Signal", ascending=False, na_position="last")
        st.dataframe(df, hide_index=True)

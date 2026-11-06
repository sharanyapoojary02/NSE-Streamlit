"""
NSE Stock Analysis API — for n8n integration
================================================================================
A single FastAPI service that does ALL the real work (data pull + every
indicator built in our Jupyter session) and exposes it as one clean HTTP
endpoint. n8n calls this URL and gets back a complete JSON analysis for
ANY NSE-listed company — no coding required inside n8n itself.

REQUIREMENTS
------------
    pip install fastapi uvicorn yfinance pandas numpy

RUN LOCALLY (for testing in Jupyter/terminal)
-----------------------------------------------
    uvicorn nse_analysis_api:app --reload --port 8000

Then test in your browser or Jupyter:
    http://127.0.0.1:8000/analyze/RELIANCE

EXPOSING THIS TO n8n
---------------------
- If n8n runs on the SAME machine: use http://127.0.0.1:8000 directly (self-hosted n8n)
- If n8n is in the cloud (n8n.cloud) or on a different machine: you need a
  public URL. Easiest free options:
    1. ngrok (quick, good for testing):  ngrok http 8000
       -> gives you a temporary public URL like https://abc123.ngrok.io
    2. Deploy this file properly on Render.com or Railway.app (free tier)
       for a permanent URL — recommended once you move past testing

USAGE FROM n8n
---------------
In n8n, add an "HTTP Request" node:
    Method: GET
    URL: http://YOUR_URL/analyze/{{ $json.symbol }}
No code needed on the n8n side at all — this endpoint does everything.

ENDPOINTS
---------
    GET /                              health check
    GET /analyze/{symbol}              technicals + fundamentals + composite signal score
    GET /chart/{symbol}?months=6       OHLCV + indicator series for charting
    GET /watchlist?symbols=A,B,C       lightweight multi-stock snapshot (with signal score)
    GET /relative-strength/{symbol}    performance vs Nifty 50 (or another index), beta, RS trend
    GET /backtest/{symbol}             long-only backtest: ma_cross | macd | rsi vs buy-and-hold

All signals and backtests are screening/educational aids, not investment advice.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI(title="NSE Stock Analysis API")

# Allow any frontend (v0.dev, Lovable, or anywhere else) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── INDICATOR FUNCTIONS (same ones built in our Jupyter session today) ────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_roc(prices, period=12):
    return prices.pct_change(period) * 100

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calculate_moving_averages(prices, short_window=50, long_window=200):
    ma_short = prices.rolling(short_window).mean()
    ma_long = prices.rolling(long_window).mean()
    return ma_short, ma_long

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0; minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calculate_52w_high_proximity(prices, window=252):
    return (prices / prices.rolling(window).max()) * 100

def calculate_bollinger_bands(prices, window=20, num_std=2):
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    percent_b = (prices - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle * 100
    return percent_b, bandwidth

def get_yoy_earnings_growth(net_income, tolerance_days=45):
    """Date-matched YoY growth — the corrected version, fixing the bug
    we caught earlier today where fixed-position matching broke on
    stocks with an irregularly reported quarter."""
    if len(net_income) < 2:
        return None, None
    latest_date = net_income.index[0]
    latest_val = net_income.iloc[0]
    target_date = latest_date - pd.DateOffset(months=12)
    date_diffs = (net_income.index - target_date).days.to_series(index=net_income.index).abs()
    closest_idx = date_diffs.idxmin()
    if date_diffs[closest_idx] > tolerance_days:
        return None, None
    base_val = net_income.loc[closest_idx]
    if base_val == 0 or pd.isna(base_val):
        return None, None
    return round(((latest_val / base_val) - 1) * 100, 2), str(closest_idx.date())


def safe_round(val, decimals=2):
    if val is None or pd.isna(val):
        return None
    return round(float(val), decimals)


def get_yoy_revenue_growth(revenue, tolerance_days=45):
    """Same date-matched logic as earnings growth, applied to revenue."""
    return get_yoy_earnings_growth(revenue, tolerance_days)


def get_fundamentals(stock):
    """Extended fundamentals: earnings growth, revenue growth, margins, ROE, debt-to-equity.

    IMPORTANT: this deliberately avoids stock.info — Yahoo Finance blocks/rate-limits
    that specific endpoint for many cloud hosting providers (Render, Railway, Heroku),
    returning an almost-empty dict even though the price-history endpoint works fine.
    Instead, everything here is calculated directly from the income statement and
    balance sheet, which use the endpoint that's actually working for us."""
    fundamentals = {
        "yoy_earnings_growth_pct": None,
        "earnings_base_quarter": None,
        "yoy_revenue_growth_pct": None,
        "revenue_base_quarter": None,
        "operating_margin_pct": None,
        "net_margin_pct": None,
        "roe_pct": None,
        "debt_to_equity": None,
        "market_cap_cr": None,
    }

    latest_revenue = None
    latest_net_income = None
    latest_operating_income = None

    try:
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is not None:
            if "Net Income" in income_stmt.index:
                net_income = income_stmt.loc["Net Income"].dropna()
                growth, base_date = get_yoy_earnings_growth(net_income)
                fundamentals["yoy_earnings_growth_pct"] = growth
                fundamentals["earnings_base_quarter"] = base_date
                if len(net_income) > 0:
                    latest_net_income = net_income.iloc[0]

            if "Total Revenue" in income_stmt.index:
                revenue = income_stmt.loc["Total Revenue"].dropna()
                growth, base_date = get_yoy_revenue_growth(revenue)
                fundamentals["yoy_revenue_growth_pct"] = growth
                fundamentals["revenue_base_quarter"] = base_date
                if len(revenue) > 0:
                    latest_revenue = revenue.iloc[0]

            if "Operating Income" in income_stmt.index:
                op_income = income_stmt.loc["Operating Income"].dropna()
                if len(op_income) > 0:
                    latest_operating_income = op_income.iloc[0]

        # margins calculated directly — same math a real analyst would do by hand
        if latest_revenue and latest_revenue != 0:
            if latest_net_income is not None:
                fundamentals["net_margin_pct"] = safe_round((latest_net_income / latest_revenue) * 100)
            if latest_operating_income is not None:
                fundamentals["operating_margin_pct"] = safe_round((latest_operating_income / latest_revenue) * 100)

    except Exception as e:
        fundamentals["_debug_income_stmt_error"] = str(e)

    try:
        balance_sheet = stock.quarterly_balance_sheet
        if balance_sheet is not None:
            equity = None
            total_debt = None

            for equity_label in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if equity_label in balance_sheet.index:
                    equity_series = balance_sheet.loc[equity_label].dropna()
                    if len(equity_series) > 0:
                        equity = equity_series.iloc[0]
                        break

            for debt_label in ["Total Debt", "Net Debt"]:
                if debt_label in balance_sheet.index:
                    debt_series = balance_sheet.loc[debt_label].dropna()
                    if len(debt_series) > 0:
                        total_debt = debt_series.iloc[0]
                        break

            if equity and equity != 0:
                if total_debt is not None:
                    fundamentals["debt_to_equity"] = safe_round(total_debt / equity)
                # ROE using latest quarter's net income annualized (x4) against latest equity —
                # a simplified TTM-style approximation, standard when only one quarter is available
                if latest_net_income is not None:
                    fundamentals["roe_pct"] = safe_round(((latest_net_income * 4) / equity) * 100)

    except Exception as e:
        fundamentals["_debug_balance_sheet_error"] = str(e)

    # market cap: try the lighter fast_info endpoint, which is less likely to be blocked
    # than the full .info dict — separate try block so a failure here doesn't affect anything above
    try:
        fast_info = stock.fast_info
        if fast_info and fast_info.get("marketCap"):
            fundamentals["market_cap_cr"] = safe_round(fast_info["marketCap"] / 1e7)
    except Exception:
        pass  # market cap is a nice-to-have; not worth failing the request over

    return fundamentals


def get_technical_snapshot(hist):
    """The core technical calculation, factored out so both /analyze and /watchlist can reuse it."""
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]

    daily_pct_change = close.pct_change()
    corporate_action_flag = bool((daily_pct_change.abs() > 0.15).any())

    rsi = calculate_rsi(close)
    roc = calculate_roc(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    adx = calculate_adx(high, low, close)
    high52 = calculate_52w_high_proximity(close)
    percent_b, bandwidth = calculate_bollinger_bands(close)

    ma_trend = None
    if not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]):
        ma_trend = "bullish" if ma50.iloc[-1] > ma200.iloc[-1] else "bearish"

    technical = {
        "current_price": safe_round(close.iloc[-1]),
        "rsi_14": safe_round(rsi.iloc[-1]),
        "roc_12": safe_round(roc.iloc[-1]),
        "macd_line": safe_round(macd_line.iloc[-1]),
        "macd_signal": safe_round(signal_line.iloc[-1]),
        "macd_histogram": safe_round(macd_hist.iloc[-1]),
        "ma_50": safe_round(ma50.iloc[-1]),
        "ma_200": safe_round(ma200.iloc[-1]),
        "ma_trend": ma_trend,
        "adx_14": safe_round(adx.iloc[-1]),
        "pct_of_52w_high": safe_round(high52.iloc[-1]),
        "bollinger_percent_b": safe_round(percent_b.iloc[-1], 3),
        "bollinger_bandwidth_pct": safe_round(bandwidth.iloc[-1]),
    }
    return technical, corporate_action_flag


# ── FEATURE 1: COMPOSITE SIGNAL SCORE ─────────────────────────────────────
def compute_signal_score(technical):
    """
    Transparent rule-based score from -5 (strongly bearish) to +5 (strongly bullish).
    Five components, each worth -1 / 0 / +1, and every point is explained in `components`
    so the user can see exactly WHY the score is what it is.

    NOTE: RSI and Bollinger are mean-reversion signals (oversold = +1), while MACD,
    MA trend and ADX are trend/momentum signals. They can disagree, which is why the
    components are listed rather than hidden inside one number.
    """
    components = []
    score = 0

    def add(name, points, reason):
        nonlocal score
        score += points
        components.append({"indicator": name, "points": points, "reason": reason})

    # 1. RSI (mean-reversion)
    rsi = technical.get("rsi_14")
    if rsi is None:
        add("RSI", 0, "RSI unavailable (insufficient data)")
    elif rsi < 30:
        add("RSI", 1, f"RSI {rsi} < 30: oversold")
    elif rsi > 70:
        add("RSI", -1, f"RSI {rsi} > 70: overbought")
    else:
        add("RSI", 0, f"RSI {rsi} is in the neutral 30-70 zone")

    # 2. MACD histogram (momentum)
    hist_val = technical.get("macd_histogram")
    if hist_val is None:
        add("MACD", 0, "MACD unavailable")
    elif hist_val > 0:
        add("MACD", 1, "MACD line is above its signal line (positive momentum)")
    elif hist_val < 0:
        add("MACD", -1, "MACD line is below its signal line (negative momentum)")
    else:
        add("MACD", 0, "MACD line equals its signal line")

    # 3. Long-term trend: 50-day vs 200-day MA
    ma_trend = technical.get("ma_trend")
    if ma_trend == "bullish":
        add("MA trend", 1, "50-day MA is above 200-day MA")
    elif ma_trend == "bearish":
        add("MA trend", -1, "50-day MA is below 200-day MA")
    else:
        add("MA trend", 0, "MA trend unavailable (needs 200 trading days of history)")

    # 4. ADX: only counts when the trend is strong enough (>= 25), and in the MA trend's direction
    adx = technical.get("adx_14")
    if adx is None or ma_trend is None:
        add("ADX", 0, "ADX or MA trend unavailable")
    elif adx >= 25:
        pts = 1 if ma_trend == "bullish" else -1
        add("ADX", pts, f"ADX {adx} >= 25: strong {ma_trend} trend")
    else:
        add("ADX", 0, f"ADX {adx} < 25: weak trend, trend signals are less reliable")

    # 5. Bollinger %B (mean-reversion): below 0 = under lower band, above 1 = over upper band
    pb = technical.get("bollinger_percent_b")
    if pb is None:
        add("Bollinger", 0, "Bollinger %B unavailable")
    elif pb < 0:
        add("Bollinger", 1, f"%B {pb} < 0: price is below the lower band")
    elif pb > 1:
        add("Bollinger", -1, f"%B {pb} > 1: price is above the upper band")
    else:
        add("Bollinger", 0, f"%B {pb} is inside the bands")

    if score >= 3:
        label = "Strong Bullish"
    elif score >= 1:
        label = "Bullish"
    elif score == 0:
        label = "Neutral"
    elif score >= -2:
        label = "Bearish"
    else:
        label = "Strong Bearish"

    return {
        "score": score,
        "min_score": -5,
        "max_score": 5,
        "label": label,
        "components": components,
        "note": "Rule-based screening aid, not investment advice.",
    }


@app.get("/")
def health_check():
    return {"status": "ok", "message": "NSE Stock Analysis API is running"}


@app.get("/analyze/{symbol}")
def analyze_stock(symbol: str):
    """Full technical + fundamental analysis for one NSE-listed company."""
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", auto_adjust=True)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")

        technical, corporate_action_flag = get_technical_snapshot(hist)
        fundamentals = get_fundamentals(stock)

        return {
            "symbol": ticker,
            "company_name": ticker,
            "timestamp": datetime.now().isoformat(),
            "corporate_action_flag": corporate_action_flag,
            "corporate_action_note": (
                "A >15% single-day price move was detected in the last 2 years — "
                "moving-average and Bollinger figures may be distorted by a demerger, "
                "split, or similar event. Verify before trusting the technical signals."
                if corporate_action_flag else None
            ),
            "technical": technical,
            "signal": compute_signal_score(technical),
            "fundamentals": fundamentals,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing {ticker}: {str(e)}")


@app.get("/chart/{symbol}")
def chart_data(symbol: str, months: int = 6):
    """
    OHLC candlestick data + overlay indicator series for charting.
    Returns one row per trading day: date, OHLCV, RSI, MACD line/signal, MA50, MA200.
    `months` controls how much history to return in the chart (default 6, max 24).
    """
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    months = min(max(months, 1), 24)

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", auto_adjust=True)  # need 2y so MA200 has enough lookback, even though we only return the recent slice
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

        close = hist["Close"]
        rsi = calculate_rsi(close)
        macd_line, signal_line, _ = calculate_macd(close)
        ma50, ma200 = calculate_moving_averages(close)

        trading_days = months * 21  # approx trading days per month
        recent = hist.tail(trading_days)

        rows = []
        for date, row in recent.iterrows():
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": safe_round(row["Open"]),
                "high": safe_round(row["High"]),
                "low": safe_round(row["Low"]),
                "close": safe_round(row["Close"]),
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else None,
                "rsi": safe_round(rsi.loc[date]) if date in rsi.index else None,
                "macd_line": safe_round(macd_line.loc[date]) if date in macd_line.index else None,
                "macd_signal": safe_round(signal_line.loc[date]) if date in signal_line.index else None,
                "ma_50": safe_round(ma50.loc[date]) if date in ma50.index else None,
                "ma_200": safe_round(ma200.loc[date]) if date in ma200.index else None,
            })

        return {"symbol": ticker, "months": months, "data": rows}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data for {ticker}: {str(e)}")


@app.get("/watchlist")
def watchlist(symbols: str):
    """
    Lightweight multi-stock snapshot for a watchlist view.
    Pass symbols comma-separated: /watchlist?symbols=RELIANCE,TCS,INFY
    Capped at 10 symbols per request to keep response time reasonable on free hosting.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="Provide at least one symbol, e.g. ?symbols=RELIANCE,TCS")

    results = []
    for symbol in symbol_list:
        ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y", auto_adjust=True)
            if hist.empty:
                results.append({"symbol": ticker, "error": "No data found"})
                continue

            technical, corporate_action_flag = get_technical_snapshot(hist)
            signal = compute_signal_score(technical)
            results.append({
                "symbol": ticker,
                "company_name": ticker,
                "signal_score": signal["score"],
                "signal_label": signal["label"],
                "current_price": technical["current_price"],
                "rsi_14": technical["rsi_14"],
                "macd_histogram": technical["macd_histogram"],
                "ma_trend": technical["ma_trend"],
                "pct_of_52w_high": technical["pct_of_52w_high"],
                "corporate_action_flag": corporate_action_flag,
            })
        except Exception as e:
            results.append({"symbol": ticker, "error": str(e)})

    return {"count": len(results), "stocks": results}


# ── FEATURE 2: RELATIVE STRENGTH vs BENCHMARK INDEX ───────────────────────
DEFAULT_BENCHMARK = "^NSEI"  # Nifty 50. Others on Yahoo: ^NSEBANK (Nifty Bank), ^CNXIT (Nifty IT)


def _close_series(hist):
    """Close prices with a timezone-free, date-normalised index so two tickers can be aligned."""
    s = hist["Close"].copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep="last")]


def compute_relative_strength(stock_close, bench_close):
    """Pure function (no network) so it can be unit-tested. Inputs: aligned-or-not close series."""
    df = pd.concat([stock_close.rename("stock"), bench_close.rename("bench")], axis=1, join="inner").dropna()
    if len(df) < 64:
        raise ValueError("Not enough overlapping history (need at least ~3 months of common trading days).")

    periods = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
    perf = {}
    outperformed = 0
    available = 0
    for label, n in periods.items():
        if len(df) > n:
            s_ret = df["stock"].iloc[-1] / df["stock"].iloc[-1 - n] - 1
            b_ret = df["bench"].iloc[-1] / df["bench"].iloc[-1 - n] - 1
            excess = s_ret - b_ret
            perf[label] = {
                "stock_return_pct": safe_round(s_ret * 100),
                "benchmark_return_pct": safe_round(b_ret * 100),
                "excess_return_pct": safe_round(excess * 100),
                "outperformed": bool(excess > 0),
            }
            available += 1
            outperformed += int(excess > 0)
        else:
            perf[label] = None

    # Beta and correlation on up to the last 252 daily returns
    rets = df.pct_change().dropna().tail(252)
    bench_var = rets["bench"].var()
    beta = rets["stock"].cov(rets["bench"]) / bench_var if bench_var and bench_var > 0 else None
    corr = rets["stock"].corr(rets["bench"])

    # RS ratio line: stock / benchmark, rebased to 100. Rising line = outperforming.
    ratio = (df["stock"] / df["bench"])
    ratio = ratio / ratio.iloc[0] * 100
    ratio_ma50 = ratio.rolling(50).mean()
    rs_trend = None
    if not pd.isna(ratio_ma50.iloc[-1]):
        rs_trend = "improving" if ratio.iloc[-1] > ratio_ma50.iloc[-1] else "weakening"

    if available == 0:
        verdict = None
    elif outperformed / available >= 0.75:
        verdict = "Outperforming"
    elif outperformed / available <= 0.25:
        verdict = "Underperforming"
    else:
        verdict = "Mixed"

    return {
        "verdict": verdict,
        "periods_outperformed": f"{outperformed}/{available}",
        "performance": perf,
        "beta_1y": safe_round(beta),
        "correlation_1y": safe_round(corr),
        "rs_ratio_latest": safe_round(ratio.iloc[-1]),
        "rs_ratio_ma50": safe_round(ratio_ma50.iloc[-1]),
        "rs_trend": rs_trend,
        "common_trading_days": int(len(df)),
        "as_of": str(df.index[-1].date()),
    }


@app.get("/relative-strength/{symbol}")
def relative_strength(symbol: str, benchmark: str = DEFAULT_BENCHMARK):
    """
    How a stock has performed relative to a benchmark index (default Nifty 50).
    Returns excess return over 1/3/6/12 months, 1-year beta and correlation, and whether the
    stock/benchmark ratio line is above its own 50-day average (improving relative trend).
    Try ?benchmark=^NSEBANK for banks or ?benchmark=^CNXIT for IT stocks.
    """
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    benchmark = benchmark.strip()

    try:
        stock_hist = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
        if stock_hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")
        bench_hist = yf.Ticker(benchmark).history(period="2y", auto_adjust=True)
        if bench_hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for benchmark {benchmark}.")

        result = compute_relative_strength(_close_series(stock_hist), _close_series(bench_hist))
        return {"symbol": ticker, "benchmark": benchmark, **result}

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing relative strength for {ticker}: {str(e)}")


# ── FEATURE 3: BACKTEST ENDPOINT ──────────────────────────────────────────
BACKTEST_STRATEGIES = {"ma_cross", "macd", "rsi"}
BACKTEST_PERIODS = {"2y", "3y", "5y", "10y"}


def build_positions(close, strategy, fast=50, slow=200, rsi_buy=30.0, rsi_sell=70.0):
    """Return (position series of 0/1 decided at each day's close, warmup_bars)."""
    if strategy == "ma_cross":
        ma_f = close.rolling(fast).mean()
        ma_s = close.rolling(slow).mean()
        pos = ((ma_f > ma_s) & ma_s.notna()).astype(float)
        warmup = slow
    elif strategy == "macd":
        macd_line, signal_line, _ = calculate_macd(close)
        pos = (macd_line > signal_line).astype(float)
        warmup = 35  # slow EMA (26) + signal (9)
    elif strategy == "rsi":
        rsi = calculate_rsi(close)
        state = 0.0
        vals = []
        for v in rsi:
            if pd.isna(v):
                vals.append(0.0)
                continue
            if state == 0.0 and v < rsi_buy:
                state = 1.0           # enter when oversold
            elif state == 1.0 and v > rsi_sell:
                state = 0.0           # exit when overbought
            vals.append(state)
        pos = pd.Series(vals, index=close.index, dtype=float)
        warmup = 15
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    return pos, warmup


def _perf_metrics(daily_ret, risk_free_pct=0.0):
    """Standard performance stats from a daily-return series."""
    equity = (1 + daily_ret).cumprod()
    years = max((daily_ret.index[-1] - daily_ret.index[0]).days / 365.25, 1e-9)
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1 if equity.iloc[-1] > 0 else -1.0
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = ((daily_ret.mean() * 252) - risk_free_pct / 100) / vol if vol and vol > 0 else None
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return_pct": safe_round(total * 100),
        "cagr_pct": safe_round(cagr * 100),
        "annual_volatility_pct": safe_round(vol * 100),
        "sharpe_ratio": safe_round(sharpe),
        "max_drawdown_pct": safe_round(drawdown.min() * 100),
    }, equity


def run_backtest(close, strategy, fast=50, slow=200, rsi_buy=30.0, rsi_sell=70.0,
                 cost_pct=0.1, risk_free_pct=0.0, include_curve=False):
    """
    Long-only, single-stock backtest on daily closes. Pure function (no network).

    Assumptions (stated openly so nobody over-trusts the output):
      * Signal is computed at day t's close and traded at that same close; the position earns
        returns from day t+1 onward (so there is no look-ahead bias).
      * `cost_pct` is charged per side (buy and sell) to approximate brokerage, STT and slippage.
      * Prices are adjusted for splits and dividends (auto_adjust=True).
      * Indicator warm-up days are excluded from BOTH the strategy and buy-and-hold figures.
    """
    close = close.dropna()
    pos, warmup = build_positions(close, strategy, fast, slow, rsi_buy, rsi_sell)
    if len(close) <= warmup + 30:
        raise ValueError("Not enough history for this strategy; try a longer period.")

    ret = close.pct_change().fillna(0.0)
    held = pos.shift(1).fillna(0.0)                    # position actually held during day t
    switches = held.diff().abs().fillna(held.abs())     # 1 on any entry/exit day
    cost = cost_pct / 100
    strat_ret = held * ret - switches * cost

    # drop warm-up so strategy and buy-and-hold are measured over the same window
    ret_eval = ret.iloc[warmup:]
    strat_eval = strat_ret.iloc[warmup:]
    held_eval = held.iloc[warmup:]

    strat_stats, strat_equity = _perf_metrics(strat_eval, risk_free_pct)
    bh_stats, bh_equity = _perf_metrics(ret_eval, risk_free_pct)

    # Build the trade list from position changes (entry/exit price = close of the signal day)
    dates = close.index
    prices = close.values
    start_i = warmup
    trades = []
    entry_i = None
    held_vals = held.values
    for i in range(start_i, len(held_vals)):
        if held_vals[i] == 1.0 and entry_i is None:
            entry_i = i
        elif held_vals[i] == 0.0 and entry_i is not None:
            e, x = entry_i - 1, i - 1               # signal days
            gross = prices[x] / prices[e] - 1
            net = (1 + gross) * (1 - cost) ** 2 - 1
            trades.append({
                "entry_date": str(dates[e].date()), "entry_price": safe_round(prices[e]),
                "exit_date": str(dates[x].date()), "exit_price": safe_round(prices[x]),
                "return_pct": safe_round(net * 100), "holding_days": int(i - entry_i),
            })
            entry_i = None

    open_position = None
    if entry_i is not None:
        e = entry_i - 1
        gross = prices[-1] / prices[e] - 1
        open_position = {
            "entry_date": str(dates[e].date()), "entry_price": safe_round(prices[e]),
            "latest_price": safe_round(prices[-1]),
            "unrealised_return_pct": safe_round(((1 + gross) * (1 - cost) - 1) * 100),
        }

    wins = [t for t in trades if t["return_pct"] is not None and t["return_pct"] > 0]
    win_rate = round(len(wins) / len(trades) * 100, 1) if trades else None
    avg_trade = safe_round(np.mean([t["return_pct"] for t in trades])) if trades else None

    # Today's signal state (decided at the latest close)
    current_signal = "in position (long)" if pos.iloc[-1] == 1.0 else "flat (no position)"

    result = {
        "strategy": strategy,
        "parameters": (
            {"fast_ma": fast, "slow_ma": slow} if strategy == "ma_cross"
            else {"rsi_buy_below": rsi_buy, "rsi_sell_above": rsi_sell} if strategy == "rsi"
            else {"macd": "12/26/9"}
        ),
        "cost_per_side_pct": cost_pct,
        "test_window": {
            "start": str(held_eval.index[0].date()),
            "end": str(held_eval.index[-1].date()),
            "trading_days": int(len(held_eval)),
        },
        "strategy_results": {
            **strat_stats,
            "time_in_market_pct": safe_round(held_eval.mean() * 100),
            "completed_trades": len(trades),
            "win_rate_pct": win_rate,
            "avg_trade_return_pct": avg_trade,
        },
        "buy_and_hold_results": bh_stats,
        "beat_buy_and_hold": bool(strat_stats["total_return_pct"] is not None
                                  and bh_stats["total_return_pct"] is not None
                                  and strat_stats["total_return_pct"] > bh_stats["total_return_pct"]),
        "current_signal": current_signal,
        "open_position": open_position,
        "recent_trades": trades[-10:],
        "disclaimer": (
            "Historical simulation on a single stock; past performance does not predict future results. "
            "Educational use only, not investment advice."
        ),
    }

    if include_curve:
        weekly_s = strat_equity.resample("W").last().dropna()
        weekly_b = bh_equity.resample("W").last().dropna()
        result["equity_curve_weekly"] = [
            {"date": str(d.date()), "strategy": safe_round(weekly_s.loc[d], 4),
             "buy_and_hold": safe_round(weekly_b.loc[d], 4) if d in weekly_b.index else None}
            for d in weekly_s.index
        ]
    return result


@app.get("/backtest/{symbol}")
def backtest(
    symbol: str,
    strategy: str = "ma_cross",
    period: str = "5y",
    fast: int = 50,
    slow: int = 200,
    rsi_buy: float = 30.0,
    rsi_sell: float = 70.0,
    cost_pct: float = 0.1,
    risk_free_pct: float = 0.0,
    include_curve: bool = False,
):
    """
    Backtest a simple long-only strategy and compare it with buy-and-hold.
      strategy = ma_cross  -> hold while the `fast` MA is above the `slow` MA (default 50/200)
      strategy = macd      -> hold while MACD line is above its signal line
      strategy = rsi       -> buy when RSI < rsi_buy, sell when RSI > rsi_sell
    Example: /backtest/RELIANCE?strategy=rsi&period=5y&rsi_buy=30&rsi_sell=60
    Set include_curve=true to get a weekly equity curve for charting.
    """
    strategy = strategy.lower().strip()
    if strategy not in BACKTEST_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"strategy must be one of {sorted(BACKTEST_STRATEGIES)}")
    if period not in BACKTEST_PERIODS:
        raise HTTPException(status_code=400, detail=f"period must be one of {sorted(BACKTEST_PERIODS)}")
    if strategy == "ma_cross" and not (1 <= fast < slow <= 400):
        raise HTTPException(status_code=400, detail="For ma_cross, need 1 <= fast < slow <= 400.")
    if strategy == "rsi" and not (0 < rsi_buy < rsi_sell < 100):
        raise HTTPException(status_code=400, detail="For rsi, need 0 < rsi_buy < rsi_sell < 100.")
    if not (0 <= cost_pct <= 2):
        raise HTTPException(status_code=400, detail="cost_pct must be between 0 and 2 (percent per side).")

    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"

    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

        result = run_backtest(
            hist["Close"], strategy, fast=fast, slow=slow, rsi_buy=rsi_buy, rsi_sell=rsi_sell,
            cost_pct=cost_pct, risk_free_pct=risk_free_pct, include_curve=include_curve,
        )
        return {"symbol": ticker, "period": period, **result}

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error backtesting {ticker}: {str(e)}")

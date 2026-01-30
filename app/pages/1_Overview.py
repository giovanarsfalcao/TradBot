"""
Overview - Landing Page

At-a-glance view: Portfolio KPIs, equity curve, allocation, and active signals.
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import yfinance_fix

from tradbot.strategy import TechnicalIndicators
from tradbot.risk.metrics import (
    calculate_sharpe_ratio, calculate_max_drawdown, calculate_var,
    calculate_annualized_return, calculate_annualized_volatility,
)
from tradbot.signals.signal_rsi_macd import generate_signal as rsi_signal
from tradbot.signals.signal_logreg import generate_signal as logreg_signal
from components.charts import equity_curve, drawdown_chart, COLORS
from components.kpi_cards import render_kpi_row, render_signal_badge


st.header("Overview")

# --- Sidebar ---
with st.sidebar:
    st.subheader("Settings")
    ticker = st.text_input("Ticker", value="SPY", key="overview_ticker")
    period = st.selectbox("Period", ["1y", "2y", "5y", "max"], index=2, key="overview_period")
    portfolio_tickers = st.text_input(
        "Portfolio Tickers (comma separated)",
        value="AAPL, MSFT, GOOGL, AMZN, META",
        key="overview_portfolio",
    )

# --- Load Data ---
@st.cache_data(ttl=300)
def load_ticker_data(symbol, period):
    df = yf.download(symbol, period=period, session=yfinance_fix.chrome_session, progress=False)
    if df.empty:
        return None
    df.columns = df.columns.get_level_values(0)
    return df

with st.spinner("Loading data..."):
    df = load_ticker_data(ticker, period)

if df is None:
    st.error(f"Could not load data for {ticker}")
    st.stop()

# --- Calculate Metrics ---
returns = df["Close"].pct_change().dropna()
prices = df["Close"]
equity = (1 + returns).cumprod()

sharpe = calculate_sharpe_ratio(returns)
max_dd = calculate_max_drawdown(equity)
var_95 = calculate_var(returns, 0.95)
ann_ret = calculate_annualized_return(returns)
ann_vol = calculate_annualized_volatility(returns)

# --- KPI Row ---
daily_return = float(returns.iloc[-1]) if len(returns) > 0 else 0.0

render_kpi_row([
    {"label": "Last Close", "value": f"${float(df['Close'].iloc[-1]):.2f}",
     "delta": f"{daily_return:.2%}"},
    {"label": "Sharpe Ratio", "value": f"{sharpe:.2f}"},
    {"label": "Max Drawdown", "value": f"{max_dd:.2%}"},
    {"label": "VaR 95%", "value": f"{var_95:.4f}"},
    {"label": "Ann. Return", "value": f"{ann_ret:.2%}"},
])

st.divider()

# --- Equity Curve + Drawdown ---
col1, col2 = st.columns([3, 2])

with col1:
    cum_returns = (1 + returns).cumprod() - 1
    eq_df = pd.DataFrame({"cumulative_return": cum_returns})
    fig = equity_curve(eq_df, {"cumulative_return": f"{ticker} Equity"}, title="Equity Curve")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig_dd = drawdown_chart(equity, title="Drawdown")
    st.plotly_chart(fig_dd, use_container_width=True)

st.divider()

# --- Active Signals ---
st.subheader("Active Signals")

try:
    sig_rsi = rsi_signal(df)
    sig_log = logreg_signal(df)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**RSI + MACD**")
        render_signal_badge(sig_rsi["signal"], sig_rsi.get("reason", ""))

    with c2:
        st.markdown("**Logistic Regression**")
        render_signal_badge(sig_log["signal"], sig_log.get("reason", ""))
        if sig_log.get("probability") is not None:
            st.caption(f"P(Up) = {sig_log['probability']:.3f} | AUC = {sig_log.get('auc', 'N/A')}")

except Exception as e:
    st.warning(f"Could not generate signals: {e}")

st.divider()

# --- Quick Stats Table ---
st.subheader("Summary Statistics")
stats = pd.DataFrame({
    "Metric": ["Total Return", "Ann. Return", "Ann. Volatility", "Sharpe Ratio",
               "Max Drawdown", "VaR 95%", "Data Points"],
    "Value": [
        f"{float(equity.iloc[-1] - 1):.2%}",
        f"{ann_ret:.2%}",
        f"{ann_vol:.2%}",
        f"{sharpe:.2f}",
        f"{max_dd:.2%}",
        f"{var_95:.4f}",
        f"{len(df):,}",
    ]
})
st.dataframe(stats, use_container_width=True, hide_index=True)

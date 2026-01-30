"""
Risk Management - VaR, Sharpe, Drawdown, Volatility Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import yfinance_fix

from tradbot.risk.metrics import (
    calculate_sharpe_ratio, calculate_max_drawdown, calculate_var,
    calculate_annualized_return, calculate_annualized_volatility,
)
from components.charts import drawdown_chart, return_distribution, rolling_volatility
from components.kpi_cards import render_kpi_row


st.header("Risk Management")

# --- Sidebar ---
with st.sidebar:
    st.subheader("Settings")
    ticker = st.text_input("Ticker", value="SPY", key="risk_ticker")
    period = st.selectbox("Period", ["1y", "2y", "5y", "max"], index=2, key="risk_period")
    vol_window = st.slider("Volatility Window (days)", 10, 90, 30, key="risk_vol_window")

# --- Load Data ---
@st.cache_data(ttl=300)
def load_data(symbol, period):
    df = yf.download(symbol, period=period, session=yfinance_fix.chrome_session, progress=False)
    if df.empty:
        return None
    df.columns = df.columns.get_level_values(0)
    return df

with st.spinner("Loading data..."):
    df = load_data(ticker, period)

if df is None:
    st.error(f"Could not load data for {ticker}")
    st.stop()

returns = df["Close"].pct_change().dropna()
prices = df["Close"]
equity = (1 + returns).cumprod()

# --- Metrics ---
sharpe = calculate_sharpe_ratio(returns)
max_dd = calculate_max_drawdown(equity)
var_95 = calculate_var(returns, 0.95)
var_99 = calculate_var(returns, 0.99)
ann_ret = calculate_annualized_return(returns)
ann_vol = calculate_annualized_volatility(returns)

# --- KPI Row ---
render_kpi_row([
    {"label": "Sharpe Ratio", "value": f"{sharpe:.3f}"},
    {"label": "Max Drawdown", "value": f"{max_dd:.2%}"},
    {"label": "VaR 95%", "value": f"{var_95:.4f}"},
    {"label": "VaR 99%", "value": f"{var_99:.4f}"},
    {"label": "Ann. Return", "value": f"{ann_ret:.2%}"},
    {"label": "Ann. Volatility", "value": f"{ann_vol:.2%}"},
])

st.divider()

# --- Drawdown Chart ---
st.subheader("Drawdown")
fig_dd = drawdown_chart(equity, title=f"Drawdown - {ticker}")
st.plotly_chart(fig_dd, use_container_width=True)

st.divider()

# --- Return Distribution + Rolling Volatility ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Return Distribution")
    fig_dist = return_distribution(returns, var_95=var_95, var_99=var_99)
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    st.subheader("Rolling Volatility")
    fig_vol = rolling_volatility(returns, window=vol_window)
    st.plotly_chart(fig_vol, use_container_width=True)

st.divider()

# --- Detailed Stats Table ---
st.subheader("Detailed Statistics")
stats = pd.DataFrame({
    "Metric": [
        "Total Return", "Annualized Return", "Annualized Volatility",
        "Sharpe Ratio", "Max Drawdown",
        "VaR 95%", "VaR 99%",
        "Skewness", "Kurtosis",
        "Best Day", "Worst Day",
        "Positive Days", "Negative Days",
        "Data Points",
    ],
    "Value": [
        f"{float(equity.iloc[-1] - 1):.2%}",
        f"{ann_ret:.2%}",
        f"{ann_vol:.2%}",
        f"{sharpe:.3f}",
        f"{max_dd:.2%}",
        f"{var_95:.4f}",
        f"{var_99:.4f}",
        f"{returns.skew():.3f}",
        f"{returns.kurtosis():.3f}",
        f"{returns.max():.2%}",
        f"{returns.min():.2%}",
        f"{(returns > 0).sum()} ({(returns > 0).mean():.1%})",
        f"{(returns < 0).sum()} ({(returns < 0).mean():.1%})",
        f"{len(df):,}",
    ]
})
st.dataframe(stats, use_container_width=True, hide_index=True)

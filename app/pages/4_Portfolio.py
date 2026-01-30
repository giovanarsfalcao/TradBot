"""
Portfolio Optimization - Markowitz MPT: Max Sharpe, Min Volatility, Efficient Frontier
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import yfinance_fix

from tradbot.portfolio import optimize_max_sharpe, optimize_min_volatility, get_efficient_frontier
from components.charts import weight_bar_chart, efficient_frontier, allocation_donut, COLORS
from components.kpi_cards import render_kpi_row


st.header("Portfolio Optimization")

# --- Sidebar ---
with st.sidebar:
    st.subheader("Settings")
    portfolio_tickers = st.text_input(
        "Tickers (comma separated)",
        value="AAPL, MSFT, GOOGL, AMZN, META",
        key="port_tickers",
    )
    risk_free_rate = st.number_input(
        "Risk-Free Rate", value=0.02, step=0.005, format="%.3f", key="port_rfr"
    )
    n_frontier_points = st.slider("Frontier Points", 10, 100, 50, key="port_frontier")

tickers = [t.strip() for t in portfolio_tickers.split(",") if t.strip()]

if len(tickers) < 2:
    st.warning("Enter at least 2 tickers separated by commas")
    st.stop()

# --- Load Data ---
@st.cache_data(ttl=300)
def load_portfolio_data(tickers_list, period="2y"):
    prices = yf.download(
        tickers_list, period=period,
        session=yfinance_fix.chrome_session, progress=False
    )["Close"]
    return prices

with st.spinner("Loading portfolio data..."):
    prices = load_portfolio_data(tickers, "2y")

if prices.empty:
    st.error("Could not load portfolio data")
    st.stop()

# --- Optimize ---
try:
    weights_sharpe, perf_sharpe = optimize_max_sharpe(prices, risk_free_rate)
    weights_minvol, perf_minvol = optimize_min_volatility(prices, risk_free_rate)
except Exception as e:
    st.error(f"Optimization failed: {e}")
    st.stop()

# --- Side by Side Portfolios ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Max Sharpe Portfolio")
    render_kpi_row([
        {"label": "Return", "value": f"{perf_sharpe[0]:.2%}"},
        {"label": "Volatility", "value": f"{perf_sharpe[1]:.2%}"},
        {"label": "Sharpe", "value": f"{perf_sharpe[2]:.2f}"},
    ])
    fig = weight_bar_chart(weights_sharpe, "Weights - Max Sharpe", COLORS["blue"])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Min Volatility Portfolio")
    render_kpi_row([
        {"label": "Return", "value": f"{perf_minvol[0]:.2%}"},
        {"label": "Volatility", "value": f"{perf_minvol[1]:.2%}"},
        {"label": "Sharpe", "value": f"{perf_minvol[2]:.2f}"},
    ])
    fig = weight_bar_chart(weights_minvol, "Weights - Min Volatility", COLORS["green"])
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Allocation Donuts ---
col3, col4 = st.columns(2)
with col3:
    fig = allocation_donut(weights_sharpe, "Max Sharpe Allocation")
    st.plotly_chart(fig, use_container_width=True)
with col4:
    fig = allocation_donut(weights_minvol, "Min Volatility Allocation")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Efficient Frontier ---
st.subheader("Efficient Frontier")

with st.spinner("Computing efficient frontier..."):
    try:
        frontier = get_efficient_frontier(prices, risk_free_rate, n_frontier_points)
        fig = efficient_frontier(frontier, perf_sharpe, perf_minvol)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not compute efficient frontier: {e}")

st.divider()

# --- Weight Comparison Table ---
st.subheader("Weight Comparison")
comparison = pd.DataFrame({
    "Ticker": list(weights_sharpe.keys()),
    "Max Sharpe": [f"{v:.1%}" for v in weights_sharpe.values()],
    "Min Volatility": [f"{weights_minvol.get(k, 0):.1%}" for k in weights_sharpe.keys()],
})
st.dataframe(comparison, use_container_width=True, hide_index=True)

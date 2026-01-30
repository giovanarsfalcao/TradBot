"""
Backtesting - Strategy vs Buy & Hold comparison
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import yfinance_fix

from tradbot.signals.signal_rsi_macd import generate_strategy as rsi_strategy
from tradbot.signals.signal_logreg import generate_strategy as logreg_strategy
from tradbot.backtest import Backtest
from components.charts import equity_curve, signal_distribution_chart, COLORS
from components.kpi_cards import render_metric_table


st.header("Backtesting")

# --- Sidebar ---
with st.sidebar:
    st.subheader("Settings")
    ticker = st.text_input("Ticker", value="SPY", key="bt_ticker")
    period = st.selectbox("Period", ["1y", "2y", "5y", "max"], index=2, key="bt_period")
    strategy_name = st.selectbox(
        "Strategy", ["RSI + MACD", "Logistic Regression"], key="bt_strategy"
    )

    if strategy_name == "Logistic Regression":
        st.divider()
        st.subheader("LogReg Parameters")
        logreg_shift = st.slider("Shift (days)", 1, 20, 5, key="bt_lr_shift")
        logreg_threshold = st.slider("Threshold", 0.50, 0.70, 0.55, 0.01, key="bt_lr_thresh")

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

# --- Generate Strategy ---
with st.spinner(f"Running {strategy_name} strategy..."):
    try:
        if strategy_name == "RSI + MACD":
            df_strat = rsi_strategy(df)
        else:
            df_strat = logreg_strategy(df, shift=logreg_shift, threshold=logreg_threshold)
    except Exception as e:
        st.error(f"Strategy generation failed: {e}")
        st.stop()

# --- Run Backtest ---
with st.spinner("Running backtest..."):
    bt = Backtest(df_strat)
    bt.run()
    summary = bt.summary()

st.success(f"Backtest complete: {len(df_strat)} data points")

# --- Metrics Table ---
st.subheader("Performance Comparison")
render_metric_table(summary["strategy"], summary["buyhold"])

st.divider()

# --- Equity Curve ---
st.subheader("Equity Curve")
result_df = bt.get_df()

fig = equity_curve(
    result_df,
    {
        "asset_cum_returns": "Buy & Hold",
        "strategy_cum_returns": strategy_name,
    },
    title=f"{strategy_name} vs Buy & Hold ({ticker})",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Signal Distribution ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Signal Distribution")
    fig_dist = signal_distribution_chart(result_df["Strategy"])
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    st.subheader("Signal Summary")
    counts = result_df["Strategy"].value_counts().sort_index()
    total = len(result_df)
    signal_data = []
    label_map = {-1.0: "Short", 0.0: "Flat", 1.0: "Long"}
    for val, count in counts.items():
        signal_data.append({
            "Position": label_map.get(val, str(val)),
            "Count": count,
            "Percentage": f"{count / total * 100:.1f}%",
        })
    st.dataframe(pd.DataFrame(signal_data), use_container_width=True, hide_index=True)

    # Strategy activity rate
    active = (result_df["Strategy"] != 0).sum()
    st.metric("Active Rate", f"{active / total:.1%}",
              help="Percentage of time the strategy has an active position")

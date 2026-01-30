"""
Strategy & Signals - Technical Indicators, Regression Models, Signal Generation

Chronological pipeline: Data -> Indicators -> Models -> Signals
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import yfinance_fix

from tradbot.strategy import TechnicalIndicators
from tradbot.strategy.models import LinearRegression, LogisticRegression
from tradbot.signals.signal_rsi_macd import generate_signal as rsi_signal
from tradbot.signals.signal_logreg import generate_signal as logreg_signal
from components.charts import (
    indicator_subplot, roc_curve_chart, confusion_matrix_chart, CHART_LAYOUT, COLORS,
)
from components.kpi_cards import render_signal_badge

import plotly.graph_objects as go


st.header("Strategy & Signals")

# --- Sidebar ---
with st.sidebar:
    st.subheader("Data Settings")
    ticker = st.text_input("Ticker", value="SPY", key="strat_ticker")
    interval = st.selectbox("Interval", ["1d", "1h", "15m", "5m"], index=0, key="strat_interval")
    lookback = st.number_input("Lookback (rows)", value=5000, step=100, key="strat_lookback")

    st.divider()
    st.subheader("Indicator Parameters")

    with st.expander("MACD"):
        macd_fast = st.number_input("Fast", 12, key="macd_fast")
        macd_slow = st.number_input("Slow", 27, key="macd_slow")
        macd_span = st.number_input("Signal", 9, key="macd_span")

    with st.expander("RSI"):
        rsi_length = st.number_input("RSI Length", 14, key="rsi_len")

    with st.expander("MFI"):
        mfi_length = st.number_input("MFI Length", 14, key="mfi_len")

    with st.expander("Bollinger Bands"):
        bb_length = st.number_input("BB Length", 20, key="bb_len")
        bb_std = st.number_input("Std Dev", 2, key="bb_std")

    features = st.multiselect(
        "Features",
        ["MACD_HIST", "MFI", "BB", "RSI"],
        default=["MACD_HIST", "MFI", "BB", "RSI"],
        key="strat_features",
    )


# --- Load Data ---
@st.cache_data(ttl=300)
def load_data(symbol, interval, lookback):
    period = "730d" if interval == "1h" else "max"
    df = yf.download(symbol, session=yfinance_fix.chrome_session,
                     interval=interval, period=period, progress=False)
    if df.empty:
        return None
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df.iloc[-lookback:, :].copy()


with st.spinner("Loading data..."):
    df_raw = load_data(ticker, interval, lookback)

if df_raw is None:
    st.error("Could not load data")
    st.stop()

# Add indicators
ti = TechnicalIndicators(df_raw)
ti.add_macd(macd_fast, macd_slow, macd_span)
ti.add_mfi(mfi_length)
ti.add_bb(bb_length, bb_std)
ti.add_rsi(rsi_length)
df = ti.dropna().get_df()

st.success(f"{len(df)} data points loaded for {ticker}")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Technical Indicators", "Linear Regression",
    "Logistic Regression", "Signal Dashboard"
])

# === Tab 1: Technical Indicators ===
with tab1:
    for ind in ["MACD", "RSI", "MFI", "BB"]:
        try:
            fig = indicator_subplot(df, ind)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render {ind}: {e}")

# === Tab 2: Linear Regression ===
with tab2:
    st.subheader("OLS Linear Regression")

    shift_lr = st.slider("Shift (days)", 1, 30, 5, key="lr_shift")

    if not features:
        st.warning("Select at least one feature")
        st.stop()

    try:
        lr = LinearRegression(df, features=features)
        lr.add_target(shift=shift_lr)
        lr.fix_autocorrelation()
        result = lr.fit()
        validation = lr.validate()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R-squared", f"{result['r_squared']:.4f}")
        with col2:
            st.metric("F p-value", f"{result['p_value']:.4e}")
        with col3:
            st.metric("Intercept", f"{result['intercept']:.4f}")

        # Coefficient table
        coef_df = pd.DataFrame({
            "Feature": list(result["coefficients"].keys()),
            "Coefficient": list(result["coefficients"].values()),
        })
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        # Validation plots
        st.subheader("Residual Diagnostics")
        c1, c2, c3 = st.columns(3)

        residuals = validation["residuals"]
        predictions = validation["predictions"]

        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=predictions, y=residuals, mode="markers",
                                     marker=dict(color=COLORS["blue"], opacity=0.5)))
            fig.add_hline(y=0, line_color=COLORS["red"])
            fig.update_layout(**CHART_LAYOUT, title="Linearity",
                              xaxis_title="Predicted", yaxis_title="Residuals")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=residuals.values[:-1], y=residuals.values[1:],
                                     mode="markers", marker=dict(color=COLORS["blue"], opacity=0.5)))
            fig.update_layout(**CHART_LAYOUT, title="Independence (Lag Plot)",
                              xaxis_title="Residual(t)", yaxis_title="Residual(t+1)")
            st.plotly_chart(fig, use_container_width=True)

        with c3:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=residuals, nbinsx=30,
                                       marker_color=COLORS["blue"]))
            fig.update_layout(**CHART_LAYOUT, title="Normality",
                              xaxis_title="Residual")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Linear Regression failed: {e}")


# === Tab 3: Logistic Regression ===
with tab3:
    st.subheader("Logistic Regression")

    if not features:
        st.warning("Select at least one feature")
        st.stop()

    try:
        logreg = LogisticRegression(df, features=features)

        # AUC Optimization
        st.markdown("**AUC Optimization (Train Set)**")
        train_df_all, _ = logreg.train_test_split(train_size=0.7)

        auc_results = logreg.explore_shift_auc(train_df_all)
        best_shift = int(auc_results.iloc[0]["Shift"])

        fig_auc = go.Figure()
        fig_auc.add_trace(go.Scatter(
            x=auc_results["Shift"], y=auc_results["AUC"],
            mode="lines+markers",
            line=dict(color=COLORS["blue"], width=2),
            marker=dict(size=8),
        ))
        fig_auc.update_layout(**CHART_LAYOUT, title=f"AUC by Shift (Best: {best_shift})",
                              xaxis_title="Shift", yaxis_title="AUC")
        st.plotly_chart(fig_auc, use_container_width=True)

        # Re-fit with optimal shift
        logreg2 = LogisticRegression(df, features=features)
        logreg2.add_target(shift=best_shift)
        train_df, test_df = logreg2.train_test_split(train_size=0.7)

        # Train
        train_result = logreg2.fit(train_df)
        roc_train = logreg2.get_roc_data()
        cm_train = logreg2.get_confusion_matrix().values

        # Predict on test
        test_probs = logreg2.predict(test_df)
        test_target = test_df["Target"].loc[test_probs.index]
        test_auc = roc_auc_score(test_target, test_probs)
        test_fpr, test_tpr, _ = roc_curve(test_target, test_probs)
        test_cm = confusion_matrix(test_target, (test_probs > 0.5).astype(int))

        # Display side by side
        st.divider()
        st.subheader("Train vs Test Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Train** (AUC = {train_result['auc']:.3f})")
            st.plotly_chart(roc_curve_chart(roc_train["fpr"], roc_train["tpr"],
                                            roc_train["auc"], "ROC - Train"),
                           use_container_width=True)
            st.plotly_chart(confusion_matrix_chart(cm_train, "Confusion Matrix - Train"),
                           use_container_width=True)

        with col2:
            st.markdown(f"**Test** (AUC = {test_auc:.3f})")
            st.plotly_chart(roc_curve_chart(test_fpr, test_tpr, test_auc, "ROC - Test"),
                           use_container_width=True)
            st.plotly_chart(confusion_matrix_chart(test_cm, "Confusion Matrix - Test"),
                           use_container_width=True)

    except Exception as e:
        st.error(f"Logistic Regression failed: {e}")


# === Tab 4: Signal Dashboard ===
with tab4:
    st.subheader("Signal Dashboard")

    # Generate signals from raw data (need enough rows)
    try:
        sig_rsi = rsi_signal(df_raw)
        sig_log = logreg_signal(df_raw)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### RSI + MACD Signal")
            render_signal_badge(sig_rsi["signal"], sig_rsi.get("reason", ""))
            if sig_rsi.get("rsi") is not None:
                st.markdown(f"**RSI:** {sig_rsi['rsi']:.1f}")
            if sig_rsi.get("macd_hist") is not None:
                st.markdown(f"**MACD Hist:** {sig_rsi['macd_hist']:.4f}")

        with c2:
            st.markdown("#### LogReg Signal")
            render_signal_badge(sig_log["signal"], sig_log.get("reason", ""))
            if sig_log.get("probability") is not None:
                st.markdown(f"**P(Up):** {sig_log['probability']:.3f}")
            if sig_log.get("auc") is not None:
                st.markdown(f"**AUC:** {sig_log['auc']:.3f}")

    except Exception as e:
        st.warning(f"Could not generate signals: {e}")

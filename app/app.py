import streamlit as st
import numpy as np
import pandas as pd
from pandas.plotting import lag_plot
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import yfinance_fix

# Import tradbot modules
import sys
sys.path.append('..')
from tradbot.strategy import TechnicalIndicators
from tradbot.risk import calculate_sharpe_ratio, calculate_max_drawdown, calculate_var
from tradbot.portfolio import optimize_max_sharpe, optimize_min_volatility, get_efficient_frontier

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(page_title="Quant Trading Dashboard", layout="wide")

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header("Daten")
TICKER = st.sidebar.text_input("Ticker", value="SPY")
INTERVAL = st.sidebar.selectbox("Intervall", ["1d", "1h", "15m", "5m"], index=0)
LOOKBACK = st.sidebar.number_input("Lookback", value=5000, step=100)

st.sidebar.divider()
ANALYSIS = st.sidebar.radio(
    "Analyse",
    ["Indikatoren", "Lineare Regression", "Logistische Regression", "Risk Metrics", "Portfolio Optimization"]
)

st.sidebar.divider()
st.sidebar.header("Indikator Parameter")

with st.sidebar.expander("MACD"):
    MACD_FAST = st.number_input("Fast", 12)
    MACD_SLOW = st.number_input("Slow", 27)
    MACD_SPAN = st.number_input("Signal", 9)

with st.sidebar.expander("MFI"):
    MFI_LENGTH = st.number_input("Length", 14)

with st.sidebar.expander("Bollinger Bands"):
    BB_LENGTH = st.number_input("BB Length", 20)
    BB_STD = st.number_input("Std Dev", 2)

with st.sidebar.expander("RSI"):
    RSI_LENGTH = st.number_input("RSI Length", 14)

FEATURES = st.sidebar.multiselect(
    "Features",
    ["MACD_HIST", "MFI", "BB", "RSI"],
    default=["MACD_HIST", "MFI", "BB", "RSI"]
)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data(ticker, interval, lookback):
    """Load price data from Yahoo Finance."""
    period = "730d" if interval == "1h" else "max"
    df = yf.download(
        ticker,
        session=yfinance_fix.chrome_session,
        interval=interval,
        period=period,
        progress=False
    )
    if df.empty:
        return None
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df.iloc[-lookback:, :].copy()


def add_indicators(df):
    """Add technical indicators using TechnicalIndicators class."""
    ti = TechnicalIndicators(df)
    ti.add_macd(MACD_FAST, MACD_SLOW, MACD_SPAN)
    ti.add_mfi(MFI_LENGTH)
    ti.add_bb(BB_LENGTH, BB_STD)
    ti.add_rsi(RSI_LENGTH)
    return ti.dropna().get_df()

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_indicators(df):
    """Plot technical indicators in tabs."""
    subset = df.tail(150).reset_index(drop=True)

    tab1, tab2, tab3, tab4 = st.tabs(["MACD", "MFI", "Bollinger Bands", "RSI"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['green' if x >= 0 else 'red' for x in subset["MACD_HIST"]]
        ax.bar(range(len(subset)), subset["MACD_HIST"], color=colors, alpha=0.5)
        ax.plot(subset["MACD"], label="MACD", color="blue")
        ax.plot(subset["Signal"], label="Signal", color="orange")
        ax.legend()
        ax.set_title(f"MACD ({MACD_FAST}, {MACD_SLOW}, {MACD_SPAN})")
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(subset["MFI"], color="purple")
        ax.axhline(70, color="red", linestyle="--")
        ax.axhline(30, color="green", linestyle="--")
        ax.set_title(f"MFI ({MFI_LENGTH})")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(subset["Close"], color="black", alpha=0.7)
        ax.plot(subset["Upper"], color="red", linestyle="--", alpha=0.5)
        ax.plot(subset["Lower"], color="green", linestyle="--", alpha=0.5)
        ax.fill_between(range(len(subset)), subset["Upper"], subset["Lower"], alpha=0.1)
        ax.set_title(f"Bollinger Bands ({BB_LENGTH}, {BB_STD})")
        st.pyplot(fig)

    with tab4:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(subset["RSI"], color="blue")
        ax.axhline(70, color="red", linestyle="--")
        ax.axhline(30, color="green", linestyle="--")
        ax.set_title(f"RSI ({RSI_LENGTH})")
        st.pyplot(fig)

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def run_linear_regression(df, features, shift):
    """Run OLS Linear Regression."""
    df_ols = df.copy()
    df_ols["Target"] = (df_ols["Close"].shift(-shift) - df_ols["Close"]) / df_ols["Close"] * 100
    df_ols = df_ols.iloc[::shift].dropna()

    X = sm.add_constant(df_ols[features])
    y = df_ols["Target"]
    model = sm.OLS(y, X).fit()

    return model, y, model.predict(X)


def run_logistic_regression(df, features, shift):
    """Run Logistic Regression."""
    df_log = df.copy()
    df_log["Target"] = (df_log["Close"].shift(-shift) > df_log["Close"]).astype(int)
    df_log = df_log.dropna()

    X = sm.add_constant(df_log[features])
    y = df_log["Target"]
    model = sm.Logit(y, X).fit(disp=0)

    return model, y, model.predict(X)


def find_optimal_shift(df, features, max_shift=20):
    """Find optimal shift by AUC."""
    results = []
    for s in range(1, max_shift + 1):
        try:
            _, y, probs = run_logistic_regression(df, features, s)
            auc = roc_auc_score(y, probs)
            results.append({"Shift": s, "AUC": auc})
        except:
            pass
    return pd.DataFrame(results)

# =============================================================================
# MAIN APP
# =============================================================================

st.title(f"Quant Trading Dashboard: {TICKER}")

if st.sidebar.button("Analyse Starten"):

    # Load Data
    with st.spinner("Lade Daten..."):
        df_raw = load_data(TICKER, INTERVAL, LOOKBACK)

    if df_raw is None:
        st.error("Fehler beim Laden der Daten")
        st.stop()

    # Add Indicators
    df = add_indicators(df_raw)
    st.success(f"{len(df)} Datenpunkte geladen")

    # ==========================================================================
    # INDIKATOREN
    # ==========================================================================
    if ANALYSIS == "Indikatoren":
        st.header("Technische Indikatoren")
        plot_indicators(df)

    # ==========================================================================
    # LINEARE REGRESSION
    # ==========================================================================
    elif ANALYSIS == "Lineare Regression":
        st.header("Lineare Regression (OLS)")

        SHIFT = st.slider("Shift (Tage)", 1, 30, 5)
        model, y, preds = run_linear_regression(df, FEATURES, SHIFT)

        st.text(model.summary())

        # Validation Plots
        resid = y - preds
        c1, c2, c3 = st.columns(3)

        with c1:
            fig, ax = plt.subplots()
            ax.scatter(preds, resid, alpha=0.5)
            ax.axhline(0, color='red')
            ax.set_title("Linearität")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots()
            lag_plot(resid, ax=ax)
            ax.set_title("Unabhängigkeit")
            st.pyplot(fig)

        with c3:
            fig, ax = plt.subplots()
            ax.hist(resid, bins=30)
            ax.set_title("Normalverteilung")
            st.pyplot(fig)

    # ==========================================================================
    # LOGISTISCHE REGRESSION
    # ==========================================================================
    elif ANALYSIS == "Logistische Regression":
        st.header("Logistische Regression")

        # Train/Test Split
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df_shuffled) * 0.7)
        train_df = df_shuffled.iloc[:split_idx].copy()
        test_df = df_shuffled.iloc[split_idx:].copy()

        st.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

        # Find Optimal Shift
        st.subheader("AUC Optimierung")
        auc_results = find_optimal_shift(train_df, FEATURES)

        if not auc_results.empty:
            best_shift = int(auc_results.sort_values("AUC", ascending=False).iloc[0]["Shift"])
            st.success(f"Optimaler Shift: {best_shift}")

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(auc_results["Shift"], auc_results["AUC"], marker="o")
            ax.set_xlabel("Shift")
            ax.set_ylabel("AUC")
            ax.grid(True)
            st.pyplot(fig)

            # Train Model
            model, y_train, prob_train = run_logistic_regression(train_df, FEATURES, best_shift)

            # Predict on Test
            test_df["Target"] = (test_df["Close"].shift(-best_shift) > test_df["Close"]).astype(int)
            test_df = test_df.dropna()
            X_test = sm.add_constant(test_df[FEATURES])
            y_test = test_df["Target"]
            prob_test = model.predict(X_test)

            # Compare Train vs Test
            st.divider()
            st.subheader("Train vs Test Vergleich")

            col1, col2 = st.columns(2)

            for col, (y_true, probs, title) in zip(
                [col1, col2],
                [(y_train, prob_train, "Train"), (y_test, prob_test, "Test")]
            ):
                with col:
                    st.markdown(f"### {title}")

                    # ROC
                    fpr, tpr, _ = roc_curve(y_true, probs)
                    auc_score = roc_auc_score(y_true, probs)

                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(fpr, tpr, label=f"AUC={auc_score:.2f}")
                    ax.plot([0, 1], [0, 1], linestyle="--")
                    ax.legend()
                    ax.set_title("ROC Curve")
                    st.pyplot(fig)

                    # Confusion Matrix
                    cm = confusion_matrix(y_true, (probs > 0.5).astype(int))
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)

    # ==========================================================================
    # RISK METRICS
    # ==========================================================================
    elif ANALYSIS == "Risk Metrics":
        st.header("Risk Metrics")

        returns = df["Close"].pct_change().dropna()
        prices = df["Close"].dropna()

        # Metrics
        sharpe = calculate_sharpe_ratio(returns)
        max_dd = calculate_max_drawdown(prices)
        var_95 = calculate_var(returns, 0.95)
        var_99 = calculate_var(returns, 0.99)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col2.metric("Max Drawdown", f"{max_dd:.2%}")
        col3.metric("VaR (95%)", f"{var_95:.2%}")
        col4.metric("VaR (99%)", f"{var_99:.2%}")

        # Drawdown Chart
        st.subheader("Drawdown")
        rolling_max = prices.cummax()
        drawdown = (prices - rolling_max) / rolling_max

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(range(len(drawdown)), drawdown * 100, 0, color='red', alpha=0.3)
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Return Distribution
        st.subheader("Return Verteilung")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(returns * 100, bins=50, color='steelblue', alpha=0.7)
        ax.axvline(var_95 * 100, color='orange', linestyle='--', label=f'VaR 95%')
        ax.axvline(var_99 * 100, color='red', linestyle='--', label=f'VaR 99%')
        ax.legend()
        ax.set_xlabel("Daily Return (%)")
        st.pyplot(fig)

    # ==========================================================================
    # PORTFOLIO OPTIMIZATION
    # ==========================================================================
    elif ANALYSIS == "Portfolio Optimization":
        st.header("Portfolio Optimization")

        portfolio_tickers = st.text_input(
            "Tickers (comma separated)",
            value="AAPL, MSFT, GOOGL, AMZN, META"
        )
        tickers = [t.strip() for t in portfolio_tickers.split(",")]

        with st.spinner("Lade Portfolio Daten..."):
            prices = yf.download(
                tickers,
                period="2y",
                session=yfinance_fix.chrome_session,
                progress=False
            )['Close']

        if prices.empty:
            st.error("Keine Daten")
            st.stop()

        try:
            weights_sharpe, perf_sharpe = optimize_max_sharpe(prices)
            weights_minvol, perf_minvol = optimize_min_volatility(prices)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Max Sharpe")
                st.metric("Return", f"{perf_sharpe[0]:.2%}")
                st.metric("Volatility", f"{perf_sharpe[1]:.2%}")
                st.metric("Sharpe", f"{perf_sharpe[2]:.2f}")

                w = pd.Series(weights_sharpe)
                w = w[w > 0.01]
                fig, ax = plt.subplots()
                ax.barh(w.index, w.values * 100, color='steelblue')
                ax.set_xlabel("Weight (%)")
                st.pyplot(fig)

            with col2:
                st.subheader("Min Volatility")
                st.metric("Return", f"{perf_minvol[0]:.2%}")
                st.metric("Volatility", f"{perf_minvol[1]:.2%}")
                st.metric("Sharpe", f"{perf_minvol[2]:.2f}")

                w = pd.Series(weights_minvol)
                w = w[w > 0.01]
                fig, ax = plt.subplots()
                ax.barh(w.index, w.values * 100, color='green')
                ax.set_xlabel("Weight (%)")
                st.pyplot(fig)

            # Efficient Frontier
            st.divider()
            st.subheader("Efficient Frontier")

            frontier = get_efficient_frontier(prices, n_points=30)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(frontier['volatility'] * 100, frontier['return'] * 100, 'b-', lw=2)
            ax.scatter(perf_sharpe[1] * 100, perf_sharpe[0] * 100, marker='*', s=300, c='red', label='Max Sharpe')
            ax.scatter(perf_minvol[1] * 100, perf_minvol[0] * 100, marker='o', s=200, c='green', label='Min Vol')
            ax.set_xlabel("Volatility (%)")
            ax.set_ylabel("Return (%)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

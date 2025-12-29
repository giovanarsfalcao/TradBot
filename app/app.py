import streamlit as st
import numpy as np 
import pandas as pd
from pandas.plotting import lag_plot
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import yfinance_fix # Importiert den zuvor erstellten Fix

# --- SEITEN-KONFIGURATION ---
st.set_page_config(page_title="Quant Regression Tool", layout="wide")

# --- SIDEBAR: BENUTZER-EINGABEN ---
st.sidebar.header("Daten-Einstellungen")
TICKER = st.sidebar.text_input("Ticker Symbol", value="SPY")
INTERVAL = st.sidebar.selectbox("Intervall", ["1d", "1h", "15m", "5m"], index=0)
LOOKBACK = st.sidebar.number_input("Lookback (Anzahl Zeilen)", value=10000, step=100)
SHIFT = st.sidebar.slider("Vorhersage-Zeitraum (Shift)", 1, 30, 5)

if INTERVAL == "1h":
    PERIOD = "730d"
else: 
    PERIOD = "max"

st.sidebar.header("Indikator-Parameter")

# MACD Einstellungen
with st.sidebar.expander("MACD Parameter"):
    MACD_FAST = st.number_input("Fast EMA", value=12)
    MACD_SLOW = st.number_input("Slow EMA", value=27)
    MACD_SPAN = st.number_input("Signal Span", value=9)

# MFI Einstellungen
with st.sidebar.expander("MFI Parameter"):
    MFI_LENGTH = st.number_input("MFI Länge", value=14)
    MFI_OB = st.slider("MFI Overbought", 50, 90, 70)
    MFI_OS = st.slider("MFI Oversold", 10, 50, 30)

# BB Einstellungen
with st.sidebar.expander("Bollinger Bands"):
    BB_LENGTH = st.number_input("BB Länge", value=20)
    BB_STD = st.number_input("Std Abweichung", value=2)

# RSI Einstellungen
with st.sidebar.expander("RSI Parameter"):
    RSI_LENGTH = st.number_input("RSI Länge", value=14)
    RSI_OB = st.slider("RSI Overbought", 50, 90, 70)
    RSI_OS = st.slider("RSI Oversold", 10, 50, 30)

STRATEGY = st.sidebar.multiselect(
    "Features für Regression wählen", 
    ["MACD_HIST", "MFI", "BB", "RSI"], 
    default=["MACD_HIST", "MFI", "BB", "RSI"]
)

# --- FUNKTIONEN FÜR BERECHNUNGEN ---

@st.cache_data
def load_data(ticker, interval, period, lookback):
    df = yf.download(ticker, session=yfinance_fix.chrome_session, interval=interval, period=period, progress=False)
    if df.empty:
        return None
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df.iloc[-lookback:, :].copy()

def add_indicators(df):
    # MACD
    df["ema_f"] = df["Close"].ewm(span=MACD_FAST).mean()
    df["ema_s"] = df["Close"].ewm(span=MACD_SLOW).mean()
    df["MACD"] = df["ema_f"] - df["ema_s"]
    df["Signal"] = df["MACD"].ewm(span=MACD_SPAN).mean()
    df["MACD_HIST"] = df["MACD"] - df["Signal"]
    
    # MFI
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    pos_f = np.where(tp.diff() > 0, mf, 0)
    neg_f = np.where(tp.diff() < 0, mf, 0)
    mfr = pd.Series(pos_f).rolling(MFI_LENGTH).sum() / pd.Series(neg_f).rolling(MFI_LENGTH).sum()
    df["MFI"] = 100 - (100 / (1 + mfr.values))
    
    # BB
    df["BB_SMA"] = df["Close"].rolling(BB_LENGTH).mean()
    df["BB_STD"] = df["Close"].rolling(BB_LENGTH).std()
    u_band = df["BB_SMA"] + (BB_STD * df["BB_STD"])
    l_band = df["BB_SMA"] - (BB_STD * df["BB_STD"])
    df["BB"] = (u_band - df["Close"]) / (u_band - l_band)
    
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_LENGTH).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_LENGTH).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    return df.dropna()

def process_regression_data(df, shift):
    df[f"Close+{shift}"] = df["Close"].shift(-shift)
    df["Target"] = (df[f"Close+{shift}"] - df["Close"]) / df["Close"] * 100
    # Down-Sampling gegen Autokorrelation
    df_sampled = df.iloc[::shift].copy()
    return df_sampled.dropna()

# --- MAIN UI ---

st.title("Quant Strategy Regression Tool")
st.markdown(f"Analyse für **{TICKER}** | Intervall: **{INTERVAL}** | Shift: **{SHIFT}**")

if st.button("Analyse Starten"):
    with st.spinner("Daten werden verarbeitet..."):
        # 1. Daten laden
        df_raw = load_data(TICKER, INTERVAL, PERIOD, LOOKBACK)
        
        if df_raw is not None:
            # 2. Indikatoren berechnen
            df_ind = add_indicators(df_raw)
            
            # 3. Target & Down-Sampling
            df_final = process_regression_data(df_ind, SHIFT)
            
            if len(df_final) < 20:
                st.error("Zu wenig Datenpunkte nach dem Down-Sampling. Erhöhe den Lookback!")
            else:
                # 4. Regression
                X = df_final[STRATEGY]
                y = df_final["Target"]
                X_const = sm.add_constant(X)
                model = sm.OLS(y, X_const).fit()
                
                # --- ERGEBNISSE ANZEIGEN ---
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Modell Zusammenfassung")
                    st.text(model.summary())
                
                with col2:
                    st.subheader("Wichtigste Metriken")
                    st.metric("R-Squared", f"{model.rsquared:.4f}")
                    st.metric("Anzahl Beobachtungen", len(df_final))
                    st.metric("P-Value (Modell)", f"{model.f_pvalue:.6f}")

                # --- VISUALISIERUNG ---
                st.divider()
                st.subheader("Visualisierung der Ergebnisse")
                
                v_col1, v_col2 = st.columns(2)
                
                with v_col1:
                    # Actual vs Predicted
                    fig_res, ax_res = plt.subplots()
                    y_pred = model.predict(X_const)
                    ax_res.scatter(y_pred, y, alpha=0.5, color="royalblue")
                    ax_res.plot(y_pred, y_pred, color='red', linestyle='--')
                    ax_res.set_title("Tatsächlich vs. Vorhergesagt")
                    ax_res.set_xlabel("Vorhersage (y_pred)")
                    ax_res.set_ylabel("Tatsächlich (y_actual)")
                    st.pyplot(fig_res)
                
                with v_col2:
                    # MACD Histogramm Plot
                    fig_macd, ax_macd = plt.subplots()
                    colors = ['green' if x >= 0 else 'red' for x in df_ind["MACD_HIST"].tail(100)]
                    ax_macd.bar(range(100), df_ind["MACD_HIST"].tail(100), color=colors)
                    ax_macd.set_title("MACD Histogramm (Letzte 100 Perioden)")
                    st.pyplot(fig_macd)

                # --- VALIDIERUNG ---
                st.divider()
                st.subheader("Statistische Validierung (Residuen-Analyse)")
                
                # Residuen berechnen
                df_final["Predictions"] = model.predict(X_const)
                df_final["Residuals"] = df_final["Target"] - df_final["Predictions"]
                
                val_col1, val_col2, val_col3 = st.columns(3)
                
                with val_col1:
                    st.write("**1. Linearität & Homoskedastizität**")
                    fig_v1, ax_v1 = plt.subplots()
                    ax_v1.scatter(df_final["Predictions"], df_final["Residuals"], alpha=0.5)
                    ax_v1.axhline(0, color='red', linestyle='--')
                    st.pyplot(fig_v1)
                    
                with val_col2:
                    st.write("**2. Unabhängigkeit (Lag Plot)**")
                    fig_v2, ax_v2 = plt.subplots()
                    lag_plot(df_final["Residuals"], ax=ax_v2)
                    st.pyplot(fig_v2)
                    
                with val_col3:
                    st.write("**3. Normalverteilung**")
                    fig_v3, ax_v3 = plt.subplots()
                    ax_v3.hist(df_final["Residuals"], bins=30, color="skyblue", edgecolor="black")
                    st.pyplot(fig_v3)
                
                # Rohdaten Download
                st.divider()
                st.download_button(
                    label="Analysierte Daten als CSV exportieren",
                    data=df_final.to_csv().encode('utf-8'),
                    file_name=f"{TICKER}_quant_data.csv",
                    mime="text/csv"
                )
        else:
            st.error("Daten konnten nicht geladen werden. Prüfe Ticker und Verbindung.")
else:
    st.info("Klicke auf 'Analyse Starten', um die Regression mit den aktuellen Parametern durchzuführen.")

# 1.  **Installation:** Stellen Sie sicher, dass Sie `streamlit` installiert haben:
#     pip install 
# 2.  **App starten:** Öffnen Sie Ihr Terminal, navigieren Sie zu dem Ordner, in dem die Datei liegt, und führen Sie aus:
#     streamlit run app.py
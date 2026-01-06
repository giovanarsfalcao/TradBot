import streamlit as st
import numpy as np 
import pandas as pd
from pandas.plotting import lag_plot
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import yfinance_fix # Voraussetzung: Datei liegt im selben Ordner

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="Quant Analysis Dashboard", layout="wide")

# --- SIDEBAR: GLOBALE EINSTELLUNGEN ---
st.sidebar.header("📊 Daten-Konfiguration")
TICKER = st.sidebar.text_input("Ticker Symbol", value="SPY")
INTERVAL = st.sidebar.selectbox("Intervall", ["1d", "1h", "15m", "5m"], index=0)
LOOKBACK = st.sidebar.number_input("Lookback (Zeilen)", value=5000, step=100)

# Modell-Modus wählen
MODEL_TYPE = st.sidebar.radio("Wähle Analyse-Modell", ["Lineare Regression (OLS)", "Logistische Regression (Logit)"])

st.sidebar.divider()
st.sidebar.header("⚙️ Indikator-Parameter")

with st.sidebar.expander("Indikator Längen"):
    MACD_FAST = st.number_input("MACD Fast", value=12)
    MACD_SLOW = st.number_input("MACD Slow", value=27)
    MACD_SPAN = st.number_input("MACD Signal", value=9)
    MFI_LENGTH = st.number_input("MFI Länge", value=14)
    BB_LENGTH = st.number_input("BB Länge", value=20)
    RSI_LENGTH = st.number_input("RSI Länge", value=14)

# Strategie-Auswahl
STRATEGY_OPTIONS = ["MACD_HIST", "MFI", "BB", "RSI", "Volume_Change", "Close_Change"]
STRATEGY = st.sidebar.multiselect("Features für das Modell", STRATEGY_OPTIONS, default=["MACD_HIST", "MFI", "BB", "RSI"])

# --- FUNKTIONEN ---

@st.cache_data
def get_data(ticker, interval, lookback):
    period = "730d" if interval == "1h" else "max"
    df = yf.download(ticker, session=yfinance_fix.chrome_session, interval=interval, period=period, progress=False)
    if df.empty: return None
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    
    # Basis-Änderungen berechnen
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[f"{c}_Change"] = df[c].pct_change() * 100
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
    u_b = df["BB_SMA"] + (2 * df["BB_STD"])
    l_b = df["BB_SMA"] - (2 * df["BB_STD"])
    df["BB"] = (df["Close"] - l_b) / (u_b - l_b)
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_LENGTH).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_LENGTH).mean()
    df["RSI"] = 100 - (100 / (1 + (gain/loss)))
    return df.dropna()

# --- HAUPTBEREICH ---
st.title(f"Quant Analysis: {MODEL_TYPE}")

if st.sidebar.button("🚀 Analyse Starten"):
    df_raw = get_data(TICKER, INTERVAL, LOOKBACK)
    
    if df_raw is not None:
        df = add_indicators(df_raw)
        
        # --- MODUS 1: LINEARE REGREESSION ---
        if MODEL_TYPE == "Lineare Regression (OLS)":
            SHIFT = st.sidebar.slider("Vorhersage-Zeitraum (Shift)", 1, 30, 5)
            df[f"Target"] = (df["Close"].shift(-SHIFT) - df["Close"]) / df["Close"] * 100
            df_final = df.iloc[::SHIFT].dropna()
            
            X = sm.add_constant(df_final[STRATEGY])
            y = df_final["Target"]
            model = sm.OLS(y, X).fit()
            
            # UI
            st.subheader("Modell Zusammenfassung (OLS)")
            st.text(model.summary())
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Actual vs Predicted**")
                fig, ax = plt.subplots()
                ax.scatter(model.predict(X), y, alpha=0.5)
                ax.plot(y, y, color="red")
                st.pyplot(fig)
            with c2:
                st.write("**Residuen Verteilung**")
                fig, ax = plt.subplots()
                ax.hist(model.resid, bins=30)
                st.pyplot(fig)

        # --- MODUS 2: LOGISTISCHE REGRESSION ---
        else:
            st.subheader("Optimierung & Klassifizierung (Logit)")
            
            # 1. Train-Test Split (Gegen Overfitting)
            df_shuffled = df.sample(frac=1, random_state=42)
            split = int(len(df_shuffled) * 0.7)
            train_df = df_shuffled.iloc[:split].copy()
            test_df = df_shuffled.iloc[split:].copy()
            
            st.write(f"Datensatz geteilt: {len(train_df)} Training, {len(test_df)} Test-Reihen.")
            
            # 2. AUC Exploration (Welcher Shift ist optimal?)
            st.write("🔍 Suche optimalen Vorhersage-Horizont (Shift)...")
            auc_results = []
            for s in range(1, 15): # Kleinerer Range für Performance in App
                temp_df = train_df.copy()
                temp_df["T"] = (temp_df["Close"].shift(-s) > temp_df["Close"]).astype(int)
                temp_df = temp_df.dropna()
                if not temp_df.empty:
                    X_t = sm.add_constant(temp_df[STRATEGY])
                    m_t = sm.Logit(temp_df["T"], X_t).fit(disp=0)
                    auc_results.append({"Shift": s, "AUC": roc_auc_score(temp_df["T"], m_t.predict(X_t))})
            
            res_df = pd.DataFrame(auc_results)
            best_shift = int(res_df.loc[res_df['AUC'].idxmax()]['Shift'])
            
            # Plot AUC Entwicklung
            fig_auc, ax_auc = plt.subplots(figsize=(8,3))
            ax_auc.plot(res_df["Shift"], res_df["AUC"], marker="o")
            ax_auc.set_title("AUC Score nach Shift")
            st.pyplot(fig_auc)
            
            st.success(f"Optimaler Shift gefunden: **{best_shift}** (Maximale AUC)")
            
            # 3. Finales Modell auf Test-Daten validieren
            st.divider()
            st.subheader(f"Finales Modell Performance (Shift={best_shift})")
            
            def run_logit(data, s):
                data["T"] = (data["Close"].shift(-s) > data["Close"]).astype(int)
                data = data.dropna()
                X_f = sm.add_constant(data[STRATEGY])
                m_f = sm.Logit(data["T"], X_f).fit(disp=0)
                return data, m_f, X_f
            
            # Training vs Testing Comparison
            col_a, col_b = st.columns(2)
            
            for d, title, col in zip([train_df, test_df], ["TRAINING SET", "TEST SET (UNSEEN)"], [col_a, col_b]):
                with col:
                    st.write(f"**{title}**")
                    d_final, m_final, X_final = run_logit(d.copy(), best_shift)
                    probs = m_final.predict(X_final)
                    preds = (probs > 0.5).astype(int)
                    
                    # Metrics
                    fpr, tpr, _ = roc_curve(d_final["T"], probs)
                    st.write(f"AUC: {auc(fpr, tpr):.4f}")
                    
                    # Confusion Matrix Plot
                    fig_cm, ax_cm = plt.subplots(figsize=(4,3))
                    sns.heatmap(confusion_matrix(d_final["T"], preds), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual")
                    st.pyplot(fig_cm)

    else:
        st.error("Daten konnten nicht geladen werden.")
else:
    st.info("Klicken Sie auf 'Analyse Starten', um die Berechnungen zu beginnen.")

# 1.  **Installation:** Stellen Sie sicher, dass Sie `streamlit` installiert haben:
#     pip install 
# 2.  **App starten:** Öffnen Sie Ihr Terminal, navigieren Sie zu dem Ordner, in dem die Datei liegt, und führen Sie aus:
#     streamlit run app.py
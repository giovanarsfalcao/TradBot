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
st.set_page_config(page_title="Quant Analysis Ultimate", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("📊 Daten-Konfiguration")
TICKER = st.sidebar.text_input("Ticker Symbol", value="SPY")
INTERVAL = st.sidebar.selectbox("Intervall", ["1d", "1h", "15m", "5m"], index=0)
LOOKBACK = st.sidebar.number_input("Lookback (Zeilen)", value=5000, step=100)

MODEL_TYPE = st.sidebar.radio("Modell", ["Lineare Regression (OLS)", "Logistische Regression (Logit)"])

st.sidebar.divider()
st.sidebar.header("⚙️ Indikator-Parameter")

with st.sidebar.expander("MACD Settings"):
    MACD_FAST = st.number_input("Fast", 12)
    MACD_SLOW = st.number_input("Slow", 27)
    MACD_SPAN = st.number_input("Signal", 9)

with st.sidebar.expander("MFI Settings"):
    MFI_LENGTH = st.number_input("MFI Len", 14)
    OVERBOUGHT = st.number_input("Overbought", 70)
    OVERSOLD = st.number_input("Oversold", 30)

with st.sidebar.expander("Bollinger Bands"):
    BB_LENGTH = st.number_input("BB Len", 20)
    STD_DEV = st.number_input("Std Dev", 2)

with st.sidebar.expander("RSI Settings"):
    RSI_LENGTH = st.number_input("RSI Len", 14)
    RSI_OB = st.number_input("RSI OB", 70)
    RSI_OS = st.number_input("RSI OS", 30)

STRATEGY = st.sidebar.multiselect("Features", ["MACD_HIST", "MFI", "BB", "RSI", "Volume_Change"], default=["MACD_HIST", "MFI", "BB", "RSI"])

# --- DATEN & INDIKATOREN ---

@st.cache_data
def get_data(ticker, interval, lookback):
    period = "730d" if interval == "1h" else "max"
    df = yf.download(ticker, session=yfinance_fix.chrome_session, interval=interval, period=period, progress=False)
    if df.empty: return None
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[f"{c}_Change"] = df[c].pct_change() * 100
    return df.iloc[-lookback:, :].copy()

def add_MACD(df, fast, slow, span):
    df[f"{fast}_ema"] = df["Close"].ewm(span=fast).mean()
    df[f"{slow}_ema"] = df["Close"].ewm(span=slow).mean()
    df["MACD"] = df[f"{fast}_ema"] - df[f"{slow}_ema"]
    df["Signal"] = df["MACD"].ewm(span=span).mean()
    df["MACD_HIST"] = df["MACD"] - df["Signal"]
    return df

def add_MFI(df, length, ob, os):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    pos_f = np.where(tp.diff() > 0, mf, 0)
    neg_f = np.where(tp.diff() < 0, mf, 0)
    mfr = pd.Series(pos_f).rolling(length).sum() / pd.Series(neg_f).rolling(length).sum()
    df["MFI"] = 100 - (100 / (1 + mfr.values))
    return df

def add_BB(df, length, std):
    df["BB_SMA"] = df["Close"].rolling(length).mean()
    df["BB_STD"] = df["Close"].rolling(length).std()
    df["Upper"] = df["BB_SMA"] + (std * df["BB_STD"])
    df["Lower"] = df["BB_SMA"] - (std * df["BB_STD"])
    # Normalized BB value
    df["BB"] = (df["Close"] - df["Lower"]) / (df["Upper"] - df["Lower"])
    return df

def add_RSI(df, length):
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def prepare_data(df):
    df = add_MACD(df, MACD_FAST, MACD_SLOW, MACD_SPAN)
    df = add_MFI(df, MFI_LENGTH, OVERBOUGHT, OVERSOLD)
    df = add_BB(df, BB_LENGTH, STD_DEV)
    df = add_RSI(df, RSI_LENGTH)
    return df.dropna()

# --- PLOTTING FUNKTIONEN ---

def plot_indicators(df):
    st.subheader("📈 Technische Indikatoren Visualisierung")
    
    # Letzte 150 Perioden für bessere Sichtbarkeit
    subset = df.tail(150).reset_index(drop=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["MACD", "MFI", "Bollinger Bands", "RSI"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['green' if x >= 0 else 'red' for x in subset["MACD_HIST"]]
        ax.bar(range(len(subset)), subset["MACD_HIST"], color=colors, alpha=0.5, label="Hist")
        ax.plot(subset["MACD"], label="MACD", color="blue")
        ax.plot(subset["Signal"], label="Signal", color="orange")
        ax.set_title(f"MACD ({MACD_FAST}, {MACD_SLOW}, {MACD_SPAN})")
        ax.legend()
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(subset["MFI"], color="purple", label="MFI")
        ax.axhline(OVERBOUGHT, color="red", linestyle="--")
        ax.axhline(OVERSOLD, color="green", linestyle="--")
        ax.set_title(f"MFI ({MFI_LENGTH})")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(subset["Close"], label="Close", color="black", alpha=0.7)
        ax.plot(subset["Upper"], label="Upper", color="red", linestyle="--", alpha=0.5)
        ax.plot(subset["Lower"], label="Lower", color="green", linestyle="--", alpha=0.5)
        ax.fill_between(range(len(subset)), subset["Upper"], subset["Lower"], color="gray", alpha=0.1)
        ax.set_title(f"Bollinger Bands ({BB_LENGTH}, {STD_DEV})")
        st.pyplot(fig)

    with tab4:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(subset["RSI"], color="blue", label="RSI")
        ax.axhline(RSI_OB, color="red", linestyle="--")
        ax.axhline(RSI_OS, color="green", linestyle="--")
        ax.set_title(f"RSI ({RSI_LENGTH})")
        st.pyplot(fig)

# --- LOGIT HELPERS ---

def explore_shift_auc(df_train, features):
    results = []
    # Teste Shift 1 bis 20
    shifts = range(1, 21)
    
    for s in shifts:
        temp = df_train.copy()
        temp["T"] = (temp["Close"].shift(-s) > temp["Close"]).astype(int)
        temp = temp.dropna()
        if len(temp) > 50:
            X = sm.add_constant(temp[features])
            try:
                m = sm.Logit(temp["T"], X).fit(disp=0)
                preds = m.predict(X)
                score = roc_auc_score(temp["T"], preds)
                results.append({"Shift": s, "AUC": score})
            except:
                pass
    return pd.DataFrame(results)

def run_logit_model(df, features, target_col):
    df_clean = df.dropna(subset=features + [target_col])
    X = sm.add_constant(df_clean[features])
    y = df_clean[target_col]
    model = sm.Logit(y, X).fit(disp=0)
    probs = model.predict(X)
    return df_clean, y, probs

# --- HAUPTPROGRAMM ---

st.title(f"Quant Trading Dashboard: {TICKER}")

if st.sidebar.button("🚀 Analyse Starten"):
    with st.spinner("Lade Daten & Berechne Indikatoren..."):
        df_raw = get_data(TICKER, INTERVAL, LOOKBACK)
        
        if df_raw is not None:
            # 1. Indikatoren berechnen
            df_ind = prepare_data(df_raw)
            
            # 2. Indikatoren Visualisieren
            plot_indicators(df_ind)
            
            st.divider()
            
            # 3. Modellierung
            if MODEL_TYPE == "Lineare Regression (OLS)":
                st.header("Lineare Regression")
                SHIFT = st.sidebar.slider("Manueller Shift", 1, 30, 5)
                
                # Target und Downsampling
                df_ols = df_ind.copy()
                df_ols["Target"] = (df_ols["Close"].shift(-SHIFT) - df_ols["Close"]) / df_ols["Close"] * 100
                df_ols = df_ols.iloc[::SHIFT].dropna()
                
                X = sm.add_constant(df_ols[STRATEGY])
                y = df_ols["Target"]
                model = sm.OLS(y, X).fit()
                
                st.text(model.summary())
                
                # Validation Plots
                c1, c2, c3 = st.columns(3)
                preds = model.predict(X)
                resid = y - preds
                
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

            else:
                # LOGISTISCHE REGRESSION
                st.header("Logistische Regression & Machine Learning")
                
                # A) Train Test Split (Random)
                df_shuffled = df_ind.sample(frac=1, random_state=42).reset_index(drop=True)
                split_idx = int(len(df_shuffled) * 0.7)
                train_df = df_shuffled.iloc[:split_idx].copy()
                test_df = df_shuffled.iloc[split_idx:].copy()
                
                st.info(f"Split: {len(train_df)} Training | {len(test_df)} Test Zeilen")
                
                # B) AUC Optimierung (Shift Suche auf Training Set)
                st.subheader("1. Optimierung: AUC by Shift (Training Set)")
                auc_res = explore_shift_auc(train_df, STRATEGY)
                
                if not auc_res.empty:
                    best_shift = int(auc_res.sort_values(by="AUC", ascending=False).iloc[0]["Shift"])
                    st.success(f"Optimaler Shift gefunden: **{best_shift}**")
                    
                    fig_auc, ax_auc = plt.subplots(figsize=(10, 3))
                    ax_auc.plot(auc_res["Shift"], auc_res["AUC"], marker="o", color="purple")
                    ax_auc.set_title("AUC Score vs. Shift")
                    ax_auc.set_xlabel("Shift")
                    ax_auc.set_ylabel("AUC")
                    ax_auc.grid(True)
                    st.pyplot(fig_auc)
                    
                    # C) Modell Training & Testing mit optimalem Shift
                    st.divider()
                    st.subheader(f"Vergleich: Training vs. Testing (Shift {best_shift})")
                    
                    # Target berechnen für beide Sets
                    train_df["Target"] = (train_df["Close"].shift(-best_shift) > train_df["Close"]).astype(int)
                    test_df["Target"] = (test_df["Close"].shift(-best_shift) > test_df["Close"]).astype(int)
                    
                    # Training ausführen
                    train_clean, y_train, prob_train = run_logit_model(train_df, STRATEGY, "Target")
                    # Testing ausführen (Modell auf Testdaten anwenden wäre korrekter, hier vereinfacht neu gefittet für Analyse-Zwecke des Users)
                    # Besser: Modell von Train nehmen und auf Test predicten
                    X_train = sm.add_constant(train_clean[STRATEGY])
                    model_final = sm.Logit(y_train, X_train).fit(disp=0)
                    
                    # Predict Test
                    test_clean = test_df.dropna(subset=STRATEGY + ["Target"])
                    X_test = sm.add_constant(test_clean[STRATEGY])
                    y_test = test_clean["Target"]
                    prob_test = model_final.predict(X_test)
                    
                    # --- 4 GRAPHEN VERGLEICH ---
                    c_train, c_test = st.columns(2)
                    
                    def plot_4_graphs(y_true, y_prob, title_prefix):
                        # 1. Distribution
                        fig1, ax1 = plt.subplots(figsize=(5,3))
                        ax1.hist(y_prob, bins=30, color="gray")
                        ax1.set_title(f"{title_prefix}: Distribution")
                        st.pyplot(fig1)
                        
                        # 2. ROC
                        fpr, tpr, _ = roc_curve(y_true, y_prob)
                        score = roc_auc_score(y_true, y_prob)
                        fig2, ax2 = plt.subplots(figsize=(5,3))
                        ax2.plot(fpr, tpr, color="orange", label=f"AUC={score:.2f}")
                        ax2.plot([0,1], [0,1], linestyle="--")
                        ax2.legend()
                        ax2.set_title(f"{title_prefix}: ROC Curve")
                        st.pyplot(fig2)
                        
                        # 3. Confusion Matrix
                        y_pred = (y_prob > 0.5).astype(int)
                        cm = confusion_matrix(y_true, y_pred)
                        fig3, ax3 = plt.subplots(figsize=(5,3))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
                        ax3.set_title(f"{title_prefix}: Confusion Matrix")
                        st.pyplot(fig3)

                    with c_train:
                        st.markdown("### 🏋️ Training Data")
                        plot_4_graphs(y_train, prob_train, "Train")
                        
                    with c_test:
                        st.markdown("### 🧪 Test Data (Unseen)")
                        plot_4_graphs(y_test, prob_test, "Test")

                else:
                    st.error("Konnte keine validen AUC-Werte berechnen.")
        else:
            st.error("Fehler beim Daten-Download.")

# 1.  **Installation:** Stellen Sie sicher, dass Sie `streamlit` installiert haben:
#     pip install 
# 2.  **App starten:** Öffnen Sie Ihr Terminal, navigieren Sie zu dem Ordner, in dem die Datei liegt, und führen Sie aus:
#     streamlit run app.py
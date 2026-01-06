import streamlit as st
import numpy as np 
import pandas as pd
from pandas.plotting import lag_plot
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import yfinance_fix # Voraussetzung: yfinance_fix.py im selben Ordner

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="Quant Analysis Pro", layout="wide")

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
STRATEGY_OPTIONS = ["MACD_HIST", "MFI", "BB", "RSI", "Volume_Change", "Close_Change", "High_Change", "Low_Change", "Open_Change"]
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
    
    # Plotting Price Movement
    st.sidebar.subheader("Kursverlauf")
    st.sidebar.line_chart(df["Close"].iloc[-200:])
    
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

def train_test_split_func(df, train_size=0.7):
    # Chronologischer Split (wichtig bei Zeitreihen!)
    # Alternative: Random Split (wie im User Code gewünscht mit sample(frac=1))
    # Wir nutzen hier den Random Split wie angefordert:
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * train_size)
    train = df_shuffled.iloc[:split_idx]
    test = df_shuffled.iloc[split_idx:]
    return train, test

def add_target(df, shift):
    df = df.copy()
    df[f"Close + {shift}"] = df["Close"].shift(-shift)
    df["Target"] = (df[f"Close + {shift}"] > df["Close"]).astype(int)
    return df.dropna()

def generate_logit_model(df, features, target="Target"):
    subset = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 10: return None, None, None
    
    X = sm.add_constant(subset[features])
    y = subset[target]
    
    try:
        model = sm.Logit(y, X).fit(disp=0)
        y_pred_prob = model.predict(X)
        return model, y, y_pred_prob
    except:
        return None, None, None

def explore_shift_auc(df_raw, features):
    results = []
    # Progress Bar für die Suche
    progress_bar = st.progress(0)
    shift_range = range(1, 30) # Begrenzt für Performance, kann erhöht werden
    
    for i, shift in enumerate(shift_range):
        df_temp = add_target(df_raw.copy(), shift)
        model, y, y_prob = generate_logit_model(df_temp, features)
        
        if y is not None:
            score = roc_auc_score(y, y_prob)
            results.append({"Shift": shift, "AUC": score})
        
        progress_bar.progress((i + 1) / len(shift_range))
    
    progress_bar.empty()
    return pd.DataFrame(results)

# --- PLOTTING HELPER ---

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["Down", "Up"], columns=["Down", "Up"])
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    return fig

def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0,1], [0,1], color="navy", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return fig

def plot_distribution(y_prob, title="Prediction Distribution"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y_prob, bins=30, color="grey", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Frequency")
    return fig

# --- HAUPTBEREICH ---
st.title(f"Quant Analysis: {MODEL_TYPE}")

if st.sidebar.button("🚀 Analyse Starten"):
    df_raw = get_data(TICKER, INTERVAL, LOOKBACK)
    
    if df_raw is not None:
        df = add_indicators(df_raw)
        
        # --- MODUS 1: LINEARE REGREESSION ---
        if MODEL_TYPE == "Lineare Regression (OLS)":
            SHIFT_OLS = st.sidebar.slider("Manueller Shift", 1, 30, 5)
            df[f"Target"] = (df["Close"].shift(-SHIFT_OLS) - df["Close"]) / df["Close"] * 100
            df_final = df.iloc[::SHIFT_OLS].dropna()
            
            X = sm.add_constant(df_final[STRATEGY])
            y = df_final["Target"]
            model = sm.OLS(y, X).fit()
            
            st.subheader("OLS Modell Zusammenfassung")
            st.text(model.summary())
            
            # Annahmen Graphen
            st.subheader("Statistische Validierung")
            preds = model.predict(X)
            resids = y - preds
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**1. Linearität**")
                fig, ax = plt.subplots()
                ax.scatter(preds, resids, alpha=0.5)
                ax.axhline(0, color='red', linestyle='--')
                st.pyplot(fig)
            with c2:
                st.write("**2. Unabhängigkeit**")
                fig, ax = plt.subplots()
                lag_plot(resids, ax=ax)
                st.pyplot(fig)
            with c3:
                st.write("**3. Normalverteilung**")
                fig, ax = plt.subplots()
                ax.hist(resids, bins=30, edgecolor="black")
                st.pyplot(fig)

        # --- MODUS 2: LOGISTISCHE REGRESSION ---
        else:
            st.subheader("Logistische Regression & Optimierung")
            
            # 1. Train Test Split (vor Target, um Data Leakage zu minimieren bei Features, aber Target kommt gleich)
            # Da Target vom Shift abhängt, machen wir den Split erst nach der Shift-Analyse oder wir nutzen Train für Shift-Analyse.
            # Sauberer Weg: Wir nutzen df für Shift-Suche, dann splitten wir.
            
            # A) AUC Optimierung auf dem gesamten Datensatz (oder Train)
            st.write("🔍 Suche optimalen Shift (Holding Period)...")
            auc_df = explore_shift_auc(df, STRATEGY)
            
            if not auc_df.empty:
                best_shift = int(auc_df.sort_values(by="AUC", ascending=False).iloc[0]["Shift"])
                
                # PLOT 1: AUC by Shift (nur einmal, da global für Modellfindung)
                st.write(f"### 1. Shift Optimierung (Optimal: {best_shift})")
                fig_auc, ax_auc = plt.subplots(figsize=(10, 4))
                ax_auc.plot(auc_df["Shift"], auc_df["AUC"], marker="o", color="purple")
                ax_auc.set_title(f"AUC Score über verschiedene Shifts ({TICKER})")
                ax_auc.set_xlabel("Shift (Perioden)")
                ax_auc.set_ylabel("AUC Score")
                ax_auc.grid(True)
                st.pyplot(fig_auc)
                
                # B) Datensatz mit optimalem Shift erstellen
                df_target = add_target(df, best_shift)
                
                # C) Train/Test Split
                train_df, test_df = train_test_split_func(df_target)
                st.info(f"Datensatz Split: {len(train_df)} Training | {len(test_df)} Testing")
                
                # D) Modell Training
                model_train, y_train, y_prob_train = generate_logit_model(train_df, STRATEGY)
                
                # E) Modell Testing (auf Testdaten anwenden)
                X_test = sm.add_constant(test_df[STRATEGY])
                y_test = test_df["Target"]
                y_prob_test = model_train.predict(X_test)
                
                # F) Die 4 Vergleichsgraphen
                st.divider()
                st.subheader("Modell Evaluation: Training vs. Testing")
                
                col_train, col_test = st.columns(2)
                
                # --- TRAINING SPALTE ---
                with col_train:
                    st.markdown("### 🏋️ Training Set")
                    st.write("**2. Prediction Distribution**")
                    st.pyplot(plot_distribution(y_prob_train, "Train: Predictions"))
                    
                    st.write("**3. ROC Curve**")
                    st.pyplot(plot_roc_curve(y_train, y_prob_train, "Train: ROC"))
                    
                    st.write("**4. Confusion Matrix**")
                    y_pred_train = (y_prob_train > 0.5).astype(int)
                    st.pyplot(plot_confusion_matrix(y_train, y_pred_train, "Train: Matrix"))

                # --- TESTING SPALTE ---
                with col_test:
                    st.markdown("### 🧪 Test Set (Unseen)")
                    st.write("**2. Prediction Distribution**")
                    st.pyplot(plot_distribution(y_prob_test, "Test: Predictions"))
                    
                    st.write("**3. ROC Curve**")
                    st.pyplot(plot_roc_curve(y_test, y_prob_test, "Test: ROC"))
                    
                    st.write("**4. Confusion Matrix**")
                    y_pred_test = (y_prob_test > 0.5).astype(int)
                    st.pyplot(plot_confusion_matrix(y_test, y_pred_test, "Test: Matrix"))
                
                st.success("Analyse abgeschlossen.")
            else:
                st.error("Keine validen AUC Scores gefunden.")

    else:
        st.error("Keine Daten geladen.")
else:
    st.info("Parameter einstellen und 'Analyse Starten' klicken.")
import streamlit as st
import numpy as np
import pandas as pd
from pandas.plotting import lag_plot
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="Quant Regression Analysis", layout="wide")

st.title("📈 Quantitative Regressions-Analyse")
st.markdown("""
Diese App führt eine OLS-Regression durch, um die Beziehung zwischen technischen Indikatoren 
(wie Fair Value Gaps & MACD) und zukünftigen Preisänderungen zu untersuchen.
""")

# --- SIDEBAR: BENUTZER-INPUTS (GLOBALE VARIABLEN) ---
st.sidebar.header("⚙️ Einstellungen")

# Schritt 1: Ticker
TICKER = st.sidebar.text_input("Ticker Symbol", value="SPY")

# Schritt 2: Zeitraum
INTERVAL = st.sidebar.selectbox("Intervall", options=["1d", "1h", "1m"], index=0)
if INTERVAL == "1h":
    PERIOD = "730d"
else:
    PERIOD = "max"

# Schritt 3: Strategie (Features) auswählen
available_features = ["Close", "Volume", "Both_FVG", "MACD_HIST"]
STRATEGY = st.sidebar.multiselect(
    "Unabhängige Variablen (Features)", 
    options=available_features, 
    default=["Both_FVG", "MACD_HIST"]
)

# MACD Einstellungen (Optional in Expander verstecken, um es sauber zu halten)
with st.sidebar.expander("MACD Einstellungen"):
    MACD_FAST = st.number_input("Fast EMA", value=12)
    MACD_SLOW = st.number_input("Slow EMA", value=27)
    MACD_SPAN = st.number_input("Signal Span", value=9)

# Schritt 4: Shift (Vorhersage-Horizont)
SHIFT = st.sidebar.number_input("Shift (Tage in die Zukunft)", min_value=1, value=1)

# Schritt 5: Lookback
LOOKBACK = st.sidebar.slider("Lookback (Anzahl Zeilen)", min_value=100, max_value=20000, value=10000, step=100)

# --- FUNKTIONEN (Angepasst für Streamlit) ---

@st.cache_data # WICHTIG: Verhindert ständiges Neuladen der Daten
def get_data(ticker, interval, period, lookback):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        # Handle MultiIndex columns if necessary
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index(drop=True)
        return df.iloc[-lookback:, :]
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")
        return pd.DataFrame()

def add_target(df, shift):
    # Zielvariable: Prozentuale Änderung in 'shift' Tagen
    df["Target"] = (df["Close"].shift(-shift) - df["Close"]) / df["Close"] * 100
    return df

def bull_fvg(df):
    df['High_2prev'] = df['High'].shift(2)
    df['Bull_FVG'] = (df['Low'] > df['High_2prev']).astype(int)
    df['Bull_FVG_Val'] = (df['Low'] - df['High_2prev']) * df['Bull_FVG'] / df['Close']
    return df

def bear_fvg(df):
    df['Low_2prev'] = df['Low'].shift(2)
    df['Bear_FVG'] = (df['High'] < df['Low_2prev']).astype(int)
    df['Bear_FVG_Val'] = (df['High'] - df['Low_2prev']) * df['Bear_FVG'] / df['Close']
    return df

def add_MACD(df, fast, slow, span):
    df[f"{fast}_ema"] = df["Close"].ewm(span=fast).mean()
    df[f"{slow}_ema"] = df["Close"].ewm(span=slow).mean()
    df["MACD"] = df[f"{fast}_ema"] - df[f"{slow}_ema"]
    df["Signal"] = df["MACD"].ewm(span=span).mean()
    df["MACD_HIST"] = df["MACD"] - df["Signal"]
    return df

def prepare_df_for_regression(df):
    # Logik: Filtern nach FVG Existenz
    df = df[(df['Bull_FVG'] == 1) | (df['Bear_FVG'] == 1)].copy()
    df['Both_FVG'] = df['Bear_FVG_Val'] + df['Bull_FVG_Val']
    return df

def run_regression_and_plot(df, features, target="Target"):
    # Daten bereinigen
    subset = df[features + [target]].dropna()
    
    if subset.empty:
        st.warning("Nicht genügend Daten nach dem Bereinigen (dropna). Bitte Parameter prüfen.")
        return None, None, None, None

    X = subset[features]
    y = subset[target]
    
    # Modell erstellen
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    
    # Ergebnisse
    intercept = model.params['const']
    coefficients = model.params.drop('const')
    model_p_value = model.f_pvalue
    y_pred = model.predict(X_with_const)
    
    # Ergebnisse in Streamlit anzeigen
    st.subheader("1. Regressions-Ergebnisse (OLS)")
    
    # Summary als Textbox
    st.text(model.summary())
    
    # Wichtige Metriken hervorheben
    col1, col2, col3 = st.columns(3)
    col1.metric("R-Squared", f"{model.rsquared:.4f}")
    col2.metric("P-Value (Model)", f"{model_p_value:.6f}")
    col3.metric("Beobachtungen", f"{len(subset)}")

    # Plot Actual vs Predicted
    st.subheader("2. Vorhersage vs. Realität")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_pred, y, alpha=0.6, label="Datenpunkte")
    # 45-Grad Linie (Perfekte Vorhersage)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Ideallinie (y=x)")
    ax.set_xlabel("Vorhersage (Predicted)")
    ax.set_ylabel("Tatsächlich (Actual)")
    ax.set_title("Actual vs. Predicted")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig) # Streamlit Plot Befehl

    return df, intercept, coefficients, y_pred

def validate_model_plots(df, coef, intercept, features):
    # Vorhersage berechnen (manuell zur Validierung)
    # Hinweis: Da wir mehrere Features haben können, ist die manuelle Berechnung komplexer.
    # Wir nutzen hier vereinfacht die Residuen, die wir aus der Regression ableiten könnten,
    # aber um dem Originalcode treu zu bleiben, berechnen wir es hier neu basierend auf den Features.
    
    # Achtung: Wir müssen sicherstellen, dass wir auf dem gleichen Subset arbeiten
    # Um Fehler zu vermeiden, berechnen wir Predictions direkt im Regressionsschritt (siehe oben)
    # und nutzen hier nur die Visualisierung, wenn df 'Predictions' hat.
    
    # Wir berechnen Predictions neu für den ganzen DF (wo möglich)
    df["Predictions"] = intercept
    for feature in features:
        if feature in coef.index:
            df["Predictions"] += df[feature] * coef[feature]
            
    df["Residuals"] = df["Target"] - df["Predictions"]
    
    # Wir droppen NaNs für die Plots
    df_clean = df.dropna(subset=["Residuals", "Predictions"])

    st.subheader("3. Validierung & Residuen Analyse")
    
    tab1, tab2, tab3 = st.tabs(["Linearität", "Unabhängigkeit", "Normalität"])

    with tab1:
        st.markdown("**Test auf Linearität & Homoskedastizität**")
        fig1, ax1 = plt.subplots()
        ax1.scatter(df_clean['Predictions'], df_clean['Residuals'], alpha=0.5)
        ax1.axhline(0, color='red', linestyle='--')
        ax1.set_xlabel("Predicted Values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs. Predictions")
        st.pyplot(fig1)

    with tab2:
        st.markdown("**Test auf Unabhängigkeit (Autokorrelation)**")
        fig2, ax2 = plt.subplots()
        lag_plot(df_clean['Residuals'], ax=ax2)
        ax2.set_title("Lag Plot of Residuals")
        st.pyplot(fig2)

    with tab3:
        st.markdown("**Test auf Normalverteilung**")
        fig3, ax3 = plt.subplots()
        ax3.hist(df_clean['Residuals'], bins=50, edgecolor='k')
        ax3.set_title("Histogram of Residuals")
        st.pyplot(fig3)

# --- HAUPT-EXECUTION (MAIN) ---

def main():
    # Prüfen ob Features ausgewählt sind
    if not STRATEGY:
        st.warning("Bitte wählen Sie mindestens ein Feature in der Sidebar aus.")
        return

    # Daten laden
    with st.spinner('Lade Daten...'):
        df = get_data(TICKER, INTERVAL, PERIOD, LOOKBACK)
    
    if df.empty:
        return

    # Indikatoren berechnen
    df = add_target(df, SHIFT)
    df = bull_fvg(df)
    df = bear_fvg(df)
    df = add_MACD(df, MACD_FAST, MACD_SLOW, MACD_SPAN)
    
    # Daten für Regression vorbereiten (Filtert nach FVG!)
    # Hinweis: Wenn Features gewählt sind, die nichts mit FVG zu tun haben, 
    # reduziert dieser Schritt trotzdem die Datenmenge drastisch.
    # Wir lassen es hier so, wie im Originalcode gewünscht.
    df = prepare_df_for_regression(df)
    
    # Raw Data Preview (Optional)
    with st.expander("Rohdaten anzeigen"):
        st.dataframe(df.tail())

    # Regression durchführen
    df_res, intercept, coef, _ = run_regression_and_plot(df, STRATEGY, target="Target")
    
    if df_res is not None:
        # Validierung
        validate_model_plots(df_res, coef, intercept, STRATEGY)

if __name__ == "__main__":
    main()


### Anleitung zum Starten der App (Stop Terminal: Ctr + c)

# 1.  **Installation:** Stellen Sie sicher, dass Sie `streamlit` installiert haben:
#     pip install streamlit statsmodels yfinance matplotlib pandas
# 2.  **Datei speichern:** Speichern Sie den obigen Code in einer Datei namens `app.py`.
# 3.  **App starten:** Öffnen Sie Ihr Terminal, navigieren Sie zu dem Ordner, in dem die Datei liegt, und führen Sie aus:
#     streamlit run app.py
# 4. Aktuellste version alle Libraries
#    conda update -c conda-forge pyarrow streamlit protobuf
#    conda install -c conda-forge --force-reinstall pyarrow streamlit protobuf
#    conda install -c conda-forge frozendict
#    conda install -c conda-forge multitasking
#    conda install -c conda-forge --force-reinstall yfinance
#    conda install -c conda-forge --force-reinstall pyarrow
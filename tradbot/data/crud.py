"""
TradBot CRUD Operationen

Einfache Funktionen zum Speichern/Laden zwischen pandas und SQLite.
Integriert mit bestehenden Modulen (yfinance, risk metrics, portfolio).

Verwendung:
    from tradbot.data import save_market_data, load_market_data

    # Nach yf.download
    df = yf.download("AAPL", ...)
    save_market_data(df, ticker="AAPL")

    # Später laden
    df = load_market_data("AAPL")
"""

import pandas as pd
from datetime import datetime
from typing import Optional, Dict

from .database import engine, Session, MarketData, Trade, Signal, PortfolioSnapshot, PerformanceMetric


# =============================================================================
# MARKET DATA
# =============================================================================

def save_market_data(df: pd.DataFrame, ticker: str, interval: str = "1d") -> int:
    """
    Speichert OHLCV DataFrame in Datenbank.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mit Spalten: Open, High, Low, Close, Volume
        Index sollte DatetimeIndex sein
    ticker : str
        Aktien-Symbol (z.B. 'AAPL')
    interval : str
        Daten-Interval ('1d', '1h', '15m', '5m')

    Returns
    -------
    int : Anzahl eingefügter Zeilen

    Beispiel
    --------
        df = yf.download("AAPL", period="1y")
        save_market_data(df, "AAPL")
    """
    df = df.copy()

    # Handle MultiIndex columns von yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset index wenn DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df.rename(columns={'index': 'Date', 'Datetime': 'Date'}, inplace=True)

    # Metadata hinzufügen
    df['ticker'] = ticker
    df['interval'] = interval

    # Spalten umbenennen für Model
    df = df.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # Nur benötigte Spalten
    columns = ['ticker', 'date', 'interval', 'open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in columns if c in df.columns]]

    # pandas to_sql nutzen (einfach und effizient)
    df.to_sql('market_data', engine, if_exists='append', index=False)

    return len(df)


def load_market_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Lädt OHLCV Daten aus Datenbank.

    Parameters
    ----------
    ticker : str
        Aktien-Symbol
    start_date : str, optional
        Start Datum 'YYYY-MM-DD'
    end_date : str, optional
        End Datum 'YYYY-MM-DD'
    interval : str
        Daten-Interval

    Returns
    -------
    pd.DataFrame : OHLCV Daten mit DatetimeIndex

    Beispiel
    --------
        df = load_market_data("AAPL", start_date="2024-01-01")
    """
    query = f"""
        SELECT date, open, high, low, close, volume
        FROM market_data
        WHERE ticker = '{ticker}' AND interval = '{interval}'
    """

    if start_date:
        query += f" AND date >= '{start_date}'"
    if end_date:
        query += f" AND date <= '{end_date}'"

    query += " ORDER BY date"

    df = pd.read_sql(query, engine, parse_dates=['date'])
    df = df.set_index('date')

    # Spalten groß schreiben für yfinance Format
    df.columns = [c.capitalize() for c in df.columns]

    return df


# =============================================================================
# TRADES
# =============================================================================

def save_trade(
    ticker: str,
    side: str,
    quantity: float,
    price: float,
    signal_id: Optional[int] = None,
    pnl: Optional[float] = None
) -> int:
    """
    Speichert einen Trade Record.

    Returns
    -------
    int : Trade ID
    """
    session = Session()

    trade = Trade(
        ticker=ticker,
        side=side.upper(),
        quantity=quantity,
        price=price,
        signal_id=signal_id,
        pnl=pnl
    )

    session.add(trade)
    session.commit()
    trade_id = trade.id
    session.close()

    return trade_id


def load_trades(ticker: Optional[str] = None) -> pd.DataFrame:
    """Lädt Trades als DataFrame."""
    query = "SELECT * FROM trades"
    if ticker:
        query += f" WHERE ticker = '{ticker}'"
    query += " ORDER BY timestamp DESC"

    return pd.read_sql(query, engine, parse_dates=['timestamp'])


# =============================================================================
# SIGNALS
# =============================================================================

def save_signal(
    ticker: str,
    signal_type: str,
    source: str,
    strength: float = 1.0,
    indicators: Optional[Dict[str, float]] = None
) -> int:
    """
    Speichert ein Trading Signal.

    Parameters
    ----------
    ticker : str
        Aktien-Symbol
    signal_type : str
        'BUY', 'SELL', oder 'HOLD'
    source : str
        Signal-Quelle ('RSI', 'MACD', 'LogReg', etc.)
    strength : float
        Konfidenz-Level 0.0-1.0
    indicators : dict, optional
        Indikator-Werte zum Signal-Zeitpunkt
        {'rsi': 25.5, 'macd_hist': 0.5, 'mfi': 30, 'bb': 0.2}

    Returns
    -------
    int : Signal ID
    """
    session = Session()

    signal = Signal(
        ticker=ticker,
        signal_type=signal_type.upper(),
        source=source,
        strength=strength
    )

    # Indikator-Werte hinzufügen wenn vorhanden
    if indicators:
        signal.rsi = indicators.get('rsi') or indicators.get('RSI')
        signal.macd_hist = indicators.get('macd_hist') or indicators.get('MACD_HIST')
        signal.mfi = indicators.get('mfi') or indicators.get('MFI')
        signal.bb = indicators.get('bb') or indicators.get('BB')

    session.add(signal)
    session.commit()
    signal_id = signal.id
    session.close()

    return signal_id


def load_signals(ticker: Optional[str] = None, source: Optional[str] = None) -> pd.DataFrame:
    """Lädt Signals als DataFrame."""
    query = "SELECT * FROM signals WHERE 1=1"
    if ticker:
        query += f" AND ticker = '{ticker}'"
    if source:
        query += f" AND source = '{source}'"
    query += " ORDER BY timestamp DESC"

    return pd.read_sql(query, engine, parse_dates=['timestamp'])


# =============================================================================
# PORTFOLIO SNAPSHOTS
# =============================================================================

def save_portfolio_snapshot(
    weights: Dict[str, float],
    total_value: float,
    cash: float = 0.0,
    strategy: str = "manual"
) -> int:
    """
    Speichert Portfolio Snapshot.

    Parameters
    ----------
    weights : dict
        Portfolio Gewichte {'AAPL': 0.4, 'MSFT': 0.6}
    total_value : float
        Gesamt Portfolio-Wert
    cash : float
        Cash Position
    strategy : str
        Optimierungs-Strategie
    """
    session = Session()

    # Weights dict zu String konvertieren
    weights_str = ",".join([f"{k}:{v:.4f}" for k, v in weights.items() if v > 0.001])

    snapshot = PortfolioSnapshot(
        weights=weights_str,
        total_value=total_value,
        cash=cash,
        strategy=strategy
    )

    session.add(snapshot)
    session.commit()
    snapshot_id = snapshot.id
    session.close()

    return snapshot_id


def load_portfolio_history() -> pd.DataFrame:
    """Lädt Portfolio Snapshots als DataFrame."""
    query = "SELECT * FROM portfolio_snapshots ORDER BY timestamp"
    return pd.read_sql(query, engine, parse_dates=['timestamp'])


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def save_performance_metrics(
    sharpe_ratio: float,
    max_drawdown: float,
    var_95: float,
    var_99: float = None,
    ticker: Optional[str] = None,
    annualized_return: float = None,
    annualized_volatility: float = None
) -> int:
    """
    Speichert Performance Metrics Snapshot.

    Nutze ticker=None für Portfolio-Level Metriken.
    """
    session = Session()

    metric = PerformanceMetric(
        ticker=ticker,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        var_95=var_95,
        var_99=var_99,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility
    )

    session.add(metric)
    session.commit()
    metric_id = metric.id
    session.close()

    return metric_id


def load_performance_history(ticker: Optional[str] = None) -> pd.DataFrame:
    """Lädt Performance Metrics History."""
    query = "SELECT * FROM performance_metrics WHERE 1=1"
    if ticker:
        query += f" AND ticker = '{ticker}'"
    elif ticker is None:
        pass  # Alle laden
    query += " ORDER BY timestamp"

    return pd.read_sql(query, engine, parse_dates=['timestamp'])

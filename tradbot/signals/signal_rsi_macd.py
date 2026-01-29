"""
Signal Generator - Nutzt bestehende TechnicalIndicators für Buy/Sell Signale.

Funktioniert mit Historical Data (keine IB Subscription nötig).
"""

import numpy as np
import pandas as pd
from tradbot.strategy import TechnicalIndicators


def generate_signal(df: pd.DataFrame) -> dict:
    """
    Generiert ein Trading-Signal basierend auf RSI + MACD.

    Args:
        df: OHLCV DataFrame (von yfinance oder TWS historical)

    Returns:
        dict mit signal, reason und indicators
    """
    ti = TechnicalIndicators(df.copy())
    ti.add_rsi().add_macd()
    data = ti.dropna().get_df()

    if data.empty:
        return {"signal": "HOLD", "reason": "Nicht genug Daten"}

    last = data.iloc[-1]

    rsi = last["RSI"]
    macd_hist = last["MACD_HIST"]

    # BUY: RSI oversold + MACD dreht positiv
    if rsi < 30 and macd_hist > 0:
        return {
            "signal": "BUY",
            "reason": f"RSI oversold ({rsi:.1f}) + MACD bullish ({macd_hist:.4f})",
            "rsi": rsi,
            "macd_hist": macd_hist,
        }

    # SELL: RSI overbought + MACD dreht negativ
    if rsi > 70 and macd_hist < 0:
        return {
            "signal": "SELL",
            "reason": f"RSI overbought ({rsi:.1f}) + MACD bearish ({macd_hist:.4f})",
            "rsi": rsi,
            "macd_hist": macd_hist,
        }

    return {
        "signal": "HOLD",
        "reason": f"Kein klares Signal (RSI={rsi:.1f}, MACD_HIST={macd_hist:.4f})",
        "rsi": rsi,
        "macd_hist": macd_hist,
    }


def generate_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized RSI+MACD strategy for backtesting.

    Returns DataFrame with 'Strategy' column:
        +1 = Long (RSI < 30 and MACD_HIST > 0)
        -1 = Short (RSI > 70 and MACD_HIST < 0)
         0 = Flat

    Position is shifted by 1 bar to avoid look-ahead bias.
    """
    ti = TechnicalIndicators(df.copy())
    ti.add_rsi().add_macd()
    result = ti.dropna().get_df()

    result["Strategy"] = np.where(
        (result["RSI"] < 30) & (result["MACD_HIST"] > 0), 1,
        np.where(
            (result["RSI"] > 70) & (result["MACD_HIST"] < 0), -1, 0
        )
    )

    result["Strategy"] = result["Strategy"].shift(1)
    return result.dropna(subset=["Strategy"])

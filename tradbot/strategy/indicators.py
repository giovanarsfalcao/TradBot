"""
Technical Indicators Module

Chainable class for adding technical indicators to price data.
Based on notebooks/indicators/
"""

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """
    Chainable Technical Indicators.

    Usage:
        ti = TechnicalIndicators(df)
        ti.add_macd().add_rsi().add_mfi().add_bb()
        result = ti.get_df()
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: Open, High, Low, Close, Volume
        """
        self.df = df.copy()

    def get_df(self) -> pd.DataFrame:
        """Return the DataFrame with all added indicators."""
        return self.df

    def add_macd(self, fast: int = 12, slow: int = 27, span: int = 9):
        """
        Add MACD (Moving Average Convergence Divergence).

        Columns added: MACD, Signal, MACD_HIST

        MACD_HIST > 0 → Bullish Momentum
        MACD_HIST < 0 → Bearish Momentum
        """
        df = self.df

        # Fast & Slow EMA
        df[f"{fast}_ema"] = df["Close"].ewm(span=fast).mean()
        df[f"{slow}_ema"] = df["Close"].ewm(span=slow).mean()

        # MACD Line = Fast EMA - Slow EMA
        df["MACD"] = df[f"{fast}_ema"] - df[f"{slow}_ema"]

        # Signal Line = EMA of MACD
        df["Signal"] = df["MACD"].ewm(span=span).mean()

        # Histogram = MACD - Signal
        df["MACD_HIST"] = df["MACD"] - df["Signal"]

        return self

    def add_rsi(self, length: int = 14):
        """
        Add RSI (Relative Strength Index).

        Columns added: RSI

        RSI > 70 → Overbought (Sell Signal)
        RSI < 30 → Oversold (Buy Signal)
        """
        df = self.df

        # Price Change (Delta)
        price_change = df["Close"].diff()

        # Separate Gains & Losses
        df["gain"] = price_change.where(price_change > 0, 0)
        df["loss"] = -price_change.where(price_change < 0, 0)

        # Rolling Average
        df["avg_gain"] = df["gain"].rolling(window=length).mean()
        df["avg_loss"] = df["loss"].rolling(window=length).mean()

        # Relative Strength
        rs = df["avg_gain"] / df["avg_loss"]

        # RSI (0-100)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Cleanup helper columns
        df.drop(columns=["gain", "loss", "avg_gain", "avg_loss"], inplace=True)

        return self

    def add_mfi(self, length: int = 14):
        """
        Add MFI (Money Flow Index).

        Columns added: MFI

        MFI > 70 → Overbought
        MFI < 30 → Oversold
        """
        df = self.df

        # Typical Price
        tp = (df["High"] + df["Low"] + df["Close"]) / 3

        # Money Flow
        mf = tp * df["Volume"]

        # Positive & Negative Flow
        pos_flow = np.where(tp.diff() > 0, mf, 0)
        neg_flow = np.where(tp.diff() < 0, mf, 0)

        # Money Flow Ratio
        mfr = pd.Series(pos_flow).rolling(length).sum() / pd.Series(neg_flow).rolling(length).sum()

        # MFI (0-100)
        df["MFI"] = 100 - (100 / (1 + mfr.values))

        return self

    def add_bb(self, length: int = 20, std_dev: int = 2):
        """
        Add Bollinger Bands.

        Columns added: BB_SMA, Upper, Lower, BB

        BB (normalized): 0 = at Lower Band, 1 = at Upper Band
        """
        df = self.df

        # Middle Band (SMA)
        df["BB_SMA"] = df["Close"].rolling(window=length).mean()

        # Standard Deviation
        df["BB_STD"] = df["Close"].rolling(window=length).std()

        # Upper & Lower Bands
        df["Upper"] = df["BB_SMA"] + (std_dev * df["BB_STD"])
        df["Lower"] = df["BB_SMA"] - (std_dev * df["BB_STD"])

        # Normalized BB (0-1 range)
        df["BB"] = (df["Close"] - df["Lower"]) / (df["Upper"] - df["Lower"])

        # Cleanup
        df.drop(columns=["BB_STD"], inplace=True)

        return self

    def add_all(
        self,
        macd_fast: int = 12,
        macd_slow: int = 27,
        macd_span: int = 9,
        rsi_length: int = 14,
        mfi_length: int = 14,
        bb_length: int = 20,
        bb_std: int = 2
    ):
        """Add all indicators at once."""
        return (
            self
            .add_macd(macd_fast, macd_slow, macd_span)
            .add_rsi(rsi_length)
            .add_mfi(mfi_length)
            .add_bb(bb_length, bb_std)
        )

    def dropna(self):
        """Drop rows with NaN values."""
        self.df = self.df.dropna()
        return self

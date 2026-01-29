"""
Logistic Regression Signal Generator

Generates BUY/SELL/HOLD signals using LogisticRegression
probability output with technical indicators as features.
"""

import pandas as pd
import numpy as np
from tradbot.strategy import TechnicalIndicators
from tradbot.strategy.models import LogisticRegression

FEATURES = ["MACD_HIST", "RSI", "MFI", "BB"]


def generate_signal(df: pd.DataFrame, shift: int = 5,
                    threshold: float = 0.55) -> dict:
    """
    Generate trading signal using Logistic Regression.

    Args:
        df: OHLCV DataFrame (needs enough rows for training)
        shift: Prediction horizon in bars
        threshold: Probability threshold for BUY/SELL (default 0.55)

    Returns:
        dict with: signal, probability, auc, shift, reason
    """
    # Add all indicators
    ti = TechnicalIndicators(df.copy())
    ti.add_all()
    data = ti.dropna().get_df()

    if len(data) < 100:
        return {
            "signal": "HOLD",
            "reason": "Not enough data for LogReg",
            "probability": None,
            "auc": None,
            "shift": shift,
        }

    # Setup LogReg
    logreg = LogisticRegression(data, features=FEATURES)
    logreg.add_target(shift=shift)

    # Temporal split (no look-ahead bias)
    train_df, test_df = logreg.train_test_split(train_size=0.7)

    # Fit on training data
    try:
        fit_result = logreg.fit(train_df)
    except Exception:
        return {
            "signal": "HOLD",
            "reason": "LogReg fit failed",
            "probability": None,
            "auc": None,
            "shift": shift,
        }

    auc = fit_result["auc"]

    # Predict on latest data point
    latest = data.tail(1)
    prob = logreg.predict(latest)

    if prob.empty:
        return {
            "signal": "HOLD",
            "reason": "Prediction failed",
            "probability": None,
            "auc": auc,
            "shift": shift,
        }

    probability = float(prob.iloc[0])

    # Convert probability to signal
    if probability > threshold:
        signal = "BUY"
        reason = f"LogReg P(Up)={probability:.3f} > {threshold} (AUC={auc:.3f})"
    elif probability < (1 - threshold):
        signal = "SELL"
        reason = f"LogReg P(Up)={probability:.3f} < {1 - threshold:.2f} (AUC={auc:.3f})"
    else:
        signal = "HOLD"
        reason = f"LogReg P(Up)={probability:.3f} neutral zone (AUC={auc:.3f})"

    return {
        "signal": signal,
        "probability": probability,
        "auc": auc,
        "shift": shift,
        "reason": reason,
    }


def generate_strategy(df: pd.DataFrame, shift: int = 5,
                      threshold: float = 0.55,
                      train_size: float = 0.7) -> pd.DataFrame:
    """
    Vectorized LogReg strategy for backtesting.

    Train on first train_size fraction, predict on the rest.
    Train period has Strategy=0 (no trading). Only test period has real positions.

    Returns DataFrame with 'Strategy' column:
        +1 = Long  (P(Up) > threshold)
        -1 = Short (P(Up) < 1 - threshold)
         0 = Flat

    Position is shifted by 1 bar to avoid look-ahead bias.
    """
    ti = TechnicalIndicators(df.copy())
    ti.add_all()
    data = ti.dropna().get_df()

    if len(data) < 100:
        data["Strategy"] = 0
        return data

    logreg = LogisticRegression(data, features=FEATURES)
    logreg.add_target(shift=shift)

    train_df, test_df = logreg.train_test_split(train_size=train_size)

    try:
        logreg.fit(train_df)
    except Exception:
        logreg.df["Strategy"] = 0
        return logreg.df

    # Predict on test data only (no look-ahead)
    probs = logreg.predict(test_df)

    # Build positions aligned by index
    result = logreg.df.copy()
    result["Strategy"] = 0

    test_positions = pd.Series(0, index=test_df.index)
    test_positions.loc[probs.index] = np.where(
        probs > threshold, 1,
        np.where(probs < (1 - threshold), -1, 0)
    )
    result.loc[test_df.index, "Strategy"] = test_positions

    result["Strategy"] = result["Strategy"].shift(1)
    return result.dropna(subset=["Strategy"])

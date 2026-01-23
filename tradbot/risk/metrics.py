"""
Risk Metrics Module:

Implementation of common risk metrics:
- Sharpe Ratio
- Max Drawdown
- Value at Risk (VaR)
"""

import numpy as np 
import pandas as pd
from typing import Union

"""
Sharpe Ratio: measures risk-adjusted returns by comparing excess
to the volatility of those returns. 
- Beantwortet die Frage: Wie viel Überschussrendite erzielen wir pro Einheit Risiko?
- Typischer Antwort: „Unsere Sharpe Ratio beträgt 1,5.“

Sharpe Ratio = (Mean Return - Risk Free Rate)/Standard Deviation of Returns

Parametern:
- returns: pd.Series ([0.01, -0.02, 0.03]) or np.ndarray
- risk_free_rate: float (default=0.02)
- periods_per_year: int
- sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02))

Ergebnis:
- sharpe: float

Implementieren:
- Durchschnittlichen Return berechnen: mean_return()
- Annualisieren: annualized_return = mean_return * periods_per_year
- Standardabweichung berechnen: std()
- Annualisieren: annualized_std =std * np. sqrt(periods_per_year)
- Sharpe = (annualized_return - risk_free_rate) / annualized_std
"""

def calculate_sharpe_ratio(
        returns: Union[pd.Series, np.ndarray],
        risk_free_rate: float=0.02,
        periods_per_year: int=252
) -> float:

    mean_return = returns.mean()
    annualized_return = mean_return * periods_per_year
    std = returns.std()
    annualized_std = std * np.sqrt(periods_per_year)
    annualized_sharpe = (annualized_return - risk_free_rate) / annualized_std
    return annualized_sharpe

"""
Maximum Drawdown: measures the largest peak-to-through decline
in the portfolio value. It measures the largest loss from a historical peak.
- Beantwortet die Frage: Wie viel Geld könnten wir höchstens verlieren, wenn wir zu einem Höchststand kaufen und zu einem Tiefststand verkaufen?
- Typischer Antwort: „Unser Maximum Drawdown beträgt -20 %.“

Max Drawdown = (Through Value - Peak Value ) / Peak Value 

Parametern:
- prices: pd.Series ([100, 95, 90, 85, 80, 90, 95, 100]) or np.ndarray

Ergebnis:
- max_drawdown: float (negative))

Implementieren:
- Cumulative Maximum berechnen: cumulative_max = prices.cummax()
- Rollierende Maximum: rolling_max = prices.cummax() oder np.maximum.accumulate()
- Drawdown berechnen:  drawdown = (prices - rolling_max) / rolling_max
- Max Drawdown: max_drawdown = drawdown.min()
"""

def calculate_max_drawdown(
       prices: Union[pd.Series, np.ndarray] 
) -> float:
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    return max_drawdown

"""
Historical Value at Risk (VaR): estimated the maximum loss 
- Beantwortet die Frage: Wenn es morgen an der Börse schlecht läuft, wie viel Geld verlieren wir dann höchstens?
- Typischer Antwort: „Unser 1-Tages-VaR liegt bei 1.000 € mit einer Wahrscheinlichkeit von 95 %.“
- Erklärung: „Wir sind uns zu 95 % sicher, dass wir morgen nicht mehr als 1.000 € verlieren werden.“
Oder andersherum: „Nur in 5 % der Fälle (also an etwa einem Tag pro Monat) wird der Verlust schlimmer als 1.000 € sein.“

Parametern:
- returns: pd.Series ([0.01, -0.02, 0.03]) or np.ndarray
- confidence_level: float (default=0.95)

Ergebnis:
- var: float (negative, -0.05 means 5% loss at risk)

Implementieren:
- VaR ist das Quantil der Verlustverteilung: var = np.percentile(returns, (1 - confidence_level) * 100)
- Bei 95 % Konfidenzniveau: var = np.percentile(returns, 5)
- alpha = 1 - confidence_level
"""

def calculate_var(
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float=0.95
) -> float:
    alpha = 1 - confidence_level
    var = np.percentile(returns, alpha * 100)
    return var

"""
Calculate all risk metrics at once.

Ziel: Convenience function um alle Metrics auf einmal zu berechnen und 
die Ergebnisse in einem Dictionary zurückgegeben. 

Parametern:
- returns: pd.Series or np.ndarray
- risk_free_rate: float, default = 0.02
- periods_per_year: int, default = 252
- prices: pd.Series or np.ndarray
- confidence_level: float, default = 0.95

Ergebnis/Return: dict

Implementieren:
- Alle Funktionen aufrufen 
- Ergebnisse in einem Dictionary speichern
"""

def calculate_all_risk_metrics(
        returns: Union[pd.Series, np.ndarray],
        prices: Union[pd.Series, np.ndarray],
        risk_free_rate: float=0.02,
        periods_per_year: int=252,
        confidence_level: float=0.95
)-> dict:
    sharpe = calculate_sharpe_ratio(
        returns, 
        risk_free_rate,
        periods_per_year
    )
    max_drawdown = calculate_max_drawdown(
        prices
    )
    var = calculate_var(
        returns,
        confidence_level
    )

    all_metrics = {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "var": var
    }

    return all_metrics

"""
Calculate Annualized Returns:

Parametern:
- returns: pd.Series or np.ndarray
- periods_per_year: int

Ergebnis/Return: 
- annualized return: float

Implementieren:
- annualized_return = returns.mean() * periods_per_year
- oder geometrisch: (1 + returns).prod() ** (periods_per_yaer / len(returns)) - 1
"""

def calculate_annualized_return(
        returns: Union[pd.Series, np.ndarray],
        periods_per_year: int=252
) -> float:
    annualized_return = returns.mean() * periods_per_year
    return annualized_return

"""
Calculate annualized Volatility:

Parametern:
- returns: pd.Series or np.ndarray
- periods_per_year: int

Ergebnis/Return: 
- annualized volatility (std): float

Implementieren:
- std(returns) * sqrt(periods_per_year)
"""

def calculate_annualized_volatility(
        returns: Union[pd.Series, np.ndarray],
        periods_per_year: int=252
) -> float:
    annualized_volatility = returns.std() * np.sqrt(periods_per_year)
    return annualized_volatility
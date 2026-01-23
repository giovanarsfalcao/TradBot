"""
Portfolio Optimization Module

Modern Portfolio Theory (MPT) - Markowitz Optimization

Kernkonzept:
- Ziel: Optimale Gewichtung von Assets im Portfolio
- Trade-off: Rendite vs. Risiko
- Efficient Frontier: Kurve der optimalen Portfolios
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

# PyPortfolioOpt imports - pip install pyportfolioopt
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt import EfficientFrontier


"""
Expected Returns berechnen:
- Input: df mit historischen Preisen 
- Output: Series mit erwarteten jährlichen Returns pro Ticker
- Methode: mean_historical_return (Durchschnitt * 252)
"""

def calculate_expected_returns(prices: pd.DataFrame) -> pd.Series:
    mu = expected_returns.mean_historical_return(prices)
    return mu


"""
Kovarianzmatrix berechnen:
- Input: DataFrame mit historischen Preisen
- Output: DataFrame (Matrix) mit Kovarianzen zwischen Assets
- Misst: Wie bewegen sich Assets zusammen?
"""

def calculate_covariance_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    S = risk_models.sample_cov(prices)
    return S

"""
Maximum Sharpe Ratio Portfolio:
- Ziel: Beste risikoadjustierte Rendite
- Beantwortet: "Welche Gewichtung gibt mir die höchste Rendite pro Risikoeinheit?"

Input:
- prices: df mit historischen Preisen
- risk_free_rate: Risikofreier Zinssatz (default 0.02 = 2%)

Output:
- weights: Dict (Value-Key) mit Gewichtungen {ticker: weight}
Bsp.: portfolio = {"Ticker": "AAPL", "Weight": 0.3}
Zugriff über Key: portfolio["AAPL"] # Output: 0.3
- performance: Tuple (Festgelegte Reihenfolge) mit expected_return, volatility, sharpe_ratio
Bsp.: (0.15, 0.20, 0.65)
Zugriff über Index: aktien_paar[0] # Output: KO

Beispiel:
    weights, perf = optimize_max_sharpe(prices)
    # weights = {'AAPL': 0.4, 'MSFT': 0.6}
    # perf = (0.15, 0.20, 0.65)  # 15% return, 20% vol, 0.65 sharpe
"""

def optimize_max_sharpe(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.02
) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    mu = calculate_expected_returns(prices)
    S = calculate_covariance_matrix(prices)
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    weights = ef.clean_weights()
    performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    return weights, performance


"""
Minimum Volatility Portfolio:
- Ziel: Geringstes Risiko
- Beantwortet: "Welche Gewichtung minimiert mein Risiko?"

Input/Output: Gleich wie optimize_max_sharpe
"""

def optimize_min_volatility(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.02
) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    mu = calculate_expected_returns(prices)
    S = calculate_covariance_matrix(prices)
    ef = EfficientFrontier(mu, S)
    ef.min_volatility()  # <-- Unterschied zu max_sharpe
    weights = ef.clean_weights()
    performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    return weights, performance


"""
Efficient Frontier generieren:
- Ziel: Kurve aller optimalen Portfolios visualisieren
- X-Achse: Volatilität (Risiko)
- Y-Achse: Expected Return

Input:
- prices: df mit historischen Preisen
- n_points: Anzahl Punkte auf der Kurve (default 50)

Output:
- df mit Spalten: ['return', 'volatility', 'sharpe']
"""

def get_efficient_frontier(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.02,
    n_points: int = 50
) -> pd.DataFrame:

    mu = calculate_expected_returns(prices)
    S = calculate_covariance_matrix(prices)
    min_ret = mu.min()
    max_ret = mu.max()

    returns = [] # [] Liste (list): sie ist veränderbar (mutable) und speichert Elemente in einer festen Reihenfolge, auf die du über liste[0] zugreifst
    volatilities = []

    for target in np.linspace(min_ret, max_ret, n_points):
        ef = EfficientFrontier(mu, S)
        ef.efficient_return(target)
        ret, vol, _ = ef.portfolio_performance()
        returns.append(ret)
        volatilities.append(vol)
    return pd.DataFrame({'return': returns, 'volatility': volatilities})

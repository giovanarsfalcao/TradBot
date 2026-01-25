"""
TradBot Data Module

- YFinanceFix: Rate Limiting Fix für Yahoo Finance
- Database: SQLAlchemy Models und Connection
- CRUD: Save/Load Operationen für DataFrames
"""

from .yfinance_fix import YFinanceFix
from .database import (
    engine,
    Session,
    create_tables,
    MarketData,
    Trade,
    Signal,
    PortfolioSnapshot,
    PerformanceMetric
)
from .crud import (
    save_market_data,
    load_market_data,
    save_trade,
    load_trades,
    save_signal,
    load_signals,
    save_portfolio_snapshot,
    load_portfolio_history,
    save_performance_metrics,
    load_performance_history
)

__all__ = [
    # YFinance
    'YFinanceFix',

    # Database
    'engine',
    'Session',
    'create_tables',
    'MarketData',
    'Trade',
    'Signal',
    'PortfolioSnapshot',
    'PerformanceMetric',

    # CRUD
    'save_market_data',
    'load_market_data',
    'save_trade',
    'load_trades',
    'save_signal',
    'load_signals',
    'save_portfolio_snapshot',
    'load_portfolio_history',
    'save_performance_metrics',
    'load_performance_history',
]

# CLAUDE.md - TradBot Project Guidelines

## Project Overview

TradBot is a quantitative trading analysis platform for paper trading, backtesting, and research. Built with Python + Streamlit, it provides technical indicators, regression models (OLS/Logistic), and portfolio optimization (Markowitz MPT).

## Quick Start

```bash
# Install dependencies
pip install -r app/requirements.txt

# Run the Streamlit app
streamlit run app/app.py
```

## Architecture

```
TradBot/
├── app/                    # Streamlit web application
│   └── app.py              # Main dashboard
├── tradbot/                # Core Python package
│   ├── data/               # Database (SQLAlchemy) + yfinance data fetching
│   ├── strategy/           # Technical indicators + regression models
│   ├── risk/               # Sharpe, VaR, max drawdown calculations
│   └── portfolio/          # Markowitz optimization (pyportfolioopt)
└── notebooks/              # Research notebooks (indicators, regression analysis)
```

## Key Modules

- **tradbot/data/yfinance_fix.py** - Rate limiting workaround using curl_cffi to impersonate Chrome
- **tradbot/data/database.py** - SQLAlchemy ORM models (MarketData, Trade, Signal, Portfolio, Metrics)
- **tradbot/data/crud.py** - CRUD operations for all database entities
- **tradbot/strategy/indicators.py** - Chainable technical indicators (MACD, RSI, MFI, Bollinger Bands)
- **tradbot/strategy/models.py** - LinearRegression (OLS) and LogisticRegression for price prediction
- **tradbot/risk/metrics.py** - Risk calculations (Sharpe ratio, VaR, max drawdown)
- **tradbot/portfolio/optimizer.py** - Portfolio optimization (max Sharpe, min volatility, efficient frontier)

## Critical Guidelines

### DO NOT break the yfinance fix
The `yfinance_fix.py` module is critical for data fetching. It uses `curl_cffi` with Chrome impersonation to bypass Yahoo Finance rate limiting. Never modify this without understanding the session/cookie handling.

### Ask before implementing new features
Discuss the approach before writing code for new functionality. Don't assume an implementation path.

### Handle errors with user-friendly messages
In Streamlit code, catch exceptions and display readable error messages to the user rather than raw tracebacks.

### Be careful with these fragile areas
1. **Database schema** - Changing SQLAlchemy models affects stored data. Migrations may be needed.
2. **Indicator calculations** - Math must match standard technical analysis definitions exactly.

## Code Style

- **Language**: Use English for all new code and comments
- **Comments**: Minimal - code should be self-documenting
- **Data manipulation**: Use pandas for all data operations
- **Testing**: No formal test suite yet; manual testing via notebooks/Streamlit UI

## Data Flow

1. **Fetch**: yfinance + curl_cffi session → OHLCV DataFrame
2. **Store**: SQLAlchemy ORM → tradbot.db (SQLite)
3. **Analyze**: TechnicalIndicators / Regression models
4. **Optimize**: pyportfolioopt for portfolio weights
5. **Display**: Streamlit dashboard with interactive plots

## Database Tables

- `MarketData` - OHLCV price data (indexed by ticker, date)
- `Trade` - Buy/sell transaction records
- `Signal` - Trading signals with source and strength
- `PortfolioSnapshot` - Portfolio weights and value over time
- `PerformanceMetric` - Risk metrics snapshots

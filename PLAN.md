# TradBot - Entwicklungsplan

> Quantitative Trading Platform mit Deep Learning, marktneutralen Strategien und Live Trading via Interactive Brokers

---

## Inhaltsverzeichnis

1. [Projektbeschreibung](#projektbeschreibung)
2. [Aktueller Stand](#aktueller-stand)
3. [Installation](#installation)
4. [Nutzung](#nutzung)
5. [Geplante Features](#geplante-features)
6. [Architektur](#architektur)
7. [Roadmap](#roadmap)
8. [Technologie-Stack](#technologie-stack)

---

## Projektbeschreibung

TradBot ist eine quantitative Trading-Analyse-Plattform für:

- **Paper Trading & Backtesting** - Strategien risikofrei testen
- **Deep Learning** - LSTM/Transformer für Signal-Generierung
- **Marktneutrale Strategien** - Pairs Trading, Statistical Arbitrage
- **Live Trading** - Automatisierte Execution via Interactive Brokers
- **Cloud Deployment** - 24/7 Betrieb auf AWS/GCP

### Zielgruppe

- Quantitative Trader
- Algorithmic Trading Entwickler
- Finance-interessierte ML Engineers

### Kernprinzipien

1. **Research First** - Jede Strategie wird gründlich backtested
2. **Risk Management** - Strenge Drawdown-Limits und Position Sizing
3. **Modularität** - Komponenten sind austauschbar und testbar
4. **Transparenz** - Alle Trades und Signale werden geloggt

---

## Aktueller Stand

### Implementiert

| Modul | Status | Beschreibung |
|-------|--------|--------------|
| `tradbot/strategy/indicators.py` | Fertig | MACD, RSI, MFI, Bollinger Bands |
| `tradbot/strategy/models.py` | Fertig | OLS & Logistic Regression |
| `tradbot/risk/metrics.py` | Fertig | Sharpe, VaR, Max Drawdown |
| `tradbot/portfolio/optimizer.py` | Fertig | Markowitz (Max Sharpe, Min Vol) |
| `tradbot/data/database.py` | Fertig | SQLAlchemy ORM Models |
| `tradbot/data/crud.py` | Fertig | CRUD Operations |
| `tradbot/data/yfinance_fix.py` | Fertig | Yahoo Finance Rate-Limit Workaround |
| `app/app.py` | Fertig | Streamlit Dashboard |

### In Entwicklung

| Feature | Status | Priorität |
|---------|--------|-----------|
| Deep Learning Models | Geplant | Hoch |
| IB Integration | Geplant | Hoch |
| Signal Engine | Geplant | Hoch |
| FastAPI Backend | Geplant | Mittel |
| React Frontend | Geplant | Mittel |
| Cloud Deployment | Geplant | Mittel |

---

## Installation

### Voraussetzungen

- Python 3.10+
- pip oder conda
- Git

### Basis-Installation (aktueller Stand)

```bash
# Repository klonen
git clone <repo-url>
cd TradBot

# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

# Dependencies installieren
pip install -r app/requirements.txt

# Streamlit App starten
streamlit run app/app.py
```

### Erweiterte Installation (zukünftig)

```bash
# Alle Dependencies (inkl. Deep Learning, IB)
pip install -r requirements.txt

# Oder mit Poetry
poetry install

# Environment Variables setzen
cp .env.example .env
# Dann .env bearbeiten mit IB Credentials etc.
```

### Docker Installation (zukünftig)

```bash
# Development
docker-compose up -d

# Production
docker-compose -f docker-compose.prod.yml up -d
```

---

## Nutzung

### 1. Streamlit Dashboard (aktuell)

```bash
streamlit run app/app.py
```

**Features:**
- Technische Indikatoren visualisieren
- Lineare & Logistische Regression
- Risk Metrics berechnen
- Portfolio Optimization

### 2. Python API (aktuell)

```python
from tradbot.strategy import TechnicalIndicators
from tradbot.risk import calculate_sharpe_ratio, calculate_var
from tradbot.portfolio import optimize_max_sharpe

# Daten laden
import yfinance as yf
df = yf.download("AAPL", period="2y")

# Indikatoren hinzufügen
ti = TechnicalIndicators(df)
ti.add_macd().add_rsi().add_bb()
result = ti.get_df()

# Risk Metrics
returns = df['Close'].pct_change().dropna()
sharpe = calculate_sharpe_ratio(returns)
var_95 = calculate_var(returns, 0.95)

# Portfolio Optimization
prices = yf.download(["AAPL", "MSFT", "GOOGL"], period="2y")['Close']
weights, performance = optimize_max_sharpe(prices)
```

### 3. Deep Learning (zukünftig)

```python
from tradbot.strategy.deep_learning import LSTMModel, Trainer

# Modell erstellen
model = LSTMModel(
    input_size=10,  # Features
    hidden_size=64,
    num_layers=2,
    output_size=1   # Buy/Sell Signal
)

# Training
trainer = Trainer(model)
trainer.fit(train_data, epochs=100)

# Prediction
signals = model.predict(new_data)
```

### 4. IB Trading (zukünftig)

```python
from tradbot.execution import IBConnector, OrderManager

# Verbindung
ib = IBConnector()
await ib.connect(port=7497)  # Paper Trading

# Market Data
quotes = await ib.get_quotes("AAPL")

# Order platzieren
order = OrderManager(ib)
await order.market_order("AAPL", "BUY", 100)
```

### 5. CLI Commands (zukünftig)

```bash
# Backtest ausführen
tradbot backtest --strategy pairs_trading --start 2023-01-01

# Signal generieren
tradbot signals --ticker AAPL --model lstm

# Paper Trading starten
tradbot trade --mode paper --strategy momentum
```

---

## Geplante Features

### Phase 1: Deep Learning

- [ ] LSTM Model für Zeitreihen
- [ ] Transformer Model (optional)
- [ ] Feature Engineering Pipeline
- [ ] Walk-Forward Cross-Validation
- [ ] Hyperparameter Tuning (Optuna)
- [ ] Model Registry & Versionierung

### Phase 2: Signal Engine

- [ ] Signal Klasse (Buy/Sell/Hold + Confidence)
- [ ] Multi-Model Ensemble
- [ ] Signal Aggregation
- [ ] Alert System (Telegram, Email)
- [ ] Signal Performance Tracking

### Phase 3: Marktneutrale Strategien

- [ ] Pairs Trading
  - [ ] Kointegrations-Tests
  - [ ] Spread-Berechnung
  - [ ] Z-Score Entry/Exit
- [ ] Statistical Arbitrage
- [ ] Beta-Hedging
- [ ] Dollar-Neutral Portfolios

### Phase 4: Execution & IB Integration

- [ ] IB Gateway Connector
- [ ] Paper Trading Support
- [ ] Live Trading Support
- [ ] Order Management
- [ ] Position Tracking
- [ ] P&L Monitoring

### Phase 5: Backtesting

- [ ] Vectorized Backtesting (vectorbt)
- [ ] Transaction Costs
- [ ] Slippage Modeling
- [ ] Walk-Forward Analysis
- [ ] Monte Carlo Simulation
- [ ] Performance Attribution

### Phase 6: Web Application

- [ ] FastAPI Backend
  - [ ] REST API
  - [ ] WebSocket für Live-Daten
  - [ ] Background Workers (Celery)
- [ ] React/Next.js Frontend
  - [ ] Trade Journal
  - [ ] Signal Dashboard
  - [ ] Risk Monitor
  - [ ] Backtest Lab
- [ ] Authentication (JWT)
- [ ] PostgreSQL Migration

### Phase 7: Cloud & DevOps

- [ ] Docker Containerization
- [ ] Docker Compose Setup
- [ ] AWS/GCP Deployment
- [ ] IB Gateway in Cloud
- [ ] CI/CD Pipeline
- [ ] Monitoring & Alerts
- [ ] Automated Backups

### Phase 8: Claude Integration

- [ ] MCP Server für Claude Code
- [ ] Trade Review Assistent
- [ ] Strategie-Diskussion
- [ ] Signal-Erklärung
- [ ] Risk Check via Chat

---

## Architektur

### Aktuelle Struktur

```
TradBot/
|-- app/
|   |-- app.py              # Streamlit Dashboard
|   |-- yfinance_fix.py     # Yahoo Finance Workaround
|   `-- requirements.txt
|
|-- tradbot/
|   |-- __init__.py
|   |-- strategy/
|   |   |-- indicators.py   # Technische Indikatoren
|   |   `-- models.py       # Regression Models
|   |-- risk/
|   |   `-- metrics.py      # Risk Calculations
|   |-- portfolio/
|   |   `-- optimizer.py    # Markowitz Optimization
|   `-- data/
|       |-- database.py     # SQLAlchemy Models
|       |-- crud.py         # CRUD Operations
|       `-- yfinance_fix.py # Data Fetching
|
`-- notebooks/              # Research & Experiments
```

### Ziel-Struktur

```
TradBot/
|-- backend/                          # FastAPI Backend
|   |-- app/
|   |   |-- main.py
|   |   |-- config.py
|   |   |-- api/v1/
|   |   |   |-- trades.py
|   |   |   |-- signals.py
|   |   |   |-- models.py
|   |   |   |-- portfolio.py
|   |   |   `-- backtest.py
|   |   |-- websocket/
|   |   |-- core/
|   |   |-- db/
|   |   |-- services/
|   |   |   |-- claude_assistant.py
|   |   |   `-- ib_connector.py
|   |   `-- workers/
|   |-- Dockerfile
|   `-- requirements.txt
|
|-- frontend/                         # React/Next.js
|   |-- src/
|   |-- package.json
|   `-- Dockerfile
|
|-- tradbot/                          # Core Library
|   |-- strategy/
|   |   |-- indicators.py
|   |   |-- models.py
|   |   |-- deep_learning/
|   |   |   |-- lstm_model.py
|   |   |   |-- transformer_model.py
|   |   |   `-- trainer.py
|   |   `-- market_neutral/
|   |       |-- pairs_trading.py
|   |       `-- stat_arb.py
|   |-- signals/
|   |   |-- signal_generator.py
|   |   `-- signal_combiner.py
|   |-- execution/
|   |   |-- ib_connector.py
|   |   |-- order_manager.py
|   |   `-- paper_trader.py
|   |-- backtest/
|   |   |-- engine.py
|   |   `-- metrics.py
|   |-- risk/
|   |-- portfolio/
|   |-- data/
|   `-- mcp/
|       `-- server.py
|
|-- infrastructure/
|   |-- docker-compose.yml
|   |-- docker-compose.prod.yml
|   |-- kubernetes/
|   `-- terraform/
|
|-- notebooks/
|-- tests/
|-- .env.example
`-- PLAN.md
```

### System-Architektur (Cloud)

```
                    +------------------+
                    |   React/Next.js  |
                    |    Frontend      |
                    +--------+---------+
                             |
                             v
+------------------+   +-----+------+   +------------------+
|   Claude Code    |-->|   FastAPI  |<--|   IB Gateway     |
|   (MCP Server)   |   |   Backend  |   |   Container      |
+------------------+   +-----+------+   +------------------+
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
       +------+----+  +------+----+  +------+----+
       | PostgreSQL|  |   Redis   |  | TimescaleDB|
       |   (RDS)   |  |(ElastiCache| | (optional) |
       +-----------+  +-----------+  +-----------+
```

---

## Roadmap

### Q1 2026: Foundation

| Woche | Fokus |
|-------|-------|
| 1-3 | PyTorch Grundlagen lernen |
| 4-6 | LSTM Model implementieren |
| 7-8 | IB Paper Trading Setup |
| 9-11 | FastAPI Backend |
| 12 | Signal Engine v1 |

### Q2 2026: Strategies

| Woche | Fokus |
|-------|-------|
| 1-2 | JavaScript/TypeScript lernen |
| 3-6 | Marktneutrale Strategien |
| 7-10 | React Frontend |
| 11-12 | Backtesting Framework |

### Q3 2026: Production

| Woche | Fokus |
|-------|-------|
| 1-4 | Docker & Cloud Setup |
| 5-8 | Paper Trading Live (Cloud) |
| 9-12 | Monitoring & Optimierung |

### Q4 2026: Live Trading

| Woche | Fokus |
|-------|-------|
| 1-4 | Live Trading (kleine Positionen) |
| 5-8 | Skalierung |
| 9-12 | Erweiterte Strategien |

---

## Technologie-Stack

### Backend

| Technologie | Verwendung | Status |
|-------------|------------|--------|
| Python 3.10+ | Core Language | Aktiv |
| Pandas | Data Manipulation | Aktiv |
| NumPy | Numerical Computing | Aktiv |
| SQLAlchemy | ORM | Aktiv |
| Statsmodels | Statistics | Aktiv |
| PyTorch | Deep Learning | Geplant |
| FastAPI | REST API | Geplant |
| Celery | Background Tasks | Geplant |
| Redis | Cache/Queue | Geplant |
| PostgreSQL | Production DB | Geplant |

### Frontend

| Technologie | Verwendung | Status |
|-------------|------------|--------|
| Streamlit | Current Dashboard | Aktiv |
| React | Future UI | Geplant |
| Next.js | React Framework | Geplant |
| TailwindCSS | Styling | Geplant |
| Recharts | Charts | Geplant |

### Trading

| Technologie | Verwendung | Status |
|-------------|------------|--------|
| yfinance | Market Data | Aktiv |
| ib_insync | IB Integration | Geplant |
| vectorbt | Backtesting | Geplant |

### DevOps

| Technologie | Verwendung | Status |
|-------------|------------|--------|
| Docker | Containerization | Geplant |
| AWS/GCP | Cloud Hosting | Geplant |
| GitHub Actions | CI/CD | Geplant |

### AI/ML

| Technologie | Verwendung | Status |
|-------------|------------|--------|
| Claude API | Trading Assistant | Geplant |
| MCP | Claude Code Tools | Geplant |
| Optuna | Hyperparameter Tuning | Geplant |

---

## Kosten (geschätzt)

| Service | Entwicklung | Production |
|---------|-------------|------------|
| Claude API | ~$10-20/Monat | ~$20-50/Monat |
| AWS/GCP | ~$30-50/Monat | ~$100-200/Monat |
| Domain + SSL | ~$2/Monat | ~$2/Monat |
| IB Daten-Feed | $0 (inkl.) | $0 (inkl.) |
| **Gesamt** | **~$50-75/Monat** | **~$125-250/Monat** |

---

## Lernressourcen

### Deep Learning
- PyTorch Tutorials (pytorch.org)
- Fast.ai Course
- "Advances in Financial Machine Learning" - Marcos López de Prado

### Algo Trading
- QuantConnect (Lernplattform)
- "Algorithmic Trading" - Ernest Chan
- IB API Dokumentation

### Marktneutrale Strategien
- "Pairs Trading" - Ganapathy Vidyamurthy
- Quantopian Lectures (GitHub Archive)

### Web Development
- JavaScript.info
- React.dev
- Next.js Learn

---

## Risiko-Hinweise

1. **Overfitting** - Zeitreihen-CV nutzen, Out-of-Sample testen
2. **Look-Ahead Bias** - Features nur aus vergangenen Daten
3. **Paper != Live** - Slippage und Emotionen unterschätzt
4. **Drawdowns** - Max Drawdown Limits setzen
5. **Nie mehr riskieren als du verlieren kannst**

---

## Kontakt & Support

- GitHub Issues für Bugs und Feature Requests
- Discussions für Fragen und Ideen

---

*Letztes Update: Januar 2026*

"""
TWS Connector - Lean Interactive Brokers TWS Integration

Minimale Implementierung für Trade-Ausführung via TWS Desktop.
Verwendet ib_insync für einfache async-fähige API-Kommunikation.

TWS Desktop Einstellungen:
- API Settings aktivieren: Edit > Global Configuration > API > Settings
- "Enable ActiveX and Socket Clients" aktivieren
- Port: 7497 (Paper Trading) oder 7496 (Live Trading)
- "Allow connections from localhost only" aktivieren
"""

from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder, Trade, util
from typing import Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class TWSConnector:
    """Lean TWS Connector für Trade-Ausführung."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        """
        Args:
            host: TWS Host (default: localhost)
            port: TWS Port (7497=Paper, 7496=Live)
            client_id: Unique Client ID für diese Verbindung
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()

    def connect(self) -> bool:
        """Verbindung zu TWS herstellen."""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            logger.info(f"Connected to TWS at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Verbindung trennen."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from TWS")

    def is_connected(self) -> bool:
        """Prüft ob Verbindung besteht."""
        return self.ib.isConnected()

    # === Market Data ===

    def get_quote(self, symbol: str, exchange: str = "SMART", currency: str = "USD"):
        """
        Live Quote abrufen.

        Args:
            symbol: Ticker Symbol (z.B. "AAPL")
            exchange: Exchange (default: SMART routing)
            currency: Währung (default: USD)

        Returns:
            Ticker object mit bid, ask, last, volume etc.
        """
        contract = Stock(symbol, exchange, currency)
        self.ib.qualifyContracts(contract)
        ticker = self.ib.reqMktData(contract)
        self.ib.sleep(2)  # Warten auf Daten
        return ticker

    def get_historical(self, symbol: str, duration: str = "5 D",
                       bar_size: str = "1 hour", exchange: str = "SMART",
                       currency: str = "USD") -> pd.DataFrame:
        """
        Historische Daten als DataFrame abrufen.

        Args:
            symbol: Ticker Symbol
            duration: Zeitraum (z.B. "5 D", "1 M", "1 Y")
            bar_size: Bar-Größe (z.B. "1 min", "5 mins", "1 hour", "1 day")
            exchange: Exchange
            currency: Währung

        Returns:
            DataFrame mit OHLCV Daten

        Example:
            >>> df = tws.get_historical("AAPL", "5 D", "1 hour")
        """
        contract = Stock(symbol, exchange, currency)
        self.ib.qualifyContracts(contract)

        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True
        )

        if not bars:
            logger.warning(f"No historical data for {symbol}")
            return pd.DataFrame()

        df = util.df(bars)
        logger.info(f"Loaded {len(df)} bars for {symbol}")
        return df

    def stop_market_data(self, ticker):
        """Market Data Stream stoppen."""
        self.ib.cancelMktData(ticker.contract)

    # === Order Execution ===

    def market_order(self, symbol: str, quantity: int, action: str = "BUY",
                     exchange: str = "SMART", currency: str = "USD") -> Optional[Trade]:
        """
        Market Order ausführen.

        Args:
            symbol: Ticker Symbol (z.B. "AAPL")
            quantity: Anzahl Aktien
            action: "BUY" oder "SELL"
            exchange: Exchange (default: SMART routing)
            currency: Währung (default: USD)

        Returns:
            Trade object oder None bei Fehler
        """
        contract = Stock(symbol, exchange, currency)
        order = MarketOrder(action.upper(), quantity)

        try:
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)  # Kurz warten auf Order-Status
            logger.info(f"Market Order placed: {action} {quantity} {symbol} - Status: {trade.orderStatus.status}")
            return trade
        except Exception as e:
            logger.error(f"Market Order failed: {e}")
            return None

    def limit_order(self, symbol: str, quantity: int, limit_price: float,
                    action: str = "BUY", exchange: str = "SMART",
                    currency: str = "USD") -> Optional[Trade]:
        """
        Limit Order ausführen.

        Args:
            symbol: Ticker Symbol
            quantity: Anzahl Aktien
            limit_price: Limit Preis
            action: "BUY" oder "SELL"

        Returns:
            Trade object oder None bei Fehler
        """
        contract = Stock(symbol, exchange, currency)
        order = LimitOrder(action.upper(), quantity, limit_price)

        try:
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            logger.info(f"Limit Order placed: {action} {quantity} {symbol} @ {limit_price} - Status: {trade.orderStatus.status}")
            return trade
        except Exception as e:
            logger.error(f"Limit Order failed: {e}")
            return None

    def stop_order(self, symbol: str, quantity: int, stop_price: float,
                   action: str = "SELL", exchange: str = "SMART",
                   currency: str = "USD") -> Optional[Trade]:
        """
        Stop Order ausführen.

        Args:
            symbol: Ticker Symbol
            quantity: Anzahl Aktien
            stop_price: Stop Preis
            action: "BUY" oder "SELL"

        Returns:
            Trade object oder None bei Fehler
        """
        contract = Stock(symbol, exchange, currency)
        order = StopOrder(action.upper(), quantity, stop_price)

        try:
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            logger.info(f"Stop Order placed: {action} {quantity} {symbol} @ {stop_price} - Status: {trade.orderStatus.status}")
            return trade
        except Exception as e:
            logger.error(f"Stop Order failed: {e}")
            return None

    def cancel_order(self, trade: Trade) -> bool:
        """Order stornieren."""
        try:
            self.ib.cancelOrder(trade.order)
            logger.info(f"Order cancelled: {trade.order.orderId}")
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def get_open_orders(self) -> list:
        """Alle offenen Orders abrufen."""
        return self.ib.openOrders()

    def get_positions(self) -> list:
        """Aktuelle Positionen abrufen."""
        return self.ib.positions()

    def get_account_summary(self) -> dict:
        """Account-Übersicht abrufen."""
        summary = {}
        for item in self.ib.accountSummary():
            summary[item.tag] = item.value
        return summary


# === Convenience Functions ===

def quick_trade(symbol: str, quantity: int, action: str = "BUY",
                order_type: str = "market", price: float = None,
                port: int = 7497) -> Optional[Trade]:
    """
    Schnelle Trade-Ausführung ohne manuelle Connection-Verwaltung.

    Args:
        symbol: Ticker Symbol
        quantity: Anzahl Aktien
        action: "BUY" oder "SELL"
        order_type: "market", "limit", oder "stop"
        price: Preis für Limit/Stop Orders
        port: TWS Port (7497=Paper, 7496=Live)

    Returns:
        Trade object oder None

    Example:
        >>> trade = quick_trade("AAPL", 10, "BUY", "market")
        >>> trade = quick_trade("AAPL", 10, "BUY", "limit", price=150.00)
    """
    tws = TWSConnector(port=port)

    if not tws.connect():
        return None

    try:
        if order_type == "market":
            return tws.market_order(symbol, quantity, action)
        elif order_type == "limit" and price:
            return tws.limit_order(symbol, quantity, price, action)
        elif order_type == "stop" and price:
            return tws.stop_order(symbol, quantity, price, action)
        else:
            logger.error(f"Invalid order_type or missing price: {order_type}")
            return None
    finally:
        tws.disconnect()

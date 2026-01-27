"""
TradBot Execution Module

Trade execution via Interactive Brokers TWS.
"""

try:
    from .tws_connector import TWSConnector, quick_trade
    __all__ = ['TWSConnector', 'quick_trade']
except ImportError:
    __all__ = []

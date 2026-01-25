"""
YFinance Fix - Umgeht Yahoo Finance Rate Limiting

Problem: Yahoo Finance blockiert Anfragen die wie Bots aussehen.
Lösung: Session die sich als Chrome Browser tarnt.

Verwendung:
    from tradbot.data import YFinanceFix

    yf_fix = YFinanceFix()
    df = yf.download("AAPL", session=yf_fix.session)
"""

import yfinance.data as _data
from requests.cookies import create_cookie
from curl_cffi import requests


class YFinanceFix:
    """
    Singleton-ähnliche Klasse für YFinance Fixes.

    Beim ersten Import:
    1. Patch auf yfinance anwenden (Cookie-Fix)
    2. Chrome-Session erstellen
    """

    _patched = False  # Class variable - wird nur einmal gepatcht

    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: Print Status-Meldungen (default True)
        """
        self.verbose = verbose
        self.session = None

        # Patch nur einmal anwenden
        if not YFinanceFix._patched:
            self._apply_patch()
            YFinanceFix._patched = True

        # Session erstellen
        self._create_session()

    def _wrap_cookie(self, cookie, session):
        """Wandelt Cookies in Objekte um, damit yfinance sie versteht."""
        if isinstance(cookie, str):
            value = session.cookies.get(cookie)
            return create_cookie(name=cookie, value=value)
        return cookie

    def _apply_patch(self):
        """Wendet den Monkey-Patch auf yfinance an."""
        original = _data.YfData._get_cookie_basic
        wrapper = self._wrap_cookie

        def _patched(self_inner, timeout=30):
            cookie = original(self_inner, timeout)
            return wrapper(cookie, self_inner._session)

        _data.YfData._get_cookie_basic = _patched

        if self.verbose:
            print("[YFinanceFix] Patch angewendet")

    def _create_session(self):
        """Erstellt eine getarnte Chrome-Session."""
        try:
            self.session = requests.Session(impersonate="chrome")
            if self.verbose:
                print("[YFinanceFix] Chrome-Session erstellt")
        except Exception as e:
            if self.verbose:
                print(f"[YFinanceFix] Fehler: {e}")
            self.session = None


# from tradbot.data import YFinanceFix
# import yfinance as yf

# # Session erstellen
# yf_fix = YFinanceFix()

# # Download mit Session
# df = yf.download("AAPL", session=yf_fix.session)
# Oder kürzer:


# from tradbot import YFinanceFix

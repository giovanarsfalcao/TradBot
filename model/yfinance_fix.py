# Anwendung in anderen Dateien: Sobald Sie import yfinance_fix schreiben, führt Python den Code in dieser Datei einmal aus. 
# Das wendet den Patch an und erstellt die chrome_session, die Sie dann überall wiederverwenden können.

# Implementieren in anderen Dateien:
    
    # Importieren Sie yfinance_fix.
    
    # Nutzen Sie session=yfinance_fix.chrome_session beim Download.

import yfinance.data as _data
from requests.cookies import create_cookie
from curl_cffi import requests

# --- INTERNE LOGIK (NICHT ÄNDERN) ---

def _wrap_cookie(cookie, session):
    """Hilfsfunktion: Wandelt Cookies in Objekte um, damit yfinance sie versteht."""
    if isinstance(cookie, str):
        value = session.cookies.get(cookie)
        return create_cookie(name=cookie, value=value)
    return cookie

def apply_patch():
    """Wendet den Monkey-Patch auf yfinance an."""
    original = _data.YfData._get_cookie_basic

    def _patched(self, timeout=30):
        cookie = original(self, timeout)
        return _wrap_cookie(cookie, self._session)

    _data.YfData._get_cookie_basic = _patched
    print("[yfinance_fix] Patch erfolgreich angewendet.")

# --- AUTOMATISCHE AUSFÜHRUNG BEIM IMPORT ---

# 1. Patch anwenden
apply_patch()

# 2. Eine getarnte Chrome-Session erstellen, die Sie importieren können
try:
    chrome_session = requests.Session(impersonate="chrome")
    print("[yfinance_fix] Chrome-Session erstellt.")
except Exception as e:
    print(f"[yfinance_fix] Fehler beim Erstellen der Session: {e}")
    chrome_session = None


# Problem: Yahoo Finance blockiert Ihr Programm, weil es sich wie ein Roboter verhält.

# Ziel: Wir müssen Ihr Programm als echten "Google Chrome" Browser verkleiden.

# Werkzeug: Die Bibliothek curl_cffi erstellt einen falschen Browser (Session), den Yahoo akzeptiert.

# Hürde: Die Bibliothek yfinance versteht die "Kekse" (Cookies) dieses falschen Browsers normalerweise nicht.

# Lösung (Patch): Wir schreiben eine kleine "Übersetzer-Funktion" (_wrap_cookie), die die Kekse für yfinance lesbar macht.

# Ergebnis: Yahoo denkt, Sie sind ein Mensch im Chrome-Browser, und gibt die Daten wieder frei.
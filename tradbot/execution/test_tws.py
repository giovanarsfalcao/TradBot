"""
Quick TWS Test - Ausführen mit: python test_tws.py
TWS muss laufen mit API auf Port 7497 (Paper)
"""

from tws_connector import TWSConnector

tws = TWSConnector(port=7497)

print("Connecting...")
if tws.connect():
    print("Connected!\n")

    # Quote Test
    print("Getting AAPL quote...")
    quote = tws.get_quote("AAPL")
    print(f"  Bid: {quote.bid} | Ask: {quote.ask} | Last: {quote.last}\n")

    # Historical Test
    print("Getting historical data...")
    df = tws.get_historical("AAPL", "5 D", "1 hour")
    print(f"  {len(df)} bars loaded")
    print(df.tail(3))

    tws.disconnect()
    print("\nDone!")
else:
    print("Connection failed - check TWS settings")

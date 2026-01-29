"""
Quick Backtest Test - Run with: python -m tradbot.backtest.test_backtest
"""

import yfinance as yf
from tradbot.data import YFinanceFix
from tradbot.signals.signal_rsi_macd import generate_strategy as rsi_strategy
from tradbot.backtest import Backtest

# Patch yfinance
YFinanceFix()

# Fetch data
print("Downloading SPY data...")
df = yf.download("SPY", period="5y")
df.columns = df.columns.get_level_values(0)
print(f"{len(df)} rows loaded\n")

# RSI + MACD Backtest
print("=" * 46)
print("RSI + MACD Strategy")
print("=" * 46)
df_rsi = rsi_strategy(df)
bt = Backtest(df_rsi)
bt.run()
bt.print_summary()

print(f"\nSignal distribution:")
counts = df_rsi["Strategy"].value_counts().sort_index()
for val, count in counts.items():
    label = {-1: "Short", 0: "Flat", 1: "Long"}.get(int(val), str(val))
    print(f"  {label}: {count} ({count/len(df_rsi)*100:.1f}%)")

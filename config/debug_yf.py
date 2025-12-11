import yfinance as yf
import sys
import os
import shutil

def run_diagnostics():
    print("--- YFINANCE DIAGNOSTICS ---")
    
    # 1. Check Python Environment
    print(f"[1] Python Executable: {sys.executable}")
    
    # 2. Check YFinance Version
    try:
        print(f"[2] YFinance Version: {yf.__version__}")
        # We want version 0.2.50 or higher to fix the recent Yahoo API issues
    except AttributeError:
        print("[2] YFinance Version: Could not determine version (AttributeError)")

    # 3. Check Installation Location
    try:
        location = os.path.dirname(yf.__file__)
        print(f"[3] Installed at: {location}")
    except Exception as e:
        print(f"[3] Installation check failed: {e}")

    # 4. Clear Cache (Crucial Step)
    print("\n[4] Attempting to clear cache...")
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "py-yfinance")
    
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print(f"    SUCCESS: Deleted cache directory at {cache_dir}")
        except Exception as e:
            print(f"    ERROR: Could not delete cache: {e}")
    else:
        print("    INFO: No cache directory found (this is fine).")

    # 5. Test Download
    print("\n[5] Testing Download (Ticker: AAPL)...")
    try:
        # Using a stable ticker like AAPL for testing
        dat = yf.download("AAPL", period="1mo", progress=False)
        
        if dat.empty:
            print("    FAILURE: Download ran, but returned empty DataFrame.")
            print("    -> This usually means the library is still outdated or Yahoo is blocking the request.")
        else:
            print("    SUCCESS: Data received!")
            print(dat.head())
    except Exception as e:
        print(f"    CRITICAL ERROR during download: {e}")

if __name__ == "__main__":
    run_diagnostics()
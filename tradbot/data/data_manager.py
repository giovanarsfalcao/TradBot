
import pandas as pd
from curl_cffi import requests
import io

class DataManager:
    """
    Manages data loading from Financial Modeling Prep (FMP).
    """
    def __init__(self, api_key):
        """
        Initializes the DataManager with the FMP API key.

        Args:
            api_key (str): Your FMP API key.
        """
        if not api_key:
            raise ValueError("FMP API key is required.")
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def get_historical_data(self, ticker, start_date=None, end_date=None):
        """
        Fetches historical price data for a given ticker.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'AAPL').
            start_date (str, optional): The start date in 'YYYY-MM-DD' format.
            end_date (str, optional): The end date in 'YYYY-MM-DD' format.

        Returns:
            pandas.DataFrame: A DataFrame containing the historical data 
                              (date, open, high, low, close, adjClose, volume, etc.),
                              or an empty DataFrame if the request fails.
        """
        url = f"{self.base_url}/historical-price-full/{ticker}"
        params = {"apikey": self.api_key}

        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date
        
        try:
            response = requests.get(url, params=params, impersonate="chrome110")
            response.raise_for_status()  # Raise an exception for bad status codes

            data = response.json()

            if not data or 'historical' not in data:
                print(f"No data found for {ticker}.")
                return pd.DataFrame()

            df = pd.DataFrame(data['historical'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.sort_index()
            
            # FMP returns data in descending order, so we reverse it
            return df.iloc[::-1]

        except requests.errors.RequestsError as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return pd.DataFrame()

if __name__ == '__main__':
    # This is an example of how to use the DataManager.
    # IMPORTANT: Replace "YOUR_FMP_API_KEY" with your actual FMP API key. 
    FMP_API_KEY = "YOUR_FMP_API_KEY" 
    
    if FMP_API_KEY == "YOUR_FMP_API_KEY":
        print("Please replace 'YOUR_FMP_API_KEY' with your actual FMP API key.")
    else:
        data_manager = DataManager(api_key=FMP_API_KEY) 
        
        # Example: Fetch data for AAPL
        ticker = "AAPL"
        print(f"Fetching historical data for {ticker}...")
        aapl_data = data_manager.get_historical_data(ticker, start_date="2023-01-01", end_date="2023-12-31")
        
        if not aapl_data.empty:
            print("Successfully fetched data:")
            print(aapl_data.head())
            print("\nData Info:")
            aapl_data.info()
        else:
            print(f"Could not fetch data for {ticker}.")

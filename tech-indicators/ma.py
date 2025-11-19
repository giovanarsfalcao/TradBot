## Moving averages File 
## We will work with daily data, instead of per minute or per hour data, since it is more limited w/ yfinace
## But you can think that if is not working on a daily frame, it wont happen on a minute frame eithe

import yfinance as yf
df = yf.download("SPY", start="1993-01-01")
#print(df)
print(df.head())


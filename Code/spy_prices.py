import yfinance as yf
import pandas as pd
from utils import paths
import os

def get_spy_prices(start_date='2011-01-03', end_date='2023-12-31'):
    path = paths()
    spy = yf.download('SPY', start=start_date, end=end_date)
    spy_prices = spy[['Close']].reset_index()
    spy_prices.columns = ['date', 'spy_price']
    spy_file = os.path.join(path['data'], 'spy_prices.csv')
    spy_prices.to_csv(spy_file, index=False)
    return spy_prices
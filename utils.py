import os
import pandas as pd
import pandas_datareader as pdr
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def get_data():
  today = datetime.today().strftime('%Y-%m-%d')
  df = pdr.DataReader('USDGBP=X', data_source='yahoo', start='2012-01-01', end=today)
  data_filtered = np.array(df.filter(['Close']))
  return np.around(data_filtered, decimals=5)

def get_scaler(env):
  low = [0] * (env.n_stock * 2 + 1)

  high = []
  max_price = env.stock_price_history.max(axis=0)
  min_price = env.stock_price_history.min(axis=0)
  max_cash = env.init_invest * 3
  max_stock_owned = max_cash // min_price
  for i in max_stock_owned:
    high.append(i)
  for i in max_price:
    high.append(i)
  high.append(max_cash)

  scaler = StandardScaler()
  scaler.fit([low, high])
  return scaler


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
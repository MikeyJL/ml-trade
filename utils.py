import os
import pandas as pd
import pandas_datareader as web
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

def get_data():
  tmr = datetime.today() + timedelta(days=1)
  yesterday = datetime.today() - timedelta(days=1)
  df = web.DataReader("GBPUSD", "av-intraday", start=yesterday, end=tmr, api_key='Q1XXYJ8NY07Q1UMS')
  data_filtered = np.array(df.filter(['close']))
  return np.around(data_filtered, decimals=5)


def get_scaler(env):
  max_price = env.stock_price_history.max(axis=0)
  min_price = env.stock_price_history.min(axis=0)

  scaler = StandardScaler()
  scaler.fit([min_price, max_price])
  return scaler


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
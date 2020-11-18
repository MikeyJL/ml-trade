import os
import pandas as pd
import pandas_datareader as pdr
import numpy as np
from datetime import datetime

def get_data():
  today = datetime.today().strftime('%Y-%m-%d')
  df = pdr.DataReader('GBPUSD=X', data_source='yahoo', start='2012-01-01', end=today)
  data_filtered = np.array(df.filter(['Close']))
  return np.around(data_filtered, decimals=5)


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
import os
import pandas_datareader as pdr

def get_data(col, currency_pair, start_date, todays_date):
  df = pdr.DataReader(currency_pair, data_source='yahoo', start=start_date, end=todays_date)
  filtered_df = df.filter([col])
  return filtered_df.values

def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
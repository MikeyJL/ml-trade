# LIVE DATA

import time
import pandas_datareader as web
from datetime import datetime, timedelta
tmr = datetime.today() + timedelta(days=1)
today = datetime.today()
prev = 0

for i in range(18):
  df = web.DataReader("GBPUSD", "av-intraday", start=today, end=tmr, api_key='Q1XXYJ8NY07Q1UMS')
  if len(df) > prev:
    print('New data!')
    print(df)
    prev = len(df)
  time.sleep(10)
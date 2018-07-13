from yahoo_fin.stock_info import get_data,build_url
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

with open('crypto.txt', 'r') as fd:
    asset_list = list(fd.read().splitlines())

file = 'crypto_data_adj_close.csv'

df = pd.DataFrame()

start_date = '2011-08-07'
end_date = '2018-07-13'

counter = 0

for ticker in asset_list:
    site = build_url(ticker , start_date , end_date)
    print(site)


for ticker in asset_list:
    print('\rprocessed %d/%d: ' % (counter, len(asset_list)), end='')
    if ticker in df.columns:
        continue
    err = True
    err_counter = 5
    while (err and err_counter > 0):
        try:
            # "date","open","high","low","close","adjclose","volume"
            data = get_data(ticker, start_date=start_date, end_date=end_date)
            df[ticker] = data.adjclose
            df = df.fillna(method='bfill')
            df = df.reindex(method='bfill')
            df.to_csv(file)
            counter += 1
            if counter > 0 and counter % 5 == 0:
                time.sleep(1)
            err = False
        except Exception as e:
            print('\r%s failed : %s' % (ticker, e), end='')
            err_counter -= 1
            err = True

print('\n')
print(df.head())
print(df.tail())
print(df.columns)

df = pd.read_csv(file)

print(df.head())
print(df.tail())
print(df.columns)

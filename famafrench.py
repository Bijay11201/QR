import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import yfinance as yf

df_ff = pdr.data.DataReader('F-F_Research_Data_Factors', 'famafrench')[0]
df_ff.head()

start_date = datetime(2015, 1, 1)
end_date = datetime.today()
sectors = ['XLB', 'XLC', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XLE']

df_sectors = pd.DataFrame()
for sym in sectors:
    print(sym)
    df = yf.download(sym, start=start_date, end=end_date)
    df = df[['Adj Close']]
    df.columns = [sym]
    df_sectors = pd.concat([df_sectors, df], axis=1, join='outer')

df_sec_ret = df_sectors.resample('M').agg(lambda x: x[-1])
df_sec_ret.index = df_sec_ret.index.to_period()
df_sec_ret = df_sec_ret.pct_change()
df_sec_ret.head()

df_sec_ret = df_sec_ret.apply(lambda x: x-df_ff['RF']/100.0)
df_sec_ret.dropna(axis=0, inplace=True)
df_Y = df_sec_ret

df_X = df_ff[['Mkt-RF', 'SMB', 'HML']]/100.0
df_X = df_X.loc[df_Y.index]
print(f'{df_Y.shape[1]} stocks, {df_X.shape[1]} factors, {df_Y.shape[0]} time steps')

df_X = sm.add_constant(df_X, prepend=False)

beta = pd.DataFrame()             # factor exposures
for sym in df_Y.columns:
    model = sm.OLS(df_Y[sym], df_X)
    results = model.fit()
    beta = pd.concat([beta, pd.DataFrame([results.params[:3]], index=[sym])])
beta
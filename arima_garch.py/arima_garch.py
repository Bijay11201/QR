import numpy as np
import pandas  as pd
import scipy
from datetime import datetime, timedelta
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas_datareader as pdr

end_date=datetime.today()
start_date=datetime(2000,1,1)
spx = pdr.DataReader(name='^GSPC', data_source='yahoo', start=start_date, end=end_date)
hist_close=spx['close']
hist_ret=hist_close/hist_close.shift(1)-1.0
hist_ret.dropna(inplace=True)
hist_ret = hist_ret * 100.0
print(hist_ret.describe())
print(f'Skew: {scipy.stats.skew(hist_ret)}, Kurtosis: {scipy.stats.kurtosis(hist_ret)}')

#ARMIA Model(p,d,q)
from statsmodels.tsa.tsatools import adfuller
result=adfuller(hist_close)
print(f"ADF Static:{result[0]} p-value:{result[1]} ") #null hypo
result=adfuller(hist_ret)
print(f"ADF Static:{result[0]} p-value:{result[1]} ")# reject null hypo

from statsmodels.tsa.stattools import acf  ,pacf
from  statsmodels.graphics.tsaplots import plot_acf ,plot_pacf
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(16, 4))
plot_acf(hist_ret, ax=axes[0])          # determines MA(q)
plot_pacf(hist_ret, ax=axes[1])         # determines AR(p)
plt.show()

acf_stats = acf(hist_ret, fft=False)[1:40] # calculation auto correlation  data frame
acf_df = pd.DataFrame([acf_stats]).T
acf_df.columns = ['acf']
acf_df.index += 1

pacf_stats = pacf(hist_ret)[1:40] # calculation partial auto correlation  data frame
pacf_df = pd.DataFrame([pacf_stats]).T
pacf_df.columns = ['pacf']
pacf_df.index += 1

df_acf_pcaf = pd.concat([acf_df, pacf_df], axis=1)
print(df_acf_pcaf.head())

from statsmodels.tsa.arima_model import ARIMA ,ARMA
his_training=hist_ret.iloc[:-45]
hist_testing=hist_ret.iloc[-45:]
dic_aic={}
for p in range(6):
    for q in range(6):
        try:
            model=ARIMA(his_training,order={p,0,q})
            model_fit=model.fit(disp=0)
            dic_aic[{p,q}]=model_fit.aic
        except:
            pass
df_aic=pd.DataFrame.from_dict(dic_aic,orient="index",columns=['aic'])
p,q=df_aic[df_aic.aic==df_aic.aic.min()].index[0]
print("ARMA Model ({p},{0},{q})")
model=ARIMA(hist_testing,order=(p,0,q))
arima_fitted=model.fit(disp=0)
arima_fitted.summary()

residuals=pd.DataFrame(arima_fitted.resid,columns=["Residuals"])
fig,axes =plt.subplot(1,2,figsize=(10,4))
residuals.plot(kind='kde',title='density',axes=axes[0])
residuals.plot(lines='s',axes=axes[1])

from scipy.stats import shapiro,normaltest
stat,p=normaltest(residuals)
print("Static[{stat:stat[0]} ,{p:p[0]}] ")
stat,p=shapiro(residuals)

fig,axes=plt.subplot(1,2,figsize=(16,4))
plot_acf(residuals,axes=axes[0])
plot_pacf(residuals,axes=axes[1])
plt.show()

fig,axes=plt.subplot(1,2,figsize=(16,4))
plot_acf(residuals**2,axes=axes[0])
plot_pacf(residuals**2,axes=axes[1])
plt.show()

return_predicted=arima_fitted.predict()
price_predict=hist_close+np.cumprod(1+return_predicted/100)

plt.figure(figsize=(12, 5))
return_expected = pd.DataFrame(forecasted, index=hist_testing.index)
return_lb = pd.DataFrame(forecasted_bounds[:, 0], index=hist_testing.index)
return_ub = pd.DataFrame(forecasted_bounds[:, 1], index=hist_testing.index)
plt.plot(return_expected, label='expected')
plt.plot(hist_testing, label='actual')
plt.fill_between(hist_testing.index, return_lb.values.reshape([-1]), return_ub.values.reshape([-1]), color='gray', alpha=0.7)
plt.legend()
plt.show()

from arch import arch_model
dict_aic={}
for l in range (5):
    for p in range(1,5):
        for q in range(1,5):
            try:
                split_date = hist_ret.index[-45]
                model = arch_model(hist_ret, mean='ARX', lags=l, vol='Garch', p=p, o=0, q=q, dist='Normal')
                res = model.fit(last_obs=split_date)
                dict_aic[(l, p, q)] = res.aic
            except:
                pass

df_aic = pd.DataFrame.from_dict(dict_aic, orient='index', columns=['aic'])
l, p, q = df_aic[df_aic.aic == df_aic.aic.min()].index[0]
print(f'ARIMA-GARCH order is ({l}, {p}, {q})')

# model fit
model = arch_model(hist_ret, mean='ARX', lags=l, vol='Garch', p=p, o=0, q=q, dist='Normal')
res = model.fit(last_obs=split_date)
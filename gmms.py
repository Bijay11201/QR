import numpy as np
import pandas as pd
import scipy
from datetime import datetime, timedelta
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas_datareader as pdr

end_date = datetime.today()
# start_date = end_date + timedelta(days=-5*365)
start_date = datetime(2000, 1, 1)
spx = pdr.DataReader(name='^GSPC', data_source='yahoo', start=start_date, end=end_date)
hist_close = spx['Close']
hist_ret = hist_close / hist_close.shift(1) - 1.0     # shift 1 shifts forward one day; today has yesterday's price
# hist_ret = hist_close.pct_change(1)
hist_ret.dropna(inplace=True)
hist_ret = hist_ret * 100.0
print(hist_ret.describe())

from sklearn.mixture import GaussianMixture

X = hist_ret.values.reshape(-1, 1)

n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X) for n in n_components]

plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');


# Here I choose 2 states
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(X)
labels = gmm.predict(X)



# the result suggests first state is low vol, second state is high vol.
print(gmm.means_, gmm.covariances_)


gmm.predict_proba(X)


gmm_regimes = gmm.predict(X)                  # save for latter

first_state = [1 if r == 0 else 0 for r in gmm_regimes]
second_state = [1 if r == 1 else 0 for r in gmm_regimes]
fig, axes = plt.subplots(2, figsize=(12, 8))
axes[0].plot(hist_ret.index, first_state, '.', label='first regime')
axes[1].plot(hist_ret.index, second_state, '.', label='second regime')
plt.show()


print(np.sum(first_state), np.sum(second_state), hist_ret.shape[0])
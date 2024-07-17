import os
import io
import time
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import scipy
import pandas_datareader.data as pdr
from pandas_datareader.fred import FredReader
import matplotlib.pyplot as plt
import seaborn as sns

# download CMT treasury curves from Fred
codes = ['DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
start_date = datetime(2000, 1, 1)
# end_date = datetime.today()
end_date = datetime(2020,12,31)
df = pd.DataFrame()

for code in codes:
    reader = FredReader(code, start_date, end_date)
    df0 = reader.read()
    df = df.merge(df0, how='outer', left_index=True, right_index=True, sort=False)
    reader.close()
df.dropna(axis = 0, inplace = True)
df = df['2006':]

df.tail(5)

# view the yield curve
plt.figure(figsize=(15,8))
plt.plot(df)
plt.show()

df_weekly = df.resample("W").last()
df_weekly.tail()

df_weekly_centered = df_weekly.sub(df_weekly.mean())
df_weekly_diff = df_weekly.diff()
df_weekly_diff.dropna(inplace=True)
df_weekly_diff_centered = df_weekly_diff.sub(df_weekly_diff.mean())
df_weekly.shape, df_weekly_diff.shape

# covariance
df_weekly_diff.cov()
#correlation
df_weekly_diff.corr()

from sklearn.decomposition import PCA
pca_level = PCA().fit(df_weekly)        # call fit or fit_transform
pca_change = PCA().fit(df_weekly_diff)

print(pca_change.explained_variance_)        # eigenvalues
print(pca_change.explained_variance_ratio_)     # normalized eigenvalues (sum to 1)
print(np.cumsum(pca_change.explained_variance_ratio_))

plt.plot(pca_change.explained_variance_ratio_.cumsum())
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

df_pca_level = pca_level.transform(df_weekly)            # T or PCs
df_pca_level = pd.DataFrame(df_pca_level, columns=[f'PCA_{x+1}' for x in range(df_pca_level.shape[1])])  # np.array to dataframe
df_pca_level.index = df_weekly.index
plt.figure(figsize=(15,8))
plt.plot(df_pca_level['PCA_1'], label='first component')
plt.plot(df_pca_level['PCA_2'], label='second component')
plt.plot(df_pca_level['PCA_3'], label='third component')
plt.legend()
plt.show()


df_pca_change = pca_change.transform(df_weekly_diff)             # T or PCs
df_pca_change = pd.DataFrame(df_pca_change, columns=[f'PCA_{x+1}' for x in range(df_pca_change.shape[1])])  # np.array to dataframe
df_pca_change.index = df_weekly_diff.index
plt.figure(figsize=(15,8))
plt.plot(df_pca_change['PCA_1'], label='first component')
plt.plot(df_pca_change['PCA_2'], label='second component')
plt.plot(df_pca_change['PCA_3'], label='third component')
plt.legend()
plt.show()

print(pca_change.singular_values_.shape)        # SVD singular values of sigma
print(pca_change.get_covariance().shape)       # covariance
print(pca_change.components_.shape)         # p*p, W^T

print(pca_level.components_.T[:5, :5])
print(pca_change.components_.T[:5, :5])

print(df_pca_change.iloc[:5,:5])    # df_pca: T = centered(X) * W
print(np.matmul(df_weekly_diff_centered, pca_change.components_.T).iloc[:5, :5])     # XW

np.matmul(pca_change.components_, pca_change.components_.T)[1,1], np.matmul(pca_change.components_.T, pca_change.components_)[1,1]

print(pca_change.explained_variance_[0])      # eigenvalue
print(np.dot(np.dot(pca_change.components_[0,:].reshape(1, -1), df_weekly_diff.cov()), pca_change.components_[0,:].reshape(-1, 1)))     # W^T X^TX W = lambda
print(np.dot(pca_change.components_[0,:].reshape(1, -1), df_weekly_diff.cov()))        # Ax
print(pca_change.components_[0,:]*pca_change.explained_variance_[0])                  # lambda x

df_pca_change_123 = PCA(n_components=3).fit_transform(df_weekly_diff)     
df_pca_change_123 = pd.DataFrame(data = df_pca_change_123, columns = ['first component', 'second component', 'third component'])
print(df_pca_change_123.head(5))
print(df_pca_change.iloc[:5, :3])

tenors_label = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
plt.figure(figsize=(15,4))
plt.subplot(131)
plt.plot(tenors_label, pca_change.components_[0, :])
plt.subplot(132)
plt.plot(tenors_label, pca_change.components_[1, :])
plt.subplot(133)
plt.plot(tenors_label, pca_change.components_[2, :])

T = np.matmul(df_weekly_diff_centered, pca_change.components_.T)     # T = XW
bump_up = np.zeros(T.shape[1]).reshape(1,-1)
bump_up[0,0] = 1        # first PC moves 1bps
bump_up = np.repeat(bump_up, T.shape[0], axis=0)
T_new = T+bump_up
df_weekly_diff_new = np.matmul(T_new, pca_change.components_)       # X_new = T_new * W^T
print((df_weekly_diff_new-df_weekly_diff_centered).head())          # X - X_new
print(pca_change.components_[0, :])


plt.figure(figsize=(15,8))
plt.plot(df_pca_level['PCA_3']*100, label='third component')

def mle(x):
  start = np.array([0.5, np.mean(x), np.std(x)])       # starting guess

  def error_fuc(params):
    theta = params[0]
    mu = params[1]
    sigma = params[2]

    muc = x[:-1]*np.exp(-theta) + mu*(1.0-np.exp(-theta))      # conditional mean
    sigmac = sigma*np.sqrt((1-np.exp(-2.0*theta))/(2*theta))     # conditional vol

    return -np.sum(scipy.stats.norm.logpdf(x[1:], loc=muc, scale=sigmac))

  result = scipy.optimize.minimize(error_fuc, start, method='L-BFGS-B',
                                     bounds=[(1e-6, None), (None, None), (1e-8, None)],
                                     options={'maxiter': 500, 'disp': False})
  return result.x

theta, mu, sigma  = mle(df_pca_level['PCA_3'])
print(theta, mu, sigma)
print(f'fly mean is {mu*100} bps')
print(f'half-life in week {np.log(2)/theta}')
print(f'annual standard deviation is {sigma/np.sqrt(2*theta)*100} bps, weekly {sigma/np.sqrt(2*theta)*100*np.sqrt(1/52)} bps')
print(np.mean(df_pca_change)[:3]*100, np.std(df_pca_change)[:3]*100)       # stats
print(df_pca_level['PCA_3'].tail(1)*100)      # current pca_3


fly5050 = df_weekly_diff['DGS5'] - (df_weekly_diff['DGS2']+df_weekly_diff['DGS10'])/2
plt.figure(figsize=(20,6))
plt.subplot(131)
sns.regplot(x=df_pca_change['PCA_1'], y=fly5050)
plt.subplot(132)
sns.regplot(x=df_pca_change['PCA_2'], y=fly5050)
plt.subplot(133)
sns.regplot(x=df_pca_change['PCA_3'], y=fly5050)




flymkt = df_weekly_diff['DGS5'] - (0.25*df_weekly_diff['DGS2']+0.75*df_weekly_diff['DGS10'])
plt.figure(figsize=(20,6))
plt.subplot(131)
sns.regplot(x=df_pca_change['PCA_1'], y=flymkt)
plt.subplot(132)
sns.regplot(x=df_pca_change['PCA_2'], y=flymkt)
plt.subplot(133)
sns.regplot(x=df_pca_change['PCA_3'], y=flymkt)


W = pd.DataFrame(pca_change.components_.T)
W.columns = [f'PCA_{i+1}' for i in range(W.shape[1])]
W.index = codes
w21 = W.loc['DGS2', 'PCA_1']
w22 = W.loc['DGS2', 'PCA_2']
w23 = W.loc['DGS2', 'PCA_3']

w51 = W.loc['DGS5', 'PCA_1']
w52 = W.loc['DGS5', 'PCA_2']
w53 = W.loc['DGS5', 'PCA_3']

w101 = W.loc['DGS10', 'PCA_1']
w102 = W.loc['DGS10', 'PCA_2']
w103 = W.loc['DGS10', 'PCA_3']

w551 = w51 - (w21+w101)/2.0
w552 = w52 - (w22+w102)/2.0
print(w551, w552)

A = np.array([[w21, w101],[w22,w102]])
b_ = np.array([w51, w52])
a, b = np.dot(np.linalg.inv(A), b_)
a, b

flypca = df_weekly_diff['DGS5']*1 - (a*df_weekly_diff['DGS2']+b*df_weekly_diff['DGS10'])
plt.figure(figsize=(20,6))
plt.subplot(131)
sns.regplot(x=df_pca_change['PCA_1'], y=flypca, ci=None)
plt.subplot(132)
sns.regplot(x=df_pca_change['PCA_2'], y=flypca, ci=None)
plt.subplot(133)
sns.regplot(x=df_pca_change['PCA_3'], y=flypca, ci=None)

plt.figure(figsize=(20,6))
plt.subplot(131)
plt.plot(df_pca_change['PCA_1'], flypca, 'o')
m1, b1 = np.polyfit(df_pca_change['PCA_1'], flypca, 1)
plt.plot(df_pca_change['PCA_1'], m1*df_pca_change['PCA_1']+b1)
plt.subplot(132)
plt.plot(df_pca_change['PCA_2'], flypca, 'o')
m2, b2 = np.polyfit(df_pca_change['PCA_2'], flypca, 1)
plt.plot(df_pca_change['PCA_2'], m2*df_pca_change['PCA_2']+b2)
plt.subplot(133)
plt.plot(df_pca_change['PCA_3'], flypca, 'o')
m3, b3 = np.polyfit(df_pca_change['PCA_3'], flypca, 1)
plt.plot(df_pca_change['PCA_3'], m3*df_pca_change['PCA_3']+b3)
print(f'slope 1: {m1}, 2: {m2}, 3: {m3}')
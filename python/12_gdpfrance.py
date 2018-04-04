## import numpy, panda, statsmodels
import pandas as pd
import urllib # import .dat file string
from pandas.compat import StringIO # read .dat file
import numpy as np
import scipy
from scipy.stats import f
import statsmodels.api as sm
import matplotlib as mpl
mpl.use('TkAgg') # backend adjustment
import matplotlib.pyplot as plt
from math import sqrt

## read dataset
url = 'https://raw.githubusercontent.com/sashaboulouds/statistics/master/data/France_GDP.csv'
france_gdp = pd.read_csv(url, sep=",", header=0)
df = pd.DataFrame(france_gdp)
gdp = df['Value']
x = np.arange(len(gdp))

## plot
plt.scatter(x, gdp)
plt.savefig('plot12_1.png')
plt.clf()
plt.plot(gdp)
plt.savefig('plot12_2.png')
plt.clf()

## time series
df_ts = pd.read_csv(url, parse_dates=['TIME'], index_col='TIME')
print df_ts.index
plt.plot(df_ts['Value'])
plt.savefig('plot12_3.png')
plt.clf()

autocf = sm.tsa.stattools.acf(df_ts['Value'])
plt.plot(autocf, color='r')
plt.savefig('plot12_4.png')
plt.clf()

'''
gdp=France.GDP$Value
plot(gdp)
plot(gdp,type="l")
ts(gdp)
plot(ts(gdp,start=1961,frequency = 4))
acf(gdp)
'''
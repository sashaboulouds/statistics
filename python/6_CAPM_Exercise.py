## import numpy, panda, statsmodels
import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm
import matplotlib as mpl
mpl.use('TkAgg') # backend adjustment
import matplotlib.pyplot as plt
from math import sqrt


## read csv dataset
url = "https://raw.githubusercontent.com/sashaboulouds/statistics/master/CAPM_Exercise_formated.csv"
capm = pd.read_csv(url, sep=";", header=0)
df = pd.DataFrame(capm)

## set data
y = pd.DataFrame(capm, columns = ['Return AMZN'])
x = pd.DataFrame(capm, columns = ['Returns S&P'])

## model
x_model = sm.add_constant(x)
model1 = sm.OLS(y, x_model)
result1 = model1.fit()
print result1.summary()

## plot
plt.scatter(x, y, color='black')
plt.savefig('plot6_1.png')
plt.scatter(x, y, color='black')
slope = result1.params['Returns S&P']
intercept = result1.params['const']
abline = slope * x['Returns S&P'] + intercept
plt.plot(x, abline, color='blue')
plt.savefig('plot6_2.png')
plt.clf()

## Z
Z = (result1.params['Returns S&P']-1)/(result1.bse[1])
scipy.stats.norm.cdf(Z)


'''
x<-CAPM_Exercise$`Returns S&P`
y<-CAPM_Exercise$`Return AMZN`
regression<-lm(y~x)
plot(x,y)
summary(regression)
abline(regression,col="blue")
Z<-(summary(regression)$coefficients[2,1]-1)/summary(regression)$coefficients[2,2]
pnorm(Z)
'''

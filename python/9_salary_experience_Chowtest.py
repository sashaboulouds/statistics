## import numpy, panda, statsmodels
import pandas as pd
import urllib # import .dat file string
from pandas.compat import StringIO # read .dat file
import numpy as np
import scipy
from scipy.stats import f
from scipy.stats import f_oneway
import statsmodels.api as sm
import matplotlib as mpl
mpl.use('TkAgg') # backend adjustment
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns
import matplotlib.mlab as mlab
#from Info_JR import Info

## read csv dataset
url = "https://raw.githubusercontent.com/sashaboulouds/statistics/master/data/salary_experience.csv"
salary_experience = pd.read_csv(url, sep=";", header=0)
df = pd.DataFrame(salary_experience)

## set data
salary = pd.DataFrame(salary_experience, columns = ['vy'])
experience = pd.DataFrame(salary_experience, columns = ['vx'])
education = pd.DataFrame(salary_experience, columns = ['vxz'])

## plot
plt.scatter(experience, salary, color='black')
experience_model = sm.add_constant(experience)
model1 = sm.OLS(salary, experience_model)
result1 = model1.fit()
print result1.summary()

## slope
slope = result1.params['vx']
intercept = result1.params['const']
abline = slope * experience['vx'] + intercept
plt.plot(experience, abline, color='blue')
plt.savefig('plot9_1.png')
plt.clf()
salary.describe()

## info
#Info(salary)
#Info(result1.resid)

## model1
vz = education + 2
colors = np.where(education.values==0, 'red', 'green')
colors = colors.ravel()
plt.scatter(experience, salary, c=colors)
eduexp = education['vxz']*experience['vx']
variables = pd.concat([experience, education, eduexp], axis=1)
variables_model = sm.add_constant(variables)
model2 = sm.OLS(salary, variables_model)
result2 = model2.fit()
intercept = result2.params['const']
slope = result2.params['vx']
abline_1 = slope * experience['vx'] + intercept
plt.plot(experience, abline_1, c='red')
intercept = result2.params['const'] + result2.params['vxz']
slope = result2.params['vx'] + result2.params[0]
abline_2 = slope * experience['vx'] + intercept
plt.plot(experience, abline_2, c='green')
plt.savefig('plot9_2.png')
plt.clf()
print result2.summary()
#Info(result2.resid)

## Ftest
ssru = sum(result2.resid**2)

## Restricted model
ssrr = sum(result1.resid**2)
nrestr = 2

Ftest = ((ssrr-ssru)/nrestr)/(ssru/(len(salary)-len(result2.params)))

## critical value
print f.ppf(0.95, nrestr, len(salary)-len(result2.params))
print 1 - f.cdf(Ftest, nrestr,len(salary)-len(result2.params))
print sm.stats.anova_lm(result1, result2)

mlow = df[df.vxz == 0]
mhigh = df[df.vxz == 1]

mlow_vx_model = sm.add_constant(mlow['vx'])
mhigh_vx_model = sm.add_constant(mhigh['vx'])
mregrlow = sm.OLS(mlow['vy'],mlow_vx_model)
mregrhigh = sm.OLS(mhigh['vy'],mhigh_vx_model)

ssru1 = sum(mregrlow.fit().resid**2)
ssru2 = sum(mregrhigh.fit().resid**2)
totalssru = ssru1 + ssru2
print ssru1, ssru2, totalssru


'''

#Ftest
ssru=sum(model2$residuals^2)

#restricted model
ssrr=sum(model1$residuals^2)
nrestr=2

Ftest=((ssrr-ssru)/nrestr)/(ssru/(length(salary)-length(model2$coefficients)))

#critical value
qf(0.95,nrestr,length(salary)-length(model2$coefficients))
1-pf(Ftest,nrestr,length(salary)-length(model2$coefficients))
anova(model2,model1)

mlow=subset(salary_experience,vxz==0)
mhigh=subset(salary_experience,vxz==1)

mregrlow=lm(mlow$vy~mlow$vx)
mregrhigh=lm(mhigh$vy~mhigh$vx)

ssru1=sum(mregrlow$residuals^2)
ssru2=sum(mregrhigh$residuals^2)
totalssru=ssru1+ssru2


'''

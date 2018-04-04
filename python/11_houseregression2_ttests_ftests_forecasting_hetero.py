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

## read csv dataset
url = 'https://raw.githubusercontent.com/sashaboulouds/statistics/master/data/housing.dat'
raw_data = urllib.urlopen(url)
housing_text = raw_data.read()
df = pd.read_csv(StringIO(housing_text), sep=r'\s{1,}', engine='python')

## set data
y = np.log(df['price'])
lnlotsize = np.log(df['lotsize'])
bedrms = df['bedrooms']
bathrms = df['bathrms']
airco = df['airco']
const = pd.DataFrame(np.repeat(1, len(airco)))

## concat
x = pd.concat([const, lnlotsize, bedrms, bathrms, airco], axis=1)
xprimex = x.transpose().dot(x)
xprimexinv = np.linalg.inv(xprimex)
xprimey = x.transpose().dot(y)
betahat= xprimexinv.dot(xprimey)

## regression
model1 = sm.OLS(y, x)
result1 = model1.fit()
print result1.summary()

## other matrix
epsilonhat=y-x.dot(betahat)
sigma2hat=(epsilonhat.transpose().dot(epsilonhat))/(len(y)-len(x.columns))
varbetahat = sigma2hat*xprimexinv
sebetahat= np.square(np.diag(varbetahat))

## zbetadiff
zbetadiff = (betahat[3]-betahat[2])/np.sqrt(varbetahat[3,3]+varbetahat[2,2]-2*varbetahat[2,3])

bedbath = bedrms + bathrms
variables_test = pd.concat([lnlotsize, bedbath, bathrms, airco], axis=1)
variables_test_model = sm.add_constant(variables_test)
housereg_test = sm.OLS(y, variables_test_model)
result_test = housereg_test.fit()
print result_test.summary()

## Ftest
ssru = sum(result1.resid**2)

## restricted model
airco_model = sm.add_constant(airco)
houserestrict = sm.OLS(y, airco_model)
result_houserestrict = houserestrict.fit()
ssrr = sum(result_houserestrict.resid**2)
nrestr = 3

Ftest = ((ssrr-ssru)/nrestr)/(ssru/(len(y)-len(result1.params)))

#critical value
f.ppf(0.95, nrestr, len(y)-len(result1.params))
sm.stats.anova_lm(result1, result_houserestrict)

# forecasting
lotf = np.log(5000)
bedf = 2
bathf = 2
aircof = 1

bhouse = result1.params
hforecast = bhouse[0] + bhouse[1]*lotf + bhouse[2]*bedf + bhouse[3]*bathf + bhouse[4]*aircof

lnlotsizef = lnlotsize - lotf
bedrmsf = bedrms-bedf
bathrmsf = bathrms - bathf
aircoff = airco - aircof

variables_final = pd.concat([lnlotsizef, bedrmsf, bathrmsf, aircoff], axis=1)
variables_final_model = sm.add_constant(variables_final)
houseregf = sm.OLS(y,variables_final_model)
print houseregf.fit().summary()
print result1.summary()

# to be verified
print result1.cov_HC0
print sm.stats.sandwich_covariance.cov_hac(result1)

'''
y=log(housing$price)

lnlotsize=log(housing$lotsize)
bedrms=housing$bedrooms
bathrms=housing$bathrms
airco=housing$airco
const=rep(1,length(airco))

x=cbind(const,lnlotsize,bedrms,bathrms,airco)

XprimeX=t(x)%*%x
XprimeXinv=solve(XprimeX)
XprimeY=t(x)%*%y
betahat=XprimeXinv%*%XprimeY

housereg=lm(y~lnlotsize+bedrms+bathrms+airco)
summary(housereg)

epsilonhat=y-x%*%betahat
sigma2hat=(t(epsilonhat)%*%epsilonhat)/(length(y)-ncol(x))
varbetahat=sigma2hat[1,1]*XprimeXinv
sebetahat=sqrt(diag(varbetahat))



zbetadiff=(betahat[4]-betahat[3])/sqrt(varbetahat[4,4]+varbetahat[3,3]-2*varbetahat[3,4])


bedbath=bedrms+bathrms
housereg_test=lm(y~lnlotsize+bedbath+bathrms+airco)
summary(housereg_test)


#Ftest
ssru=sum(housereg$residuals^2)

#restricted model
houserestrict=lm(y~airco)
ssrr=sum(houserestrict$residuals^2)
nrestr=3

Ftest=((ssrr-ssru)/nrestr)/(ssru/(length(y)-length(housereg$coefficients)))

#critical value
qf(0.95,nrestr,length(y)-length(housereg$coefficients))

anova(housereg,houserestrict)



#forecasting
lotf=log(5000)
bedf=2
bathf=2
aircof=1

bhouse=housereg$coefficients

hforecast=bhouse[1]+bhouse[2]*lotf+bhouse[3]*bedf+bhouse[4]*bathf+bhouse[5]*aircof

lnlotsizef=lnlotsize-lotf
bedrmsf=bedrms-bedf
bathrmsf=bathrms-bathf
aircoff=airco-aircof

houseregf=lm(y~lnlotsizef+bedrmsf+bathrmsf+aircoff)
summary(houseregf)

summary(housereg)

install.packages("sandwich")
install.packages("lmtest")

print(coeftest(housereg,vcov.=vcovHC,type=c("HC")))
print(coeftest(housereg,vcov.=NeweyWest))
'''

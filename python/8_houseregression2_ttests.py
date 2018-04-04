## import numpy, panda, statsmodels
import pandas as pd
import urllib # import .dat file string
from pandas.compat import StringIO # read .dat file
import numpy as np
import scipy
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

'''

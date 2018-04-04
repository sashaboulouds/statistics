## import numpy, panda, statsmodels
import pandas as pd
import urllib # import .dat file string
from pandas.compat import StringIO # read .dat file
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import f, norm
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
import matplotlib as mpl
mpl.use('TkAgg') # backend adjustment
import matplotlib.pyplot as plt
from math import sqrt

## read dataset
url = 'https://raw.githubusercontent.com/sashaboulouds/statistics/master/data/AER_App_data.csv'
AER_App_data = pd.read_csv(url, sep=",", header=0)
df = pd.DataFrame(AER_App_data)
game = df['cat5']

## bernlik
def bernlik(prob):
    vlog = game*np.log(prob)+(1-game)*np.log(1-prob)
    loglik = sum(vlog)
    return loglik

vgrid = np.arange(0.01,1.00, 0.01)
vloglik = np.repeat(0,len(vgrid))
for j in range(0,len(vgrid)):
    vloglik[j] = bernlik(vgrid[j])
    print vloglik[j]
vgrid = pd.DataFrame(data=vgrid, columns=['vgrid'])
vloglik = pd.DataFrame(data=vloglik, columns=['vloglik'])

plt.plot(vgrid, vloglik, color="black")
plt.savefig("plot13_1.png")
plt.clf()
mloglik = pd.concat([vgrid, vloglik], axis=1)
mlogliksort = mloglik.sort_values(by=['vloglik'])
mlogliksort.tail(1) #maximizer
game.describe()

def bernlik(prob):
    vlog = game*np.log(prob) + (1-game)*np.log(1-prob)
    loglik = sum(-vlog)
    return loglik

n = len(game)
bernmodel = scipy.optimize.minimize(fun=bernlik, x0=0.5, method='BFGS', options={'disp': True}) #log(0) error
print bernmodel
print "LogLik(bernmodel): " + str(bernmodel.fun)

# --------- #

AER_App_data_gameonly = df[df.cat5==1]
killer = AER_App_data_gameonly['killerappgros']
score = AER_App_data_gameonly['scoreapp']
killer.describe()

def probitlik(beta):
    vlog = killer*np.log(scipy.stats.norm.cdf(beta[0]+beta[1]*score))+(1-killer)*np.log(1-scipy.stats.norm.cdf(beta[0]+beta[1]*score))
    probit = sum(-vlog)
    return probit

n2 = len(killer)
x0 = [0.1, 0.1]
probitmodel = scipy.optimize.minimize(fun=probitlik, x0=x0, method='BFGS', options={'disp': True})
print probitmodel
llikunrestr = probitmodel.fun
#llikunrestr = probitmodel.x

def probitlik_restrict(beta0):
    vlog = killer*np.log(scipy.stats.norm.cdf(beta0))+(1-killer)*np.log(1-scipy.stats.norm.cdf(beta0))
    probit = sum(-vlog)
    return probit

n3 = len(killer)
probitmodel_restrict = scipy.optimize.minimize(fun=probitlik_restrict, x0=[0.1], method='BFGS', options={'disp': True})
print probitmodel_restrict
llikrestr = probitmodel_restrict.fun
#llikrestr = probitmodel_restrict.x

lrtest = 2*(llikunrestr-llikrestr)
scipy.stats.chi2.ppf(0.95, 1)

score_model = sm.add_constant(score)
probitmodel2 = sm.GLM(killer, score_model, family=sm.families.Binomial(link=sm.families.links.probit))
res = probitmodel2.fit()
print (res.summary())
'''
probitmodel2<-glm(killer ~ score,family = binomial(link = "probit"));
summary(probitmodel2)'''


'''
lrtest=2*(llikunrestr-llikrestr)
qchisq(0.95,1)

probitmodel2<-glm(killer ~ score,family = binomial(link = "probit"));
summary(probitmodel2)
'''


'''
n<-length(killer)
probitmodel_restrict<-mle(probitlik_restrict,start=list(beta0=0.1),method = "BFGS",n)
summary(probitmodel_restrict)
llikrestr=logLik(probitmodel_restrict)
'''


'''
AER_App_data <- read.csv("/Users/sashabouloudnine/Desktop/r/data/AER_App_data.csv")
game<-AER_App_data$cat5


bernlik<-function(prob)
{
  vlog<-game*log(prob)+(1-game)*log(1-prob);
  loglik<-sum(vlog);
  loglik;
}

vgrid<-seq(0.01,0.99,0.01);
vloglik<-rep(0,length(vgrid));
for (j in 1:length(vgrid)) 
{
  vloglik[j]<-bernlik(vgrid[j]);
}

plot(vgrid,vloglik,type="l");
mloglik<-cbind(vgrid,vloglik);
mlogliksort<-mloglik[order(mloglik[,2]),];
mlogliksort[nrow(mlogliksort),] #maximizer
summary(game);

bernlik<-function(prob)
{
  vlog<-game*log(prob)+(1-game)*log(1-prob);
  loglik<-sum(-vlog);
  loglik;
}

n<-length(game)
activate library "stats4"
bernmodel<-mle(bernlik,start=list(prob=0.5),method = "BFGS",n)
summary(bernmodel)
logLik(bernmodel)


##############
AER_App_data_gameonly<-AER_App_data[game==1,]
  
killer<-AER_App_data_gameonly$killerappgros
score<-AER_App_data_gameonly$scoreapp
summary(killer)


probitlik<-function(beta0,beta1)
{
  vlog<-killer*log(pnorm(beta0+beta1*score))+(1-killer)*log(1-pnorm(beta0+beta1*score));
  probit<-sum(-vlog);
}

n<-length(killer)
probitmodel<-mle(probitlik,start=list(beta0=0.1,beta1=0.1),method = "BFGS",n)
summary(probitmodel)
llikunrestr=logLik(probitmodel)



probitlik_restrict<-function(beta0)
{
  vlog<-killer*log(pnorm(beta0))+(1-killer)*log(1-pnorm(beta0));
  probit<-sum(-vlog);
}

n<-length(killer)
probitmodel_restrict<-mle(probitlik_restrict,start=list(beta0=0.1),method = "BFGS",n)
summary(probitmodel_restrict)
llikrestr=logLik(probitmodel_restrict)

lrtest=2*(llikunrestr-llikrestr)
qchisq(0.95,1)

probitmodel2<-glm(killer ~ score,family = binomial(link = "probit"));
summary(probitmodel2)

'''
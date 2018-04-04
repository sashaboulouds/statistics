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

def AR1_forecast(vy, nfor, hor):
    T=len(vy)
    Tout=np.round(nfor*T)
    Tin=T-Tout
    mforecast = np.zeros((len(Tout), len(hor)))
    mparam = np.zeros((len(Tout), 2)) #2 parameters in AR1 model

    for t in range(0,Tout):
        vyestim = vy[0:Tin + t - 1]
        Testim = len(vyestim)
        y = vyestim[1:Testim-1]
        x = vyestim[0:Testim - 2]
        x_constant = sm.add_constant(x)
        AR1_estim = sm.OLS(y,x_constant)
        result_AR1 = AR1_estim.fit()
        mparam[t,] = result_AR1.params
        vyfor = np.array([1, vyestim[Testim-1]])

        for h in range(0, hor):
            mforecast[t, h] = vyfor.dot(mparam[t,])
            vyfor = np.array([1, mforecast[t, h]])

    plt.plot(mparam[:,0])
    plt.plot(mparam[:,1])

    return (mforecast)

def forecast_performance(vy, nfor, mforecasts):
    T = len(vy)
    Tout = round(nfor*T)
    Tin = T - Tout
    vyestim = vy[0:Tin-1]
    vyout = vy[Tin:T-1]
    hor = len(mforecasts[1,])

    vrmse = np.zeros((hor,1))
    vmae = np.zeros((hor,1))

    for h in range(0, hor):
        Teval = Tout-h+2
        verror = np.zeros((Teval, 1))
        verror_sq = np.zeros((Teval, 1))

        for t in range(0,Teval):
            verror[t, 0] = (mforecasts[t, h] - vyout[t + h])
            verror_sq[t, 0] = (mforecasts[t, h] - vyout[t + h])**2
            vrmse[h, 0] = np.sqrt(np.mean(verror_sq))
            vmae[h, 0] = np.mean(verror)

    print "Forecast horizons: "
    print hor
    print "Mean Error: "
    print vmae
    print "Root Mean Squared Error: "
    print vrmse


    '''

AR1_forecast<-function(vy,nfor,hor)
{   
  T=length(vy);
  Tout=round(nfor*T);
  Tin=T-Tout;
  mforecast=array(0,dim=c(Tout,hor));
  mparam=array(0,dim=c(Tout,2)); #2 parameters in AR1 model
  
  for(t in 1:Tout)
  {
    vyestim=vy[1:Tin+t-1];
    Testim=length(vyestim);
    AR1_estim=lm(vyestim[2:Testim]~vyestim[1:Testim-1]);
    mparam[t,]=AR1_estim$coefficients;
    vyfor=c(1,vyestim[Testim]);
    
    for(h in 1:hor)
    {
      mforecast[t,h]=vyfor %*% mparam[t,];
      vyfor=c(1,mforecast[t,h]);      
    }
  }
  
  plot(mparam[,1]);  
  plot(mparam[,2]);
  
  return(mforecast)
}

forecast_performance<-function(vy,nfor,mforecasts)
{
  T=length(vy);
  Tout=round(nfor*T);
  Tin=T-Tout;
  vyestim=vy[1:Tin];
  vyout=vy[Tin+1:T];
  hor=length(mforecasts[1,]);
  
  vrmse=array(0,dim=c(hor,1));
  vmae=array(0,dim=c(hor,1));
  
  for(h in 1:hor)
  {
    Teval=Tout-h+1;
    verror=array(0,dim=c(Teval,1));
    verror_sq=array(0,dim=c(Teval,1));
    
    for(t in 1:Teval)
    {
      verror[t,1]=(mforecasts[t,h]-vyout[t+h-1]);      
      verror_sq[t,1]=(mforecasts[t,h]-vyout[t+h-1])^2;
    }
    
    vrmse[h,1]=sqrt(mean(verror_sq));
    vmae[h,1]=mean(verror);
    
  }
  
  print("Forecast horizons: ");
  print(hor)
  print("Mean Error: ");
  print(vmae);
  print("Root Mean Squared Error: ");
  print(vrmse)
}  
'''
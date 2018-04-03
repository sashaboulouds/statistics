## import numpy, panda, statsmodels
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib as mpl
mpl.use('TkAgg') # backend adjustment
import matplotlib.pyplot as plt
from math import sqrt
import random
import seaborn as sns


class OLB_bias:
    def mystats(self,x):
        # isna function
        p = np.where(np.isnan(x))
        if len(p[0]) > 0:
            x = np.delete(x, p)
            return x
        else:
            pass

        m = np.mean(x)
        n = len(x)
        s = np.std(x)
        skew = sum((x - m)**3 /s**3) / n
        kurt = sum((x - m)**4 / s**4) / n - 3
        return {'n':n,
                'mean':m,
                'stdev':s,
                'skewness':skew,
                'kurtosis':kurt}

    def OLS_bias(self,M,n,a,b):
        np.random.seed(1)
        x = np.random.randn(n)
        e = np.zeros((M,n))
        y = np.zeros((M,n))
        ahat = np.zeros((M,1))
        bhat = np.zeros((M,1))
        vara = np.zeros((M,1))
        varb = np.zeros((M,1))
        i = 0

        for i in range(0, M):
            e[i,:] = np.random.randn(n)
            y[i,:] = a+b*x+e[i,:]
            x_experience = sm.add_constant(x)
            model1 = sm.OLS(y[i,:], x_experience)
            result1 = model1.fit()
            ahat[i] = result1.params[0]
            bhat[i] = result1.params[1]
            vara[i] = result1.bse[0]
            vara[i] = result1.bse[1]

        d = {'ahat':ahat.ravel(),
             'bhat':bhat.ravel(),
             'vara':vara.ravel(),
             'varb':varb.ravel()}
        datafour = pd.DataFrame(data=d)
        print datafour
        statistics = []
        for i in datafour:
            u = OLB_bias.mystats(self,datafour[i])
            statistics.append(u)
        print statistics

        self.bias_a = datafour['ahat']
        self.bias_b = datafour['bhat']
        d = {'bias_a': self.bias_a.ravel(),
             'bias_b':self.bias_b.ravel()}
        biasab = pd.DataFrame(data=d)
        #plots
        plt.hist(ahat, 100, facecolor="red")
        plt.xlabel('Estimates of a')
        plt.ylabel('Frequency')
        plt.title('Histogram of estimates a')
        plt.savefig('plot4.png')
        plt.clf()
        plt.hist(bhat, 100, facecolor="blue")
        plt.xlabel('Estimates of b')
        plt.ylabel('Frequency')
        plt.title('Histogram of estimates b')
        plt.savefig('plot5.png')
        plt.clf()
        #density
        sns.kdeplot(np.array(ahat.ravel()), bw=0.5)
        plt.title('Kernel density of a estimates')
        plt.savefig('plot6.png')
        plt.clf()
        sns.kdeplot(np.array(bhat.ravel()), bw=0.5)
        plt.title('Kernel density of b estimates')
        plt.savefig('plot7.png')
        plt.clf()

        return biasab

    # OLS_ebias
    def OLS_ebias(self,M, n, a, b):
        bias = OLB_bias.OLS_bias(self, M, n, a, b)
        self.a = self.bias_a
        self.b = self.bias_b
        d = {'bias_a': self.a.ravel(),
             'bias_b': self.b.ravel()}
        Ebias = pd.DataFrame(data=d)

    #OLS_consistent
    def OLS_consistent(self,M,n,a,b,I):

        ee = np.zeros((I,2))
        for t in range(0, I):
            Ebias = OLB_bias.OLS_ebias(self,M,n,a,b)
            ee[t,0]= self.a
            ee[t,1] = self.b
            n=n+300
        plt.plot(ee[:,0], c='red')
        plt.xlabel('loops of adding 300 observations')
        plt.title('Trend of variance ahat')
        plt.savefig('plot8.png')
        plt.clf()
        plt.plot(ee[:,1], c='blue')
        plt.xlabel('loops of adding 300 observations')
        plt.title('Trend of variance ahat')
        plt.savefig('plot9.png')
        plt.clf()
        print ee


olb = OLB_bias()
mout = olb.OLS_bias(1000, 100, 1.5, 0.7)
olb.OLS_consistent(500,300,1.5,0.7,10)

'''
    for(i in 1:M)
    {
      e[i,]<-rnorm(n)
      y[i,]<-a+b*x+e[i,]
      model<-lm(y[i,]~x)
      ahat[i]<-model$coefficient[1]
      bhat[i]<-model$coefficient[2]
      vara[i]<-summary(model)$coefficient[1,2]
      varb[i]<-summary(model)$coefficient[2,2]
    }
    datafour<-data.frame(ahat,bhat,vara,varb)
    statistics<-sapply(datafour,mystats)
    print(statistics)
    
    bias_a<-datafour[1]#-a
    bias_b<-datafour[2]#-b
    biasab<-data.frame(bias_a,bias_b)
    colnames(biasab)<-c("bias_a","bias_b")
    
    par(mfrow=c(2,2))
    hist(datafour$ahat,breaks=100,col="red",xlab="estimates of a",main="Histogram of estimates a")
    hist(datafour$bhat,breaks=100,col="blue",xlab="estimates of a",main="Histogram of estimates b")
    kd_a<-density(datafour$ahat)
    kd_b<-density(datafour$bhat)
    plot(kd_a,main="Kernel density of a estimates",col="red")
    plot(kd_b,main="Kernel density of b estimates",col="blue")
  
    return(biasab)
    #return(statistics)
}

mout<-OLS_bias(1000,100,1.5,0.7)

##OLS_consistent
OLS_consistent<-function(M,n,a,b,I)
{
    OLS_ebias<-function(M,n,a,b)
    {
      bias<-OLS_bias(M,n,a,b)
      a<-var(bias$bias_a)
      b<-var(bias$bias_b)
      Ebias<-data.frame(a,b)
    }
    
    ee<-matrix(nrow=I,ncol=2)
    for(t in 1:I)
    {
      Ebias<-OLS_ebias(M,n,a,b)
      ee[t,1]<-Ebias$a
      ee[t,2]<-Ebias$b
      n<-n+300
    }
    
    plot(ee[,1],xlab="loops of adding 300 observations",ylab=" ",main="Trend of variance ahat",col="red")
    plot(ee[,2],xlab="loops of adding 300 observations",ylab=" ",main="Trend of variance bhat",col="blue")
    ee
}

OLS_consistent(500,300,1.5,0.7,10)



'''

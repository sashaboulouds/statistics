## import numpy, panda, statsmodels
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy
import matplotlib as mpl
mpl.use('TkAgg') # backend adjustment
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from math import sqrt
import random
import seaborn as sns

def Info(vy):
    # considering type(vy) = pd.dataframe
    vy_array = vy.values.ravel()

    T = len(vy)
    meany = vy.mean()
    mediany = vy.median()
    stddevy = vy.std()
    skewnessy = (((vy - meany)**3)/(stddevy**3)).mean()
    kurtosisy = (((vy - meany)**4)/(stddevy**4)).mean()

    print "Average: " + meany
    print "Median: " + mediany
    print "Standard deviation: " + stddevy
    print "Skewness: " + skewnessy
    print "Kurtosis: " + kurtosisy
    print "Percentile (0.01): " + vy.quantile(0.01)
    print "Percentile (0.25): " + vy.quantile(0.25)
    print "Percentile (0.75): " + vy.quantile(0.75)
    print "Percentile (0.99): " + vy.quantile(0.99)

    plt.plot(vy)
    plt.savefig('info_1.png')
    plt.clf()

    plt.hist(vy_array, bins=50, density=True)
    sns.kdeplot(vy_array, color='red', cut=True)
    plt.plot(vy_array, scipy.stats.norm.pdf(vy_array, meany, stddevy))
    plt.title("Histogram of returns")
    plt.xlabel("returns")
    plt.ylabel("frequency")
    plt.savefig('info_2.png')
    plt.clf()

    plt.boxplot(vy_array)
    plt.savefig('info_3.png')
    plt.clf()

    sm.qqplot(vy_array)
    plt.savefig('info_4.png')
    plt.clf()

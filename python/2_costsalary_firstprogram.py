## import numpy, panda, statsmodels
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib as mpl
mpl.use('TkAgg') # backend adjustment
import matplotlib.pyplot as plt
from math import sqrt


## read csv dataset
url = "https://raw.githubusercontent.com/sashaboulouds/statistics/master/costsalary.csv"
costsalary = pd.read_csv(url, sep=";", header=0)
df = pd.DataFrame(costsalary)

## set data
y = pd.DataFrame(costsalary, columns = ['Salary'])
x = pd.DataFrame(costsalary, columns = ['Costs'])

## descriptive statistics
print x.describe()
print y.describe()

## model
model = linear_model.LinearRegression()
model.fit(x,y)
predictions = model.predict(x)
coefficients = []
coefficients.append(model.intercept_)
coefficients.append(model.coef_)
print coefficients
print("Mean squared error: %.2f" % mean_squared_error(x, y))
print('R2 score: %.2f' % r2_score(x, y))

## plot
plt.scatter(x, y,  color='black')
plt.plot(x, predictions, color='red')
plt.savefig("plot1.png")
plt.clf()

## residuals
residuals = y-model.predict(x)
print "Residuals mean: " + str(int(residuals.mean()))
plt.plot(residuals)
plt.savefig("residuals.png")
plt.clf()

## sigma2hat
sigma2hat = sum(residuals^2)/(len(x) - len(coefficients))
sqrt(sigma2hat)
varbetahat = sigma2hat/(np.var(x)*(len(x)-1))
sqrt(varbetahat)
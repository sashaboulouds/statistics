## import numpy, panda, statsmodels
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib as mpl
mpl.use('TkAgg') # backend adjustment
import matplotlib.pyplot as plt
from math import sqrt


## read csv dataset
url = "https://raw.githubusercontent.com/sashaboulouds/statistics/master/salary_experience.csv"
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

slope = result1.params['vx']
intercept = result1.params['const']
abline = slope * experience['vx'] + intercept
plt.plot(experience, abline, color='blue')
plt.savefig('plot2.png')
plt.clf()
salary.describe()
residuals = result1.resid
residuals.describe()

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
plt.savefig('plot3.png')
print result2.summary()

'''salary_experience <- read.csv("/Users/sashabouloudnine/Desktop/r/data/salary_experience.csv", sep = ";")

salary=salary_experience$vy
experience=salary_experience$vx
education=salary_experience$vxz

plot(experience,salary)
model1=lm(salary~experience)
summary(model1)
abline(a=model1$coefficients[1],b=model1$coefficients[2],col="blue")
Info(salary)
Info(model1$residuals)


vz=education+2
plot(experience,salary,col=vz)
eduexp=education*experience
model2=lm(salary~experience+education+eduexp)
abline(a=model2$coefficients[1],b=model2$coefficients[2],col="red")
abline(a=(model2$coefficients[1]+model2$coefficients[3]),b=(model2$coefficients[2]+model2$coefficients[4]),col="green")
summary(model2)
Info(model2$residuals)

'''

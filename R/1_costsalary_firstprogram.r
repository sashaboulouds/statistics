costsalary <- read.csv("/Users/sashabouloudnine/Desktop/r/data/costsalary.csv")

y=costsalary$Salary
x=costsalary$Costs

summary(y)
summary(x)

model1=lm(y~x)
summary(model1)

plot(x,y)
abline(lm(y~x))
abline(a=model1$coefficients[1],b=model1$coefficients[2],col="red")

mean(model1$residuals) 
plot(model1$residuals)
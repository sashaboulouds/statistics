salary_experience <- read.csv("/Users/sashabouloudnine/Desktop/r/data/salary_experience.csv", sep = ";")

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


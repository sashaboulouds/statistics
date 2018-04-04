CAPM_Exercise <- read.csv("/Users/sashabouloudnine/Desktop/r/data/CAPM_Exercise_formated.csv", sep = ";")

x<-CAPM_Exercise$`Returns.S.P`
y<-CAPM_Exercise$`Return.AMZN`
regression<-lm(y~x)
plot(x,y)
summary(regression)
abline(regression,col="blue")
Z<-(summary(regression)$coefficients[2,1]-1)/summary(regression)$coefficients[2,2]
pnorm(Z)


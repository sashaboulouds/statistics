salary_experience <- read.csv("/Users/sashabouloudnine/Desktop/r/data/salary_experience.csv", sep = ";")
vy=salary_experience$vy

T=length(vy);
meany=mean(vy);
mediany=median(vy);
stddevy=sqrt(var(vy));
skewnessy=mean((vy-meany)^3)/(stddevy^3); #(T/((T-1)*(T-2)))*sum(((vy-meany)/stddevy)^3);
kurtosisy=mean((vy-meany)^4)/(stddevy^4);

print("Average: ");
print(meany)
print("Median: ");
print(mediany);
print("Standard deviation ");
print(stddevy);
print("Skewness ");
print(skewnessy);
print("Kurtosis ");
print(kurtosisy);
print("percentiles (0.01,0.25,0.5,0.75,0.99)");
print(quantile(vy,probs=c(0.01,0.25,0.5,0.75,0.99)));

plot(vy,type="l");
hist(vy, probability=TRUE, breaks = 50, main = "Histogram of returns", xlab = "returns", ylab = "frequency")
lines(density(vy), col='red', lwd=2)
curve(dnorm(x,meany,stddevy),col='blue',add=TRUE)

boxplot(vy);
qqnorm(vy);


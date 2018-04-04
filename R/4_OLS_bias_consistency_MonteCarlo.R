#Code adapted from XIE-YE-HU

##OLS_bias
mystats<-function(x,na.omit=FALSE){
  if(na.omit)
    x<-x[!is.na(x)]
  m<-mean(x)
  n<-length(x)
  s<-sd(x)
  skew<-sum((x-m)^3/s^3)/n
  kurt<-sum((x-m)^4/s^4)/n-3
  return(c(n=n,mean=m,stdev=s,skewness=skew,kurtosis=kurt))
}

OLS_bias<-function(M,n,a,b)
{   
    set.seed(124)
    x<-rnorm(n)
    e<-matrix(nrow=M,ncol=n)
    y<-matrix(nrow=M,ncol=n)
    ahat<-numeric(M)
    bhat<-numeric(M)
    vara<-numeric(M)
    varb<-numeric(M)
    i<-1
    
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
y=log(housing$price)

lnlotsize=log(housing$lotsize)
bedrms=housing$bedrooms
bathrms=housing$bathrms
airco=housing$airco
const=rep(1,length(airco))

x=cbind(const,lnlotsize,bedrms,bathrms,airco)

XprimeX=t(x)%*%x
XprimeXinv=solve(XprimeX)
XprimeY=t(x)%*%y
betahat=XprimeXinv%*%XprimeY

housereg=lm(y~lnlotsize+bedrms+bathrms+airco)
summary(housereg)

epsilonhat=y-x%*%betahat
sigma2hat=(t(epsilonhat)%*%epsilonhat)/(length(y)-ncol(x))
varbetahat=sigma2hat[1,1]*XprimeXinv
sebetahat=sqrt(diag(varbetahat))



zbetadiff=(betahat[4]-betahat[3])/sqrt(varbetahat[4,4]+varbetahat[3,3]-2*varbetahat[3,4])


bedbath=bedrms+bathrms
housereg_test=lm(y~lnlotsize+bedbath+bathrms+airco)
summary(housereg_test)


#Ftest
ssru=sum(housereg$residuals^2)

#restricted model
houserestrict=lm(y~airco)
ssrr=sum(houserestrict$residuals^2)
nrestr=3

Ftest=((ssrr-ssru)/nrestr)/(ssru/(length(y)-length(housereg$coefficients)))

#critical value
qf(0.95,nrestr,length(y)-length(housereg$coefficients))

anova(housereg,houserestrict)



#forecasting
lotf=log(5000)
bedf=2
bathf=2
aircof=1

bhouse=housereg$coefficients

hforecast=bhouse[1]+bhouse[2]*lotf+bhouse[3]*bedf+bhouse[4]*bathf+bhouse[5]*aircof

lnlotsizef=lnlotsize-lotf
bedrmsf=bedrms-bedf
bathrmsf=bathrms-bathf
aircoff=airco-aircof

houseregf=lm(y~lnlotsizef+bedrmsf+bathrmsf+aircoff)
summary(houseregf)


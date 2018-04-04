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


#AER_App_data <- read.csv("~/Rombouts_ESSEC/ESSEC_Teaching/Introduction_econometrics_2016/app/P2014_1175_data/AER_App_data.csv")
game<-AER_App_data$cat5


bernlik<-function(prob)
{
  vlog<-game*log(prob)+(1-game)*log(1-prob);
  loglik<-sum(vlog);
  loglik;
}

vgrid<-seq(0.01,0.99,0.01);
vloglik<-rep(0,length(vgrid));
for (j in 1:length(vgrid)) 
{
  vloglik[j]<-bernlik(vgrid[j]);
}

plot(vgrid,vloglik,type="l");
mloglik<-cbind(vgrid,vloglik);
mlogliksort<-mloglik[order(mloglik[,2]),];
mlogliksort[nrow(mlogliksort),] #maximizer
summary(game);

bernlik<-function(prob)
{
  vlog<-game*log(prob)+(1-game)*log(1-prob);
  loglik<-sum(-vlog);
  loglik;
}

n<-length(game)
#activate library "stats4"
bernmodel<-mle(bernlik,start=list(prob=0.5),method = "BFGS",n)
summary(bernmodel)
logLik(bernmodel)


##############
AER_App_data_gameonly<-AER_App_data[game==1,]
  
killer<-AER_App_data_gameonly$killerappgros
score<-AER_App_data_gameonly$scoreapp
summary(killer)


probitlik<-function(beta0,beta1)
{
  vlog<-killer*log(pnorm(beta0+beta1*score))+(1-killer)*log(1-pnorm(beta0+beta1*score));
  probit<-sum(-vlog);
}

n<-length(killer)
probitmodel<-mle(probitlik,start=list(beta0=0.1,beta1=0.1),method = "BFGS",n)
summary(probitmodel)
llikunrestr=logLik(probitmodel)



probitlik_restrict<-function(beta0)
{
  vlog<-killer*log(pnorm(beta0))+(1-killer)*log(1-pnorm(beta0));
  probit<-sum(-vlog);
}

n<-length(killer)
probitmodel_restrict<-mle(probitlik_restrict,start=list(beta0=0.1),method = "BFGS",n)
summary(probitmodel_restrict)
llikrestr=logLik(probitmodel_restrict)

lrtest=2*(llikunrestr-llikrestr)
qchisq(0.95,1)

probitmodel2<-glm(killer ~ score,family = binomial(link = "probit"));
summary(probitmodel2)



AR1_forecast<-function(vy,nfor,hor)
{   
  T=length(vy);
  Tout=round(nfor*T);
  Tin=T-Tout;
  mforecast=array(0,dim=c(Tout,hor));
  mparam=array(0,dim=c(Tout,2)); #2 parameters in AR1 model
  
  for(t in 1:Tout)
  {
    vyestim=vy[1:Tin+t-1];
    Testim=length(vyestim);
    AR1_estim=lm(vyestim[2:Testim]~vyestim[1:Testim-1]);
    mparam[t,]=AR1_estim$coefficients;
    vyfor=c(1,vyestim[Testim]);
    
    for(h in 1:hor)
    {
      mforecast[t,h]=vyfor %*% mparam[t,];
      vyfor=c(1,mforecast[t,h]);      
    }
  }
  
  plot(mparam[,1]);  
  plot(mparam[,2]);
  
  return(mforecast)
}

forecast_performance<-function(vy,nfor,mforecasts)
{
  T=length(vy);
  Tout=round(nfor*T);
  Tin=T-Tout;
  vyestim=vy[1:Tin];
  vyout=vy[Tin+1:T];
  hor=length(mforecasts[1,]);
  
  vrmse=array(0,dim=c(hor,1));
  vmae=array(0,dim=c(hor,1));
  
  for(h in 1:hor)
  {
    Teval=Tout-h+1;
    verror=array(0,dim=c(Teval,1));
    verror_sq=array(0,dim=c(Teval,1));
    
    for(t in 1:Teval)
    {
      verror[t,1]=(mforecasts[t,h]-vyout[t+h-1]);      
      verror_sq[t,1]=(mforecasts[t,h]-vyout[t+h-1])^2;
    }
    
    vrmse[h,1]=sqrt(mean(verror_sq));
    vmae[h,1]=mean(verror);
    
  }
  
  print("Forecast horizons: ");
  print(hor)
  print("Mean Error: ");
  print(vmae);
  print("Root Mean Squared Error: ");
  print(vrmse)
}  
gdp=France.GDP$Value
plot(gdp)
plot(gdp,type="l")
ts(gdp)
plot(ts(gdp,start=1961,frequency = 4))
acf(gdp)
# to test random walk hypothesis
# dfTest <- function(x, drift=TRUE) {
#   if (drift) {
#     xBar1 = mean(x[2:length(x)]) # X_t
#     xBar2 = mean(x[1:(length(x) - 1)]) # X_{t-1}
#     rhoEst = sum((x[2:length(x)] - xBar1)*(x[1:(length(x) - 1)] - xBar2))/sum((x[1:(length(x) - 1)] - xBar2)^2)
#     return(length(x)*(rhoEst - 1))
#   }
#   else {
#     rhoEst = sum(x[2:length(x)]*x[1:(length(x) - 1)])/sum(x[1:(length(x) - 1)])
#     return(length(x)*(rhoEst - 1))
#   }
# }
library(tseries)
prices = read.csv("prices.csv", header=FALSE)$V1
price.diff = diff(prices)

# print(dfTest(prices, drift=TRUE))
# print(dfTest(prices, drift=FALSE))
x <- cumsum(rnorm(1000)) # no unit-root
example.dftest = adf.test(x)

df.test = adf.test(price.diff)
kp.test = kpss.test(price.diff)
pacf(price.diff, lag.max=30)
# pacf(x) # comparison pacf plot
acf(price.diff, lag.max=30)
acf(x) # comparison acf plot

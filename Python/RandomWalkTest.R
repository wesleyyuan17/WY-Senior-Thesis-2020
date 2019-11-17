# to test random walk hypothesis
dfTest <- function(x, drift=TRUE) {
  if (drift) {
    xBar1 = mean(x[2:length(x)]) # X_t
    xBar2 = mean(x[1:(length(x) - 1)]) # X_{t-1}
    rhoEst = sum((x[2:length(x)] - xBar1)*(x[1:(length(x) - 1)] - xBar2))/sum((x[1:(length(x) - 1)] - xBar2)^2)
    return(length(x)*(rhoEst - 1))
  }
  else {
    rhoEst = sum(x[2:length(x)]*x[1:(length(x) - 1)])/sum(x[1:(length(x) - 1)])
    return(length(x)*(rhoEst - 1))
  }
}

prices = read.csv("prices.csv", header=FALSE)$V1
prices = prices + 100
logPrices = log(prices)
print(dfTest(logPrices, drift=TRUE))
print(dfTest(logPrices, drift=FALSE))

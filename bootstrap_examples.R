library(bootstrap)


## -----------------------------------------------------------------
## examples from function bootstrap:
## -----------------------------------------------------------------
# 100 bootstraps of the sample mean
# (this is for illustration; since "mean" is a
# built in function, bootstrap(x, 100, mean) would be simpler!)
x <- rnorm(20)
theta <- function(x) {mean(x)}
results <- bootstrap(x, 100, theta)

sd(results$thetastar)
1 / sqrt(20)


# as above, but also estimate the 95th percentile
#  of the bootstrap dist'n of the mean, and    
#  its jackknife-after-bootstrap standard error
perc95 <- function(x) {quantile(x, .95)}
results <-  bootstrap(x, 100, theta, func=perc95)

results$func.thetastar

sd(results$jack.boot.val) * (length(x)-1) / sqrt(length(x))
results$jack.boot.se


# To bootstrap functions of more complex data structures,
#  write theta so that its argument x
#  is the set of observation numbers
#  and simply pass as data to bootstrap the vector 1,2,..n.
# For example, to bootstrap
#  the correlation coefficient from a set of 15 data pairs:
xdata <- matrix(rnorm(30), ncol=2)
n <- 15
theta <- function(x, xdata) { cor(xdata[x, 1], xdata[x, 2]) }
results <- bootstrap(1:n, 20, theta, xdata)


## -----------------------------------------------------------------
## examples from function bootpred:
## -----------------------------------------------------------------
# bootstrap prediction error estimation in least squares
#  regression
x <- rnorm(85)
y <- 2*x + .5*rnorm(85)
theta.fit <- function(x, y) {lsfit(x, y)}
theta.predict <- function(fit, x) {cbind(1, x) %*% fit$coef}    
sq.err <- function(y, yhat) { (y - yhat)^2 }
results <- bootpred(x, y, 20, theta.fit, theta.predict, err.meas=sq.err)

apparentErrorRate = results[[1]]
bootstrapOptimismEstimate = results[[2]]
apparentErrorRate + bootstrapOptimismEstimate

err632 = results[[3]]
err632


# for a classification problem, a standard choice 
#  for err.meas would simply count up the
#  classification errors:
miss.clas <- function(y,yhat){ 1*(yhat!=y)}
# with this specification, bootpred estimates 
#  misclassification rate


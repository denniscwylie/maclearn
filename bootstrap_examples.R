library(bootstrap)

qprobs = c(0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1)


## -----------------------------------------------------------------
## examples from function bootstrap:
## -----------------------------------------------------------------
# 100 bootstraps of the sample mean
# (this is for illustration; since "mean" is a
# built in function, bootstrap(x, 100, mean) would be simpler!)
x <- rnorm(20)
theta <- function(x) {mean(x)}
results <- bootstrap(x, 100, theta)

t(t(quantile(results$thetastar, probs=qprobs)))
sd(results$thetastar)
1 / sqrt(20-1)


# as above, but also estimate the 95th percentile
#  of the bootstrap dist'n of the mean, and    
#  its jackknife-after-bootstrap standard error
perc95 <- function(x) {quantile(x, .95)}
results <-  bootstrap(x, 100, theta, func=perc95)

t(t(quantile(results$thetastar, probs=qprobs)))

results$func.thetastar
t(t(quantile(results$thetastar, probs=0.95)))

t(t(quantile(results$jack.boot.val, probs=qprobs)))


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

t(t(quantile(results$thetastar, probs=qprobs)))


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


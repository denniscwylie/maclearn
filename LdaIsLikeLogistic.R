library(glmnet)
library(MASS)

x = data.frame(matrix(runif(120), nrow=30, ncol=4))
colnames(x) = LETTERS[1:4]

y = factor(ifelse(
    x$B - x$D + 2*rnorm(nrow(x)) > 0,
    "B",
    "A"
))

logisticMod = glm(y ~ ., data=x, family=binomial)
logisticPred = predict(logisticMod, type="response")

ldaMod = lda(y ~ ., data=x)
ldaPred = predict(ldaMod, x)$posterior[ , "B"]
logitLdaPred = log(ldaPred / (1-ldaPred))

summary(lm(logitLdaPred ~ ., data=x))

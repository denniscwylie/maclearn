library(class)
library(ggplot2)

source("SimData.R")


x2_train = simulate2Group(n=100, p=2, effect=rep(1.25, 2))
knnResub = knn(
    train = x2_train$x,
    test = x2_train$x,
    cl = x2_train$y,
    k = 3
)
sum(diag(table(knnResub, x2_train$y)))
x2_test = simulate2Group(n=100, p=2, effect=rep(1.25, 2))
knnTest = knn(
    train = x2_train$x,
    test = x2_test$x,
    cl = x2_train$y,
    k = 3
)
sum(diag(table(knnTest, x2_test$y)))


parGrid = expand.grid(
    n = 100,
    p = c(2, 5, 10, 25, 100, 500),
    k = c(3, 5, 10, 25)
)
parGrid$effect = rep(2.5 / sqrt(parGrid$p))
rownames(parGrid) = paste0("p", parGrid$p, "_k", parGrid$k)

knnSimulate = function(param) {
    param = as.list(param)
    trainSet = simulate2Group(n=param$n, p=param$p,
            effect=rep(param$effect, param$p))
    testSet = simulate2Group(n=param$n, p=param$p,
            effect=rep(param$effect, param$p))
    out = list(
        p = param$p,
        k = param$k,
        train = trainSet,
        test = testSet,
        resubPreds = knn(trainSet$x, trainSet$x, trainSet$y,
                k=param$k),
        resubProbs = knn(trainSet$x, trainSet$x, trainSet$y,
                k=param$k, prob=TRUE),
        testPreds = knn(trainSet$x, testSet$x, trainSet$y,
                k=param$k),
        testProbs = knn(trainSet$x, testSet$x, trainSet$y,
                k=param$k, prob=TRUE)
    )
    out$resubTable = table(
        Predicted = out$resubPreds,
        Actual = trainSet$y
    )
    out$resubAccuracy = sum(diag(out$resubTable)) /
            sum(out$resubTable)
    out$testTable = table(
        Predicted = out$testPreds,
        Actual = testSet$y
    )
    out$testAccuracy = sum(diag(out$testTable)) /
            sum(out$testTable)
    return(out)
}


set.seed(123)
knnResults = apply(X=parGrid, MARGIN=1, FUN=knnSimulate)


set.seed(123)
repeatedKnnResults = lapply(X=1:10, FUN=function(...) {
    apply(X=parGrid, MARGIN=1, FUN=knnSimulate)
})
repeatedKnnResults = do.call(c, args=repeatedKnnResults)
knnResultsSimplified = data.frame(do.call(rbind, args=lapply(
    X = repeatedKnnResults,
    FUN = function(x) {
        outColnames = c("p", "k", "resubAccuracy", "testAccuracy")
        out = x[outColnames]
        return(structure(as.numeric(out), names=outColnames))
    }
)))


ggdata = rbind(
    data.frame(
        p = knnResultsSimplified$p,
        k = paste0("k=", knnResultsSimplified$k),
        type = "resub",
        Accuracy = knnResultsSimplified$resubAccuracy
    ),
    data.frame(
        p = knnResultsSimplified$p,
        k = paste0("k=", knnResultsSimplified$k),
        type = "test",
        Accuracy = knnResultsSimplified$testAccuracy
    )
)
ggdata$k = factor(as.character(ggdata$k),
        levels=c("k=3", "k=5", "k=10", "k=25"))

ggobj = ggplot(
    data = ggdata,
    mapping = aes(x=p, y=Accuracy,
            color=type, group=type, linetype=type)
) + theme_bw()
ggobj = ggobj + scale_x_log10()
ggobj = ggobj + geom_point(alpha=0.6)
ggobj = ggobj + stat_smooth(method.args=list(degree=1))
ggobj = ggobj + facet_wrap(~k)
## pdf("KnnSim.pdf", h=5, w=5*1.175)
print(ggobj)
## garbage = dev.off()


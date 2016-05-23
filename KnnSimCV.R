library(caret)
library(class)
library(ggplot2)

source("SimData.R")


x2_train = simulate2Group(n=100, p=2, effect=rep(1.25, 2))
nFold = 5
knnCV = train(
    x = x2_train$x,
    y = x2_train$y,
    method = "knn",
    tuneGrid = data.frame(k=3),
    trControl = trainControl(method="cv", number=nFold)
)
knnCV$results

x2_test = simulate2Group(n=100, p=2, effect=rep(1.25, 2))
knnTest = predict(knnCV, x2_test$x)
sum(diag(table(knnTest, x2_test$y))) / sum(table(knnTest, x2_test$y))


parGrid = expand.grid(
    n = 100,
    p = c(2, 5, 10, 25, 100, 500),
    k = c(3, 5, 10, 25)
)
parGrid$effect = rep(2.5 / parGrid$p)
rownames(parGrid) = paste0("p", parGrid$p, "_k", parGrid$k)

knnSimulate = function(param, nFold=5) {
    param = as.list(param)
    trainSet = simulate2Group(n=param$n, p=param$p,
            effect=rep(param$effect, param$p))
    testSet = simulate2Group(n=param$n, p=param$p,
            effect=rep(param$effect, param$p))
    knnCaretControl = trainControl(method="cv", number=nFold)
    knnCV = train(
        x = trainSet$x,
        y = trainSet$y,
        method = "knn",
        tuneGrid = data.frame(k=param$k),
        trControl = trainControl(method="cv", number=nFold)
    )
    out = list(
        p = param$p,
        k = param$k,
        train = trainSet,
        test = testSet,
        testPreds = predict(knnCV, testSet$x),
        testProbs = predict(knnCV, testSet$x, type="prob")[ , 2]
    )
    out$cvAccuracy = knnCV$results[ , "Accuracy"]
    out$testTable = table(
        Predicted = out$testPreds,
        Actual = testSet$y
    )
    out$testAccuracy = sum(diag(out$testTable)) /
            sum(out$testTable)
    return(out)
}


set.seed(123)
repeatedKnnResults = lapply(X=1:5, FUN=function(...) {
    apply(X=parGrid, MARGIN=1, FUN=knnSimulate)
})
repeatedKnnResults = do.call(c, args=repeatedKnnResults)
knnResultsSimplified = data.frame(do.call(rbind, args=lapply(
    X = repeatedKnnResults,
    FUN = function(x) {
        outColnames = c("p", "k", "cvAccuracy", "testAccuracy")
        out = x[outColnames]
        return(structure(as.numeric(out), names=outColnames))
    }
)))

ggdata = rbind(
    data.frame(
        p = knnResultsSimplified$p,
        k = paste0("k=", knnResultsSimplified$k),
        type = "cv",
        Accuracy = knnResultsSimplified$cvAccuracy
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
## pdf("KnnSimCV.pdf", h=5, w=5*1.175)
print(ggobj)
## garbage = dev.off()

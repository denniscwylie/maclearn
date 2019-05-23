library(caret)
library(class)
library(ggplot2)

source("modelpipe.R")

load("prepared_datasets.RData")



fitModelWithNFeat = function(fitter, n, setname,
        fold=5, seed=123) {cat(class(fitter), '\n')
    if (n > ncol(xnorms[[setname]])) {
        return(NA)
    }
    fsFitter = SolderedPipeFitter(
        FastTSelector(nFeat=n),
        fitter
    )
    fit = train(
        fsFitter,
        xnorms[[setname]],
        ys[[setname]],
        trControl = trainControl(
            method = "cv",
            number = fold,
            seeds = as.list(rep(seed, times=fold+1))
        )
    )
    return(list(
        fit = fit,
        acc = fit$results$Accuracy
    ))
}

fitters = list(
    knn5 = KnnFitter(k=5),
    knn9 = KnnFitter(k=9),
    logistic = GlmFitter(lambda=1e-10),
    l1 = GlmFitter(alpha=1, lambda=NULL),
    l2 = GlmFitter(alpha=0, lambda=NULL),
    lda = LdaFitter(),
    dlda = dldaFitter,
    rf = RandomForestFitter(),
    ada = AdaFitter()
)

xnames = names(xnorms)
names(xnames) = xnames


modelFits10 = lapply(X=xnames, FUN=function(setname) {
    return(lapply(X=fitters, FUN=function(fitter) {
        fitModelWithNFeat(fitter=fitter, n=10, setname=setname)
    }))
})
modelFits10Accs = sapply(
    X = modelFits10,
    FUN = function(u) {sapply(u, function(v) {v$fit$results$Accuracy})}
)
write.table(data.frame(M=rownames(modelFits10Accs), modelFits10Accs),
        file="modelFits10Accs.tsv", sep="\t", row.names=FALSE)


## modelFits20 = lapply(X=xnames, FUN=function(setname) {
##     return(lapply(X=fitters, FUN=function(fitter) {
##         fitModelWithNFeat(fitter=fitter, n=20, setname=setname)
##     }))
## })
## modelFits20Accs = sapply(
##     X = modelFits20,
##     FUN = function(u) {sapply(u, function(v) {v$acc})}
## )
## write.table(data.frame(M=rownames(modelFits20Accs), modelFits20Accs),
##         file="modelFits20Accs.tsv", sep="\t", row.names=FALSE)


modelFits50 = lapply(X=xnames, FUN=function(setname) {
    return(lapply(X=fitters, FUN=function(fitter) {
        fitModelWithNFeat(fitter=fitter, n=50, setname=setname)
    }))
})
modelFits50Accs = sapply(
    X = modelFits50,
    FUN = function(u) {sapply(u, function(v) {v$acc})}
)
write.table(data.frame(M=rownames(modelFits50Accs), modelFits50Accs),
        file="modelFits50Accs.tsv", sep="\t", row.names=FALSE)


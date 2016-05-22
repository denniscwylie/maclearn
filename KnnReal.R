library(caret)
library(class)
library(ggplot2)

source("modelpipe.R")

## source("LoadData.R")
## source("NormalizeData.R")
## source("RestrictData.R")
## source("ExtractYs.R")
load("prepared_datasets.RData")


knnModels = mapply(
    FUN = train,
    xnorms,
    ys,
    MoreArgs = list(
        method = "knn",
        tuneGrid = data.frame(k=3),
        trControl = trainControl(
            method = "cv",
            number = 5,
            seeds = as.list(rep(123, 6))
        )
    ),
    SIMPLIFY = FALSE
)
knnCvAccs = sapply(knnModels, function(u) {u$results$Accuracy})


## -----------------------------------------------------------------
## try with univariate filter feature selection
## -----------------------------------------------------------------
fsKnnFitter = SolderedPipeFitter(
    FastTSelector(nFeat = 10),
    KnnFitter(k = 3)
)

fsKnnFits = mapply(
    FUN = train,
    lapply(1:length(xnorms), function(...) {fsKnnFitter}),
    xnorms,
    ys,
    MoreArgs = list(
        trControl = trainControl(
            method = "cv",
            number = 5,
            seeds = as.list(rep(123, 6))
        )
    ),
    SIMPLIFY = FALSE
)


## -----------------------------------------------------------------
## vary number of features used
## -----------------------------------------------------------------
nFeatures = c(1, 2, 5, 10, 20, 50, 100, 200, 500,
        1000, 2000, 5000, 10000)
names(nFeatures) = as.character(nFeatures)
fitKnnWithNFeat = function(n, setname, fold=5, seed=123) {
    if (n > ncol(xnorms[[setname]])) {
        return(NA)
    }
    fsKnnFitter = SolderedPipeFitter(
        FastTSelector(nFeat = n),
        KnnFitter(k = 3)
    )
    fit = train(
        fsKnnFitter,
        xnorms[[setname]],
        ys[[setname]],
        trControl = trainControl(
            method = "cv",
            number = fold,
            seeds = as.list(rep(seed, times=fold+1))
        )
    )
    return(fit$results$Accuracy)
}

## xnames = names(xnorms)
## names(xnames) = xnames
## accsByNFeats = lapply(
##     X = xnames,
##     FUN = function(s) {
##         lapply(nFeatures, fitKnnWithNFeat, setname=s)
##     }
## )
## accsByNFeats = unlist(accsByNFeats)

## ggdata = data.frame(acc=accsByNFeats, row.names=names(accsByNFeats))
## ggdata$set = factor(gsub("\\..*", "",  names(accsByNFeats)),
##         levels=names(xnorms))
## ggdata$p = as.integer(gsub(".*\\.", "", names(accsByNFeats)))
## write.table(
##     ggdata,
##     file = "KnnRealAccuracyByNFeat_R.tsv",
##     sep = "\t",
##     quote = FALSE,
##     row.names = FALSE
## )

ggdata = read.table(
    "KnnRealAccuracyByNFeat_R.tsv",
    sep = "\t",
    row.names = NULL,
    header = TRUE
)

ggdata$set = factor(as.character(ggdata$set), levels=names(xnorms))
ggobj = ggplot(data=ggdata, mapping=aes(x=p, y=acc, color=set))
ggobj = ggobj + geom_point()
ggobj = ggobj + geom_line(alpha=0.5)
ggobj = ggobj + scale_x_log10(breaks=c(10, 100, 1000, 10000))
ggobj = ggobj + theme_classic()
ggobj = ggobj + scale_color_manual(
        values=c("darkgray", "black", "red", "dodgerblue3"))
ggobj = ggobj + ylab("Accuracy (5-fold CV)")
pdf("KnnRealAccuracyByNFeat.pdf", h=5, w=5*1.325)
print(ggobj)
garbage = dev.off()


## -----------------------------------------------------------------
## use PCA feature extraction
## -----------------------------------------------------------------
feKnnFitter = SolderedPipeFitter(
    PcaExtractor(k = 3),
    KnnFitter(k = 3)
)

## feKnnFits = mapply(
##     FUN = train,
##    lapply(1:length(xnorms), function(...) {feKnnFitter})
##     xnorms,
##     ys,
##     MoreArgs = list(
##         trControl = trainControl(
##             method = "cv",
##             number = 5,
##             seeds = as.list(rep(123, 6))
##         )
##     ),
##     SIMPLIFY = FALSE
## )
## save(feKnnFits, file="feKnnFits.RData")
load("feKnnFits.RData")


## -----------------------------------------------------------------
## test with varying number of principal components
## -----------------------------------------------------------------
npcs = c(1, 2, 5, 10, 20, 50, 100, 200)
names(npcs) = as.character(npcs)
fitKnnWithNPcs = function(n, setname, fold=5, seed=123) {
    if (n > floor((1-1/fold)*min(dim(xnorms[[setname]])))) {
        return(NA)
    }
    feKnnFitter = SolderedPipeFitter(
        PcaExtractor(k = n),
        KnnFitter(k = 3)
    )
    fit = train(
        feKnnFitter,
        xnorms[[setname]],
        ys[[setname]],
        trControl = trainControl(
            method = "cv",
            number = fold,
            seeds = as.list(rep(seed, times=fold+1))
        )
    )
    return(fit$results$Accuracy)
}

## xnames = names(xnorms)
## names(xnames) = xnames
## accsByNPcs = lapply(
##     X = xnames,
##     FUN = function(s) {
##         lapply(npcs, fitKnnWithNPcs, setname=s)
##     }
## )
## accsByNPcs = unlist(accsByNPcs)

## ggdata = data.frame(acc=accsByNPcs, row.names=names(accsByNPcs))
## ggdata$set = factor(gsub("\\..*", "",  names(accsByNPcs)),
##         levels=names(xnorms))
## ggdata$p = as.integer(gsub(".*\\.", "", names(accsByNPcs)))
## write.table(
##     ggdata,
##     file = "KnnRealAccuracyByNPcs_R.tsv",
##     sep = "\t",
##     quote = FALSE,
##     row.names = FALSE
## )

ggdata = read.table(
    "KnnRealAccuracyByNPcs_R.tsv",
    sep = "\t",
    row.names = NULL,
    header = TRUE
)

ggdata$set = factor(as.character(ggdata$set), levels=names(xnorms))
ggobj = ggplot(data=ggdata, mapping=aes(x=p, y=acc, color=set))
ggobj = ggobj + geom_point()
ggobj = ggobj + geom_line(alpha=0.5)
ggobj = ggobj + scale_x_log10(breaks=c(10, 100, 1000, 10000))
ggobj = ggobj + theme_classic()
ggobj = ggobj + scale_color_manual(
        values=c("darkgray", "black", "red", "dodgerblue3"))
ggobj = ggobj + ylab("Accuracy (5-fold CV)")
pdf("KnnRealAccuracyByNPcs.pdf", h=5, w=5*1.325)
print(ggobj)
garbage = dev.off()

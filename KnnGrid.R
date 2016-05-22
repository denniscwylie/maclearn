library(caret)
library(class)
library(ggplot2)

source("modelpipe.R")

## source("LoadData.R")
## source("NormalizeData.R")
## source("RestrictData.R")
## source("ExtractYs.R")
load("prepared_datasets.RData")

## Note that caret has some nice built-in capabilities
## for tuning model parameters over a grid of potential values...
## Not using them here both for more explicit illustration
## and b/c I haven't yet made SolderedPipeFitter
## objects compatible with these capabilities.


fsKnnFitterGenerator = function(k) {
    return(SolderedPipeFitter(
        FastTSelector(nFeat = 10),
        KnnFitter(k = k)
    ))
}

setnames = names(xnorms)
names(setnames) = setnames
ks = c(3, 5, 9, 15)
names(ks) = as.character(ks)

## lapply below takes a while to run!
knnModels = lapply(
    X = setnames,
    FUN = function(setname) {
        return(lapply(
            X = ks,
            FUN = function(k) {
                return(train(
                    fsKnnFitterGenerator(k),
                    xnorms[[setname]],
                    ys[[setname]],
                    trControl = trainControl(
                        method = "cv",
                        number = 5,
                        seeds = as.list(rep(123, 6))
                    )
                ))
            }
        ))
    }
)
## ## save(knnModels, file="knnGridModels.RData")
## load("knnGridModels.RData")

knnCvAccs = sapply(knnModels, function(kmods) {
    sapply(kmods, function(kmod) {kmod$results$Accuracy})
})

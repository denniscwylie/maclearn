library(caret)
library(randomForest)
library(ggplot2)

source("MaclearnUtilities.R")
source("modelpipe.R")

load("prepared_datasets.RData")

source("fitModelWithNFeat.R")


fsRf100Models = lapply(xnames, fitModelWithNFeat,
                       fitter=RandomForestFitter(ntree=100), n=10)
fsRf100Accs = sapply(fsRf100Models, function(u) {u$results$Accuracy})

fsRf500Models = lapply(xnames, fitModelWithNFeat,
                       fitter=RandomForestFitter(ntree=500), n=10)
fsRf500Accs = sapply(fsRf500Models, function(u) {u$results$Accuracy})

fsRf2500Models = lapply(xnames, fitModelWithNFeat,
                        fitter=RandomForestFitter(ntree=2500), n=10)
fsRf2500Accs = sapply(fsRf2500Models, function(u) {u$results$Accuracy})

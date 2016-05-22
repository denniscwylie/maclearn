library(caret)
library(randomForest)
library(ggplot2)

source("MaclearnUtilities.R")
source("modelpipe.R")

load("prepared_datasets.RData")

source("fitModelWithNFeat.R")


fsAda50Models = lapply(xnames, fitModelWithNFeat, fitter=AdaFitter(
	iter = 50,
	bag.frac = 1,
	control = rpart.control(maxdepth=3)
), n=10)
fsAda50Accs = sapply(fsAda50Models, function(u) {u$results$Accuracy})

fsAda100Models = lapply(xnames, fitModelWithNFeat, fitter=AdaFitter(
	iter = 100,
	bag.frac = 1,
	control = rpart.control(maxdepth=3)
), n=10)
fsAda100Accs = sapply(fsAda100Models, function(u) {u$results$Accuracy})

fsAda250Models = lapply(xnames, fitModelWithNFeat,
		fitter=AdaFitter(iter=250, bag.frac=1), n=10)
fsAda250Accs = sapply(fsAda250Models, function(u) {u$results$Accuracy})

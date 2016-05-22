library(caret)
library(class)
library(ggplot2)

library(devtools)
load_all("~/workspace/modelpipe")

## source("LoadData.R")
## source("NormalizeData.R")
## source("RestrictData.R")
## source("ExtractYs.R")
load("prepared_datasets.RData")


fitModelWithNFeat = function(fitter, n, setname,
		fold=5, seed=123) {
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
	return(fit$results$Accuracy)
}

xnames = names(xnorms)
names(xnames) = xnames

accPlot = function(accsByNFeats, dataFile, plotFile) {
	ggdata = data.frame(acc=accsByNFeats, row.names=names(accsByNFeats))
	ggdata$set = factor(gsub("\\..*", "",  names(accsByNFeats)),
			levels=names(xnorms))
	ggdata$p = as.integer(gsub(".*\\.", "", names(accsByNFeats)))

	write.table(
		ggdata,
		file = dataFile,
		sep = "\t",
		quote = FALSE,
		row.names = FALSE
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

	pdf(plotFile, h=5, w=5*1.325)
	print(ggobj)
	garbage = dev.off()

	invisible(list(data=ggdata, plot=ggobj))
}


nFeatures = c(2, 5, 10, 20, 50, 100, 200, 500,
		1000, 2000, 5000, 10000)
names(nFeatures) = as.character(nFeatures)


## -----------------------------------------------------------------
## no (err...very little) regularization
## -----------------------------------------------------------------
fitLogisticWithNFeat = function(...) {
	fitModelWithNFeat(fitter=GlmFitter(alpha=0, lambda=1e-10), ...)
}

accsByNFeats = lapply(
	X = xnames,
	FUN = function(s) {
		lapply(nFeatures, fitLogisticWithNFeat, setname=s)
    }
)
accsByNFeats = unlist(accsByNFeats)

accPlot(
	accsByNFeats,
	dataFile = "LogisticRealAccuracyByNFeat_R.tsv",
	plotFile = "LogisticRealAccuracyByNFeat.pdf"
)


## -----------------------------------------------------------------
## L2 regularization
## -----------------------------------------------------------------
fitL2LogisticWithNFeat = function(...) {
	fitModelWithNFeat(fitter=GlmFitter(alpha=0, lambda=NULL), ...)
}

accsByNFeatsL2 = lapply(
	X = xnames,
	FUN = function(s) {
		lapply(nFeatures, fitL2LogisticWithNFeat, setname=s)
    }
)
accsByNFeatsL2 = unlist(accsByNFeatsL2)

l2AccResults = accPlot(
	accsByNFeatsL2,
	dataFile = "L2LogisticRealAccuracyByNFeat_R.tsv",
	plotFile = "L2LogisticRealAccuracyByNFeat.pdf"
)


## -----------------------------------------------------------------
## L1 regularization
## -----------------------------------------------------------------
fitL1LogisticWithNFeat = function(...) {
	fitModelWithNFeat(fitter=GlmFitter(alpha=1, lambda=NULL), ...)
}

accsByNFeatsL1 = lapply(
	X = xnames,
	FUN = function(s) {
		lapply(nFeatures, fitL1LogisticWithNFeat, setname=s)
    }
)
accsByNFeatsL1 = unlist(accsByNFeatsL1)

l1AccResults = accPlot(
	accsByNFeatsL1,
	dataFile = "L1LogisticRealAccuracyByNFeat_R.tsv",
	plotFile = "L1LogisticRealAccuracyByNFeat.pdf"
)




## should be able to remove below once verified above...

## ggdata = data.frame(acc=accsByNFeats, row.names=names(accsByNFeats))
## ggdata$set = factor(gsub("\\..*", "",  names(accsByNFeats)),
## 		levels=names(xnorms))
## ggdata$p = as.integer(gsub(".*\\.", "", names(accsByNFeats)))
## write.table(
## 	ggdata,
## 	file = "LogisticRealAccuracyByNFeat_R.tsv",
## 	sep = "\t",
## 	quote = FALSE,
## 	row.names = FALSE
## )

ggdata = read.table(
	"LogisticRealAccuracyByNFeat_R.tsv",
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
pdf("LogisticRealAccuracyByNFeat.pdf", h=5, w=5*1.325)
print(ggobj)
garbage = dev.off()


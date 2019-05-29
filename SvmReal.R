library(caret)
library(e1071)
library(ggplot2)

source("MaclearnUtilities.R")
source("modelpipe.R")

load("prepared_datasets.RData")

source("fitModelWithNFeat.R")


fsSvmLinModels = lapply(xnames, fitModelWithNFeat,
                        fitter=SvmFitter(kernel="linear", cost=1), n=10)
fsSvmLinAccs = sapply(fsSvmLinModels, function(u) {u$results$Accuracy})

fsSvmRadModels = lapply(xnames, fitModelWithNFeat,
                        fitter=SvmFitter(kernel="radial", cost=1), n=10)
fsSvmRadAccs = sapply(fsSvmRadModels, function(u) {u$results$Accuracy})


## -----------------------------------------------------------------
## contour plot examples
## -----------------------------------------------------------------
source("~/workspace/miscr/Plotting.R")

svmPlot = function(kernel="radial", cost=1, gamma=1/2, ...) {
    y = factor(gsub("subtype: ", "", ys$patel))
    names(y) = rownames(xnorms$patel)
    svmMod = SvmFitter(kernel=kernel, cost=cost, gamma=gamma)(
                       xnorms$patel[ , c("NAMPT", "CFI")], y)
    svmPred = function(x, y) {
        svmMod$predict(x=data.frame(NAMPT=x, CFI=y))
    }
    ggobj = ggfuntile(svmPred, xrange=c(-7.25, 6.25), yrange=c(-4.25, 7.5),
                      density=201, zlab="P(Pro)", xlab="NAMPT", ylab="CFI")
    svm2 = xnorms$patel[ , c("NAMPT", "CFI")]
    svm2$z = svmPred(svm2[ , 1], svm2[ , 2])
    svm2$class = y[rownames(svm2)]
    ggo2 = ggobj + geom_point(data=svm2,
                              aes(x=NAMPT, y=CFI, shape=class),
                              color="white", size=3, alpha=0.8) +
                   scale_shape_manual(values=c(6, 17), guide=FALSE)
    cost = gsub("\\.", "p", as.character(round(cost, 2)))
    gamma = gsub("\\.", "p", as.character(round(gamma, 2)))
    png(paste0("svm_", kernel, "_c", cost, "_g", gamma, "_contour.png"),
        h=1920, w=1920*1.2, res=288*1.25)
    print(ggo2)
    garbage = dev.off()
}

svmPlot("linear", cost=1)

svmPlot("radial", cost=0.2, gamma=1/2)
svmPlot("radial", cost=1, gamma=1/2)
svmPlot("radial", cost=5, gamma=1/2)

svmPlot("radial", cost=1, gamma=0.2/2)
svmPlot("radial", cost=0.2, gamma=1/2)
svmPlot("radial", cost=1, gamma=5/2)
svmPlot("radial", cost=1, gamma=25/2)
svmPlot("radial", cost=1, gamma=125/2)

## svmPlot("radial", cost=0.1)
## svmPlot("radial", cost=1, gamma=25/ncol(xnorms$patel))

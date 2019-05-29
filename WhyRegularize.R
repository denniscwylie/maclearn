library(caret)
library(GGally)
library(glmnet)
library(pheatmap)

source("modelpipe.R")

source("MaclearnUtilities.R")


## -----------------------------------------------------------------
## linear regression simulated example
## -----------------------------------------------------------------
x = data.frame(matrix(rnorm(60), nrow=15, ncol=4))
colnames(x) = LETTERS[1:4]
x$B = x$A + 0.01*x$B

y = x$D + rnorm(nrow(x))

linmod = lm(y ~ ., data=x)
coef(linmod)

l2mod = glmnet(x=as.matrix(x), y=y, alpha=0, lambda=0.1)
coef(l2mod)

l1mod = glmnet(x=as.matrix(x), y=y, alpha=1, lambda=0.1)
coef(l1mod)


## -----------------------------------------------------------------
## load Hess data
## -----------------------------------------------------------------
readTab = function(file) {
    read.table(file, sep="\t",
               header=TRUE, row.names=1, check.names=FALSE)
}

x = data.frame(
    t(readTab("microarray/Hess/HessTrainingData.tsv.gz")),
    check.names = FALSE
)
annot = readTab("microarray/Hess/HessTrainingAnnotation.tsv")
y = annot$pCRtxt
names(y) = rownames(annot)

logisticFitter = SolderedPipeFitter(
    FastTSelector(nFeat = 4),
    GlmFitter(alpha=0, lambda=0)
)
logisticFit = logisticFitter(x, y)
logisticCoef = coef(logisticFit[[2]]$fit)

heatX = x[ , setdiff(rownames(logisticCoef), "(Intercept)")]
heatY = data.frame(row.names=names(y), Group=y)
pheatmap(
    heatX,
    annotation_row = heatY,
    annotation_color = list(Group=c(pCR="black", RD="gray")),
    show_rownames=FALSE
)

ggpairs(data.frame(heatX, y=y))


## -----------------------------------------------------------------
## regularized models
## -----------------------------------------------------------------
l2Fitter = SolderedPipeFitter(
    FastTSelector(nFeat = 4),
    GlmFitter(alpha=0, lambda=0.05)
)
l2Fit = l2Fitter(x, y)
l2Coef = coef(l2Fit[[2]]$fit)

l1Fitter = SolderedPipeFitter(
    FastTSelector(nFeat = 4),
    GlmFitter(alpha=1, lambda=0.05)
)
l1Fit = l1Fitter(x, y)
l1Coef = coef(l1Fit[[2]]$fit)


## -----------------------------------------------------------------
## try with cross-validation
## -----------------------------------------------------------------
cvLogistic = train(
    logisticFitter,
    x,
    y,
    trControl = trainControl(
        method = "cv",
        number = 5,
        seeds = as.list(rep(123, 6))
    )
)
cvLogistic$results

cvL1 = train(
    l1Fitter,
    x,
    y,
    trControl = trainControl(
        method = "cv",
        number = 5,
        seeds = as.list(rep(123, 6))
    )
)
cvL1$results

cvL2 = train(
    l2Fitter,
    x,
    y,
    trControl = trainControl(
        method = "cv",
        number = 5,
        seeds = as.list(rep(123, 6))
    )
)
cvL2$results

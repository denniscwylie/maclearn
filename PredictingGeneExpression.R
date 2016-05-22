library(caret)
library(GGally)
library(ggplot2)
library(txtplot)

source("modelpipe.R")
source("MaclearnUtilities.R")


## -----------------------------------------------------------------
## load Patel data
## -----------------------------------------------------------------
readTab = function(file) {
	if (grepl("gz$", file)) {
		file = gzfile(file)
	}
	read.table(file, sep="\t",
			header=TRUE, row.names=1, check.names=FALSE)
}

x = data.frame(t(readTab(
        "rnaseq/GSE57872/GSE57872_DataMatrixMapped.tsv.gz")),
        check.names=FALSE)
y = x$BRCA1
names(y) = rownames(x)
x0 = x[ , colnames(x) != "BRCA1"]

corPVals = apply(x0, 2,
        function(z) {cor.test(z, y)$p.value})
corQVals = p.adjust(corPVals, method="fdr")
head(sort(corQVals))
summary(lm(y ~ x0$CDK1))

ggobj = ggplot(data=x, mapping=aes(x=CDK1, y=BRCA1))
ggobj = ggobj + theme_classic()
ggobj = ggobj + geom_point(alpha=0.5)
ggobj = ggobj + stat_smooth(method="lm", se=FALSE)
print(ggobj)


## -----------------------------------------------------------------
## unregularized linear regression
## -----------------------------------------------------------------
nFeats = c(2, 5, 10, 20, 50, 100, 200, 500, 1000)
names(nFeats) = as.character(nFeats)
brca1Modelers = lapply(X=nFeats, FUN=function(n) {
    solder(
        PearsonSelector(nFeat = n),
        GlmFitter(fam="gaussian", alpha=0, lambda=0)
    )
})

brca1Model20 = brca1Modelers[["20"]](x0, y)
brca1Preds = predict(brca1Model20, x0)
txtplot(brca1Preds, y)

brca1Model1000 = brca1Modelers[["1000"]](x0, y)
brca1Preds = predict(brca1Model1000, x0)
txtplot(brca1Preds, y)

set.seed(123)
brca1CV = lapply(X=brca1Modelers, FUN=function(m) {train(m, x0, y)})

ggdata = data.frame(
    "Number Potential Features" = nFeats,
    Rsquared = sapply(brca1CV, function(m) {m$results$Rsquared}),
    Regularization = "none",
    Lambda = 0,
    check.names = FALSE
)
ggobj = ggplot(
    data = ggdata,
    mapping = aes(x=`Number Potential Features`, y=Rsquared)
)
ggobj = ggobj + theme_classic()
ggobj = ggobj + geom_point() + geom_line(alpha=0.5)
ggobj = ggobj + scale_x_log10()
print(ggobj)



## -----------------------------------------------------------------
## L2-regularized linear regression
## -----------------------------------------------------------------
brca1Modelers2 = lapply(X=nFeats, FUN=function(n) {
    solder(
        PearsonSelector(nFeat = n),
        GlmFitter(fam="gaussian", alpha=0, lambda=NULL)
    )
})

set.seed(123)
brca1CV2 = lapply(X=brca1Modelers2,
		FUN=function(m) {train(m, x0, y)})

lambdaMins2 = sapply(brca1CV2, FUN=function(z) {
    z$finalModel[[2]]$fit$lambda.min
})
summary(lm(lambdaMins2 ~ nFeats))
## approx 1.5 + 0.034 * nFeats

ggdata = rbind(ggdata, data.frame(
    "Number Potential Features" = nFeats,
    Rsquared = sapply(brca1CV2, function(m) {m$results$Rsquared}),
    Regularization = "L2/ridge",
    Lambda = lambdaMins2,
    check.names = FALSE
))


## -----------------------------------------------------------------
## L1-regularized linear regression
## -----------------------------------------------------------------
brca1Modelers1 = lapply(X=nFeats, FUN=function(n) {
    solder(
        PearsonSelector(nFeat = n),
        GlmFitter(fam="gaussian", alpha=1, lambda=NULL)
    )
})

set.seed(123)
brca1CV1 = lapply(X=brca1Modelers1,
		FUN=function(m) {train(m, x0, y)})

lambdaMins1 = sapply(brca1CV1, FUN=function(z) {
    z$finalModel[[2]]$fit$lambda.min
})
summary(lm(lambdaMins1 ~ log(nFeats)))
## approx -0.0157 + 0.0235 * log(nFeats)

ggdata = rbind(ggdata, data.frame(
    "Number Potential Features" = nFeats,
    Rsquared = sapply(brca1CV1, function(m) {m$results$Rsquared}),
    Regularization = "L1/Lasso",
    Lambda = lambdaMins1,
    check.names = FALSE
))

ggobj = ggplot(
    data = ggdata,
    mapping = aes(x=`Number Potential Features`, y=Rsquared,
            linetype=Regularization)
)
ggobj = ggobj + theme_classic()
ggobj = ggobj + geom_point() + geom_line(alpha=0.5)
ggobj = ggobj + scale_x_log10()
print(ggobj)

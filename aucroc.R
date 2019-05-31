library(genefilter)
library(ggplot2)
library(pROC)

source('modelpipe.R')

load("prepared_datasets.RData")

x = xnorms$hess
y = structure(ys$hess, names=rownames(x))

fsLogisticFitter = SolderedPipeFitter(
    FastTSelector(nFeat = 5),
    GlmFitter()
)

fitModel = fsLogisticFitter(x, y)


## -----------------------------------------------------------------------------
set.seed(123)
xfew = x[sample(rownames(x), 20), ]
yis1 = structure(y[rownames(xfew)] == 'pCR', names=rownames(xfew))

modelResubPreds = predict(fitModel, xfew)

thresholds = c(none=1, sort(modelResubPreds, decreasing=TRUE), all=0)

tp = sapply(thresholds, function(thresh) {sum(modelResubPreds >= thresh & yis1)})
tn = sapply(thresholds, function(thresh) {sum(modelResubPreds < thresh & !yis1)})

sens = tp / sum(yis1)
spec = tn / sum(!yis1)


## -----------------------------------------------------------------------------
aucResult = pROC::auc(as.numeric(yis1), modelResubPreds)
## Area under the curve: 0.9524
as.numeric(aucResult)
## 0.952381

wilcoxResults = wilcox.test(modelResubPreds[yis1], modelResubPreds[!yis1])
## W = 80, p-value = 0.0006192
wilcoxResults$statistic / (sum(yis1) * sum(!yis1))
## 0.952381 


## -----------------------------------------------------------------------------
ggd = data.frame(
    sample = names(sens),
    actual_class = as.numeric(yis1[names(sens)]),
    score = modelResubPreds[names(sens)],
    sensitivity = sens,
    specificity = spec
)
ggo = ggplot(ggd, aes(x=1-specificity, y=sensitivity))
ggo = ggo + geom_line(aes(color=score), size=1, alpha=0.75)
ggo = ggo + geom_text(aes(label=sample), ggd[ggd$actual_class == 1, ], color='red')
ggo = ggo + geom_text(aes(label=sample), ggd[ggd$actual_class == 0, ], angle=-90)
ggo = ggo + geom_hline(aes(yintercept=sensitivity),
                       data = ggd[ggd$actual_class == 1, ],
                       alpha=0.35, size=0.25)
ggo = ggo + geom_vline(aes(xintercept=1-specificity),
                       data = ggd[ggd$actual_class == 0, ],
                       alpha=0.35, size=0.25)
ggo = ggo + theme_classic()
ggo = ggo + scale_color_gradientn(colors=c('orangered', 'goldenrod', 'seagreen',
                                           'dodgerblue', '#606060'))
print(ggo)
 

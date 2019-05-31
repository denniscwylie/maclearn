library(genefilter)
library(ggplot2)

source('modelpipe.R')

load("prepared_datasets.RData")

x = xnorms$hess
y = structure(ys$hess, names=rownames(x))

fsKnnFitter = SolderedPipeFitter(
    FastTSelector(nFeat = 5),
    GlmFitter()
)

fitModel = fsKnnFitter(x, y)


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
ggd = data.frame(
    sample = names(sens),
    actual_class = as.numeric(yis1[names(sens)]),
    score = modelResubPreds[names(sens)],
    sensitivity = sens,
    specificity = spec
)
ggo = ggplot(ggd, aes(x=1-specificity, y=sensitivity))
ggo = ggo + geom_line()
ggo = ggo + geom_text(aes(label=sample), ggd[ggd$actual_class == 1, ], color='red')
ggo = ggo + geom_text(aes(label=sample), ggd[ggd$actual_class == 0, ], angle=-90)
ggo = ggo + geom_hline(aes(yintercept=sensitivity), ggd[ggd$actual_class == 1, ],
                       alpha=0.2)
ggo = ggo + geom_vline(aes(xintercept=1-specificity), ggd[ggd$actual_class == 0, ],
                       alpha=0.2)
ggo = ggo + theme_classic()
print(ggo)

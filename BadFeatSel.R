library(caret)
library(class)
library(ggplot2)

source("modelpipe.R")
source("SimData.R")


fsKnnFitter = SolderedPipeFitter(
    FastTSelector(nFeat = 10),
    KnnFitter(k = 3)
)

simData = simulate2Group(n=40, p=1000, effect=rep(0, 1000))
x = simData$x
y = simData$y

simSelBad = FastTSelector(nFeat=10)(x, y)
xbad = simSelBad$transform(x)
cvbad = train(KnnFitter(k=3), xbad, y,
        trControl=trainControl(method="cv", number=5))

cvgood = train(fsKnnFitter, x, y,
        trControl=trainControl(method="cv", number=5))

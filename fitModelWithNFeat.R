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
    return(fit)
}

xnames = names(xnorms)
names(xnames) = xnames

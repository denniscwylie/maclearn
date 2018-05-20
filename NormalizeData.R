## -----------------------------------------------------------------
## normalization
## -----------------------------------------------------------------
rleSizeFactors = function(x) {
    require(matrixStats)
    xno0 = x[ , colMins(x) > 0]
    geoMeans = exp(colMeans(log(xno0)))
    sizeFactors = rowMedians(sweep(xno0, 2, geoMeans, `/`))
    names(sizeFactors) = rownames(x)
    return(sizeFactors)
}

xnorms = list()

## shen set already normalizezd
xnorms$shen = xs$shen

## patel set already normalized
xnorms$patel = xs$patel

meanCenter = function(x, MARGIN=1) {
    geneHasNAs = apply(x, 3-MARGIN, function(z) {any(is.na(z))})
    means = apply(x, MARGIN, function(z) {mean(z[!geneHasNAs])})
    return(sweep(x, MARGIN, means, `-`))
}
meanCenterAndImpute = function(x, MARGIN=1,
            imputeAt=ceiling(max(x, na.rm=TRUE))) {
    geneHasNAs = apply(x, 3-MARGIN, function(z) {any(is.na(z))})
    means = apply(x, MARGIN, function(z) {mean(z[!geneHasNAs])})
    x[is.na(x)] = imputeAt
    return(sweep(x, MARGIN, means, `-`))
}
xnorms$montastier = meanCenterAndImpute(xs$montastier)

## hess set already normalized
xnorms$hess = xs$hess

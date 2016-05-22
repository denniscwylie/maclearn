## -----------------------------------------------------------------
## normalization
## -----------------------------------------------------------------
uqnormalize = function(x, MARGIN=1, scale=100) {
	## geneDetected = (apply(X=x, MARGIN=3-MARGIN, FUN=sum) > 0)
	geneDetected = if (MARGIN == 1) {
		colSums(x) > 0
	} else if (MARGIN == 2) {
		rowSums(x) > 0
	}
	return(scale * sweep(
		x = x,
		MARGIN = MARGIN,
		STATS = apply(X=x, MARGIN=MARGIN,
				FUN=function(z) {quantile(z[geneDetected], 0.75)}),
		FUN = `/`
	))
}

xnorms = list()
xnorms$bottomly = log2(uqnormalize(xs$bottomly) + 1)

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


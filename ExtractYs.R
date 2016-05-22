ys = list(
	bottomly = annots$bottomly$strain,
	patel = annots$patel$SubType,
	montastier = annots$montastier$Time,
	hess = annots$hess$pCRtxt
)
for (yname in names(ys)) {
	names(ys[[yname]]) = rownames(annots[[yname]])
}

## set order of levels to agree with Python factorize
ys = lapply(ys, function(y) {factor(as.character(y), levels=unique(y))})

ynums = lapply(ys, function(y) {
	ynames = names(y)
	y = as.numeric(y) - 1
	names(y) = ynames
	return(y)
})

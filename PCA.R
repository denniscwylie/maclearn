source("modelpipe.R")
source("MaclearnUtilities.R")

## source("LoadData.R")
## source("NormalizeData.R")
## source("RestrictData.R")
## source("ExtractYs.R")
load("prepared_datasets.RData")


pdf("BottomlyPCA.pdf", h=5, w=5*1.375)
ggpca(xnorms$bottomly, ys$bottomly, cshow=25,
        rlab=FALSE, clab=TRUE)
garbage = dev.off()

pdf("PatelPCA.pdf", h=5, w=5*1.375)
ggpca(xnorms$patel, ys$patel, cshow=25,
        rlab=FALSE, clab=TRUE)
garbage = dev.off()

set.seed(123456789)
xxx = kmeans(t(xnorms$patel), centers=3)
xc = factor(xxx$cluster)
names(xc) = colnames(xnorms$patel)
ggpca(data.frame(t(xnorms$patel)), y=xc, cshow=0, rsize=2, ralpha=0.35)

xc1 = xnorms$patel[ , xc == 1]
ggpca(xc1, ys$patel, cshow=25, clab=TRUE)

xc2 = xnorms$patel[ , xc == 2]
ggpca(xc2, ys$patel, cshow=25, clab=TRUE)

xc3 = xnorms$patel[ , xc == 3]
ggpca(xc3, ys$patel, cshow=25, clab=TRUE)


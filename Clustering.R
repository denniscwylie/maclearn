library(ggplot2)
library(pheatmap)

source("SimData.R")

set.seed(123)
simDat = simulate2Group(n=40, p=2, effect=c(1, 0.75))
xsim = simDat$x
ysim = simDat$y

## -----------------------------------------------------------------
## k-means
## -----------------------------------------------------------------
kmSim = kmeans(xsim, centers=2)

kmplot = function(xy) {
    x = xy$x
    y = xy$y
    km = kmeans(x, centers=2)
    ggdata = data.frame(x, cluster=factor(km$cluster), y=y)
    ggobj = ggplot(data=ggdata, mapping=aes(
        x = g1,
        y = g2,
        color = cluster,
        shape = y
    )) + theme_classic()
    ggobj = ggobj + geom_point(size=3)
    ggobj = ggobj + scale_shape_manual(values=c(6, 17))
    print(ggobj)
}

kmplot(simulate2Group(n=40, p=2, effect=c(10, 0)))

kmplot(simulate2Group(n=40, p=2, effect=c(1, 0.75)))


## -----------------------------------------------------------------
## hierarchical clustering
## -----------------------------------------------------------------
simData2 = simulate2Group(n=40, p=20, effect=c(2, 1, 1))
xsim2 = simData2$x
ysim2 = simData2$y

## cluster pseudosamples
ihcSim = hclust(xsim2)  ## generates error -- hclust wants distance matrix
                      ## not raw data!
xdist = dist(xsim2, method="euclidean")
ihcSim = hclust(xdist, method="average")
plot(ihcSim)

## cluster pseudogenes
ghcSim = hclust(dist(t(xsim2), method="euclidean"), method="average")
plot(ghcSim)


## -----------------------------------------------------------------
## clustered heatmap
## -----------------------------------------------------------------
heatY = data.frame(row.names=rownames(xsim2), group=ysim2)
pheatmap(t(xsim2), annotation=heatY,
        annotation_colors=list(group=c(A='black', B=rgb(1, 0, 0.4))))


## -----------------------------------------------------------------
## on real data...
## -----------------------------------------------------------------
load("prepared_datasets.RData")

hcBottomly = hclust(dist(xnorms$bottomly), method="complete")
## pdf("BottomlyHClust.pdf", h=5, w=6)
plot(hcBottomly)
## garbage = dev.off()

load("bottomlyGeneSyms.RData")
bottomlyHighVar = colnames(xnorms$bottomly)[
        apply(xnorms$bottomly, 2, sd) > 1.25]
heatX = t(xnorms$bottomly[ , bottomlyHighVar])
rownames(heatX) = ifelse(
    rownames(heatX) %in% names(bottomlyGeneSyms),
    bottomlyGeneSyms[rownames(heatX)],
    rownames(heatX)
)
## remove overall gene-means from data for more useful plot
heatX = data.frame(sweep(heatX, 1, rowMeans(heatX)), check.names=FALSE)
## pay attention to changes around mean, not far from it
maxLogFoldChange = 2.5
heatX[heatX > maxLogFoldChange] = maxLogFoldChange
heatX[heatX < -maxLogFoldChange] = -maxLogFoldChange
heatY = data.frame(row.names=colnames(heatX), strain=ys$bottomly)
## pdf("BottomlyHighVarHeatmap.pdf", h=8, w=8*1.3, onefile=FALSE)
pheatmap(
    heatX,
    annotation_col = heatY,
    annotation_colors = list(
    strain = c('C57BL/6J'='black', 'DBA/2J'=rgb(1, 0, 0.4)))
)
## garbage = dev.off()

library(ggplot2)
library(ggrepel)
library(matrixStats)
library(Rtsne)

load('prepared_datasets.RData')
load('shenGeneAnnot.RData')

theSeed = 123
set.seed(theSeed)
tsne = Rtsne(
    as.matrix(xnorms$shen),
    dims = 2,
    check.duplicates = FALSE,
    pca = TRUE,
    perplexity = 10,
    theta = 0.5
)

ggd = data.frame(
    sample = rownames(xnorms$shen),
    system = annots$shen[rownames(xnorms$shen), 'System'],
    coord1 = tsne$Y[ , 1],
    coord2 = tsne$Y[ , 2]
)
ggo = ggplot(ggd, aes(x=coord1, y=coord2, color=system, label=sample))
ggo = ggo + geom_point(alpha=0.75)
ggo = ggo + geom_text_repel(show.legend=FALSE)
ggo = ggo + scale_color_manual(
    values = c('firebrick3', 'goldenrod', 'lightseagreen',
               'darkorchid2', 'darkslategray', 'dodgerblue')
)
ggo = ggo + theme_bw() + theme(panel.grid.minor = element_blank(),
                               panel.grid.major = element_blank())
ggo = ggo + xlab('tSNE coordinate 1')
ggo = ggo + ylab('tSNE coordinate 2')
pdf(paste0('shen2012_tsne_', theSeed, '.pdf'), h=5.6, w=7.5)
print(ggo)
garbage = dev.off()

source("modelpipe.R")
source("MaclearnUtilities.R")

## source("LoadData.R")
## source("NormalizeData.R")
## source("RestrictData.R")
## source("ExtractYs.R")
load('prepared_datasets.RData')
load('shenGeneAnnot.RData')


## pdf("ShenPCA.pdf", h=5, w=5*1.375)
ggpca(xnorms$shen, annots$shen$System, cshow=25,
      rlab=FALSE, clab=TRUE, colscale=c(
          'Variable' = 'gray',
          'circulatory' = 'firebrick',
          'digestive/excretory' = 'goldenrod',
          'lymphatic' = 'lightseagreen',
          'nervous' = 'darkorchid',
          'other' = 'darkslategray',
          'respiratory' = 'dodgerblue'))
## garbage = dev.off()

## pdf("PatelPCA.pdf", h=5, w=5*1.375)
ggpca(xnorms$patel, ys$patel, cshow=25,
        rlab=FALSE, clab=TRUE)
## garbage = dev.off()

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




## -----------------------------------------------------------------
## pdf("ShenPCA_NoGenes.pdf", h=5, w=5*1.375)
ggpca(xnorms$shen, annots$shen$System, cshow=0,
      rlab=TRUE, clab=FALSE, colscale=c(
          'circulatory' = 'firebrick',
          'digestive/excretory' = 'goldenrod',
          'lymphatic' = 'lightseagreen',
          'nervous' = 'darkorchid',
          'other' = 'darkslategray',
          'respiratory' = 'dodgerblue'))
## garbage = dev.off()



## ## -----------------------------------------------------------------
## library(ggrepel)

## xsvd = svdForPca(xnorms$shen, center='col')
## dpc1 = xsvd$d[1]
## gpc1 = xsvd$v[ , 1]
## spc1 = xsvd$u[ , 1]

## ggd = data.frame(
##     u = spc1,
##     sample = names(spc1),
##     system = annots$shen[names(spc1), 'System'],
##     dummy = ''
## )
## ggo = ggplot(ggd, aes(x=dummy, y=u, color=system, label=sample))
## ggo = ggo + geom_point(alpha=0.5, shape='|', size=10)
## ggo = ggo + geom_hline(yintercept=0, linetype=2)
## ggo = ggo + geom_text_repel(alpha=0.75)
## ggo = ggo + scale_color_manual(
##     values = c('firebrick3', 'goldenrod', 'lightseagreen', 'darkorchid2',
##                'darkslategray', 'dodgerblue'),
##     guide = FALSE
## )
## ggo = ggo + ylim(-1.05*max(abs(ggd$u)), 1.05*max(abs(ggd$u)))
## ggo = ggo + coord_flip()
## ggo = ggo + xlab('')
## ggo = ggo + theme_classic()
## ggo = ggo + theme(axis.ticks.x=element_blank(),
## 	              axis.ticks.y=element_blank(),
##                   axis.line.y=element_blank())
## ## pdf('shen2012_u1.pdf', h=2, w=10)
## print(ggo)
## ## garbage = dev.off()

## ggd = data.frame(
##     v = gpc1,
##     gene = shenGeneAnnot[gsub(':.*', '', names(gpc1)), 'gene'],
##     dummy = ''
## )
## ggd[ggd$v > quantile(ggd$v, 0.0005) &
##     ggd$v < quantile(ggd$v, 0.9995), 'gene'] = ''
## ggd$shade = ifelse(is.na(ggd$gene) | ggd$gene == '', 'low', 'high')
## ggo = ggplot(ggd, aes(x=dummy, y=v, label=gene))
## ggo = ggo + geom_point(aes(alpha=shade, color=shade),
##                        shape='|', size=10)
## ggo = ggo + geom_hline(yintercept=0, linetype=2)
## ggo = ggo + geom_text_repel(alpha=1, color='red')
## ggo = ggo + scale_color_manual(values=c('black', '#444444'), guide=FALSE)
## ggo = ggo + scale_alpha_manual(values=c(0.15, 0.15), guide=FALSE)
## ggo = ggo + ylim(-1.05*max(abs(ggd$v)), 1.05*max(abs(ggd$v)))
## ggo = ggo + coord_flip()
## ggo = ggo + xlab('')
## ggo = ggo + theme_classic()
## ggo = ggo + theme(axis.ticks.x=element_blank(),
## 	              axis.ticks.y=element_blank(),
##                   axis.line.y=element_blank())
## ## png('shen2012_v1.png', h=(2/7)*1920, w=(10/7)*1920, res=288)
## print(ggo)
## ## garbage = dev.off()

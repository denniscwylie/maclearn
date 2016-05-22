library(genefilter)
library(GGally)
library(ggplot2)

source("MaclearnUtilities.R")

## source("LoadData.R")
## source("NormalizeData.R")
## source("RestrictData.R")
## source("ExtractYs.R")
load("prepared_datasets.RData")

ynums = lapply(ys, function(y) {
	ynames = names(y)
	y = as.numeric(y) - 1
	names(y) = ynames
	return(y)
})


## -----------------------------------------------------------------
## t.test example (using equal variance t test)
## -----------------------------------------------------------------
botgene = xnorms$bottomly[ , "ENSMUSG00000027855"]
botgene_C57BL = botgene[ys$bottomly == "C57BL/6J"]
botgene_DBA = botgene[ys$bottomly == "DBA/2J"]
t.test(botgene_C57BL, botgene_DBA, var.equal=TRUE)

cor(botgene, ynums$bottomly, method="pearson")


## -----------------------------------------------------------------
## t tests for all genes in bottomly set
## -----------------------------------------------------------------
tBotAll = colttests(as.matrix(xnorms$bottomly), ys$bottomly)
tBotAll$q.value = p.adjust(tBotAll$p.value, method="fdr")
## let's try something else...
xscBot = scale(xnorms$bottomly, center=TRUE, scale=TRUE)
summary(colMeans(xscBot))
summary(colSds(xscBot))
yscBot = scale(ynums$bottomly)
tBotAll$pearson = as.numeric( (t(yscBot) %*% xscBot) / (length(yscBot)-1) )
## sort by p.value
tBotAll = tBotAll[order(tBotAll$p.value), ]

plot(tBotAll$pearson, tBotAll$p.value, log='y', pch=16, cex=0.5)


## -----------------------------------------------------------------
## t tests for all genes in each set
## -----------------------------------------------------------------
tTestResults = mapply(
	FUN = function(x, y) {
		out = colttests(as.matrix(x), y)
		out$q.value = p.adjust(out$p.value, method="fdr")
		out$pearson = as.numeric(
			(t(scale(as.numeric(y))) %*% as.matrix(scale(x))) /
			(length(y)-1)
		)
		out = out[order(out$p.value), ]
		return(out)
	},
	xnorms,
	ys,
	SIMPLIFY = FALSE
)


## -----------------------------------------------------------------
## let's look at top genes in each set
## -----------------------------------------------------------------
lapply(tTestResults, head)


boxstrip(
	xnorms$bottomly[ rownames(tTestResults$bottomly)[1:9] ],
	ys$bottomly,
	colscale = c("black", "red")
)

ggpairsXY(
	xnorms$bottomly[ rownames(tTestResults$bottomly)[1:3] ],
	ys$bottomly,
	colscale = c("black", "red"),
	yname = "strain"
)


boxstrip(
	xnorms$patel[ rownames(tTestResults$patel)[1:9] ],
	ys$patel,
	colscale = c("black", "red")
)

ggpairsXY(
	xnorms$patel[ rownames(tTestResults$patel)[1:3] ],
	ys$patel,
	colscale = c("black", "red"),
	yname = "SubType"
)


boxstrip(
	xnorms$montastier[ rownames(tTestResults$montastier)[1:9] ],
	ys$montastier,
	colscale = c("black", "red")
)

ggpairsXY(
	xnorms$montastier[ rownames(tTestResults$montastier)[1:3] ],
	ys$montastier,
	colscale = c("black", "red"),
	yname = "Time"
)


boxstrip(
	xnorms$hess[ rownames(tTestResults$hess)[1:9] ],
	ys$hess,
	colscale = c("black", "red")
)

ggpairsXY(
	xnorms$hess[ rownames(tTestResults$hess)[1:3] ],
	ys$hess,
	colscale = c("black", "red"),
	yname = "pCR"
)


## -----------------------------------------------------------------
## generate fancy p.value vs pearson correlation plot-
## -----------------------------------------------------------------
ggdata = do.call(
	rbind,
	args = lapply(X=names(tTestResults), FUN=function(setname) {
		tres = tTestResults[[setname]]
		data.frame(
			gene = rownames(tres),
			set = paste0(setname, " (", nrow(xnorms[[setname]]), ")"),
			tres
		)
	})
)
ggdata$`|t|` = abs(ggdata$statistic)
ggdata$set = factor(as.character(ggdata$set),
		levels=levels(ggdata$set)[order(sapply(xnorms, nrow))])

ggobj = ggplot(data=ggdata,
		mapping=aes(x=pearson, y=`|t|`, color=set))
ggobj = ggobj + ylim(c(0, 10))
ggobj = ggobj + geom_line()
ggobj = ggobj + scale_color_manual(
		values=c("darkred", "red", "darkgray", "black"))
ggobj = ggobj + theme_bw()
pdf("TStatVsPearson.pdf", h=5, w=5*1.45)
print(ggobj)
garbage = dev.off()


## -----------------------------------------------------------------
## generate fancy p.value vs pearson correlation plot-
## -----------------------------------------------------------------
ggobj = ggplot(data=ggdata,
		mapping=aes(x=pearson, y=p.value, color=set))
ggobj = ggobj + scale_y_log10()
ggobj = ggobj + geom_line()
ggobj = ggobj + scale_color_manual(
		values=c("darkred", "red", "darkgray", "black"))
ggobj = ggobj + theme_bw()
## pdf("PValVsPearson.pdf", h=5, w=5*1.45)
print(ggobj)
## garbage = dev.off()



## -----------------------------------------------------------------
## complementary features
## -----------------------------------------------------------------
compResults = gramSchmidtSelect(x=xnorms$patel, y=ys$patel,
		g="NAMPT")
compFeats = names(compResults[
		order(abs(compResults), decreasing=TRUE)])[1:1000]
compR2 = sapply(compFeats, function(g) {
	summary(lm(ynums$patel ~ xnorms$patel$NAMPT +
			xnorms$patel[[g]]))$r.squared
})
plot(compR2, type="l")

## model being optimized is truly linear model -- not
## a good choice for classification!
summary(lm(ynums$patel ~ xnorms$patel$NAMPT + xnorms$patel$SEC61G))
## but features likely to work well for glm classification
## (and other classification methods) as well
summary(glm(ys$patel ~ xnorms$patel$NAMPT + xnorms$patel$SEC61G, family=binomial))

#!/usr/bin/env Rscript

library(ggplot2)

source("MaclearnUtilities.R")
source("SimData.R")

set.seed(123)
d2 = simulate2Group(n=40, p=2, effect=c(2, -2))

ggdata = data.frame(d2$x, y=d2$y)
ggobj = ggplot(
	ggdata,
	aes(x=g1, y=g2, color=y)
)
ggobj = ggobj + geom_point(size=3, alpha=0.65)
## now for some formatting options:
ggobj = ggobj + scale_color_manual(values=c("black", "red"))
ggobj = ggobj + theme_classic()
## generate the plot:
print(ggobj)

rotateData = function(x, angle) {
	rotationMatrix = rbind(
		c(cos(angle), -sin(angle)),
		c(sin(angle), cos(angle))
	)
	return(rotationMatrix %*% x)
}

rotateTransposedData = function(x, angle) {
	return(t(rotateData(t(x), angle)))
}

anglesToView = seq(0, pi, by=pi/4)
names(anglesToView) = paste(anglesToView*180/pi, "deg")
rotatedD2 = lapply(
	X = anglesToView,
	FUN = function(angle) {rotateTransposedData(d2$x, angle)}
)
## rotatedD2 is list containing one data.frame for each angle

lapply(rotatedD2, head)

## add angle and group information into rotatedD2 for plotting
rotatedD2Annotated = lapply(
	X = names(rotatedD2),
	FUN = function(anglename) {
		df = data.frame(
			angle = anglename,
			x = rotatedD2[[anglename]][ , 1],
			y = rotatedD2[[anglename]][ , 2],
			group = d2$y
		)
		df$mu_x = mean(df$x)
		df$sigma_x = sd(df$x)
		return(df)
    }
)
## combine all rotated matrices into one data.frame for ggplot
ggRotData = do.call(rbind, rotatedD2Annotated)

ggRotObj = ggplot(ggRotData, aes(x=x, y=y, color=group))
ggRotObj = ggRotObj + geom_point(show.legend=FALSE, alpha=0.7)
ggRotObj = ggRotObj + geom_segment(
	mapping = aes(
		x = mu_x - 2*sigma_x,
		xend = mu_x + 2*sigma_x,
		y = 0,
		yend = 0,
		size = sigma_x
	),
	alpha = 0.0075,
	color = "black",
	show.legend = FALSE
)
## put each angle in separate facet
ggRotObj = ggRotObj + facet_wrap(~ angle, nrow=1)
## formatting
ggRotObj = ggRotObj + scale_color_manual(values=c("black", "red"))
ggRotObj = ggRotObj + scale_size_continuous(range=2*range(ggRotData$sigma_x)^2)
ggRotObj = ggRotObj + theme_bw()
## and print
## pdf("pca_by_rotation.pdf", h=2, w=8.5)
print(ggRotObj)
## garbage = dev.off()

## 45-degree rotation matrix:
rbind(
	c(cos(pi/4), -sin(pi/4)),
	c(sin(pi/4), cos(pi/4))
)
##           [,1]       [,2]
## [1,] 0.7071068 -0.7071068
## [2,] 0.7071068  0.7071068
## -- first row is proportional to (+1, -1)
## -- second row to (+1, +1)

ggPcaObj = ggpca(data.frame(d2$x), d2$y,
		colscale=c("dodgerblue3", "black", "red"),
		lsize=4, center="none", clab=TRUE, print=FALSE)
ggPcaObj = ggPcaObj + geom_vline(xintercept=0) + geom_hline(yintercept=0)
## pdf("simpca2d.pdf", h=3.5, w=5.25)
print(ggPcaObj)
## garbage = dev.off()
## PC1 values for (g1, g2) approximately proportional to (+1, -1)
## PC2 values for (g1, g2) approximately proportional to (+1, +1)

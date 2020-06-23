ggfuntile = function(f, d,
                     xrange=c(0, 1), yrange=c(0, 1), limits=c(0, 1),
                     density=101,
                     xlab="x", ylab="y", zlab="f",
                     breaks=NULL, ...) {
    require(ggplot2)
    ggdata = expand.grid(list(
        x = seq(xrange[1], xrange[2],
                by=(xrange[2]-xrange[1])/(density-1)),
        y = seq(yrange[1], yrange[2],
                by=(yrange[2]-yrange[1])/(density-1))
    ))
    ggdata$z = mapply(FUN=f, ggdata$x, ggdata$y)
    gg = ggplot(data=ggdata, aes(x=x, y=y), inherit.aes=FALSE)
    gg = gg + geom_tile(aes(fill=z))
    if (length(breaks) == 0) {
        gg = gg + stat_contour(aes(z=z), bins=2, color="white")
    } else {
        gg = gg + stat_contour(aes(z=z), breaks=breaks, color="white")
    }
    gg = gg + scale_fill_gradientn(
        colors = c("black", "#404040", "#808080",
                   "dodgerblue1", "blue", "darkblue"),
        name = zlab,
        limits = limits
    )
    gg = gg + theme_classic()
    gg = gg + xlab(xlab) + ylab(ylab)
    gg = gg + geom_point(
         data = d,
         mapping = aes_string(x=xlab, y=ylab, shape="class"),
         color="greenyellow", size=2, alpha=0.8
    )
    gg = gg + scale_shape_manual(values=c(6, 17))
    return(gg)
}


predictionContour = function(fit, data, y, title, density=51) {
    data = data.frame(data, check.names=FALSE)
    predictor = function(g, h) {
        dfgh = data.frame(x=g, y=h)
        colnames(dfgh) = colnames(data)[1:2]
        return(predict(fit, dfgh, type="prob")[ , 2])
    }
    data$class = y
    xrange = 0.5 * c(floor(2*min(data[ , 1])),
                     ceiling(2*max(data[ , 1])))
    yrange = 0.5 * c(floor(2*min(data[ , 2])),
                     ceiling(2*max(data[ , 2])))
    ggfuntile(predictor, data,
              xrange=xrange, yrange=yrange, density=density,
              xlab=colnames(data)[[1]], ylab=colnames(data)[[2]],
              zlab="P(Y=1)", breaks=c(-Inf, 0.5, Inf)) +
        ggtitle(title) + theme(plot.title=element_text(hjust=0.5))
}


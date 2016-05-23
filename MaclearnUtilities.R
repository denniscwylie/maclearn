rowSds = function(x, na.rm=FALSE) {
    n = ncol(x)
    return(sqrt((n/(n-1)) *
            (rowMeans(x*x, na.rm=na.rm) - rowMeans(x, na.rm=na.rm)^2)))
}
colSds = function(x, na.rm=FALSE) {
    n = nrow(x)
    return(sqrt((n/(n-1)) *
            (colMeans(x*x, na.rm=na.rm) - colMeans(x, na.rm=na.rm)^2)))
}



svdForPca = function(
        x,
        center=c("both", "row", "col", "none"),
        scale=c("none", "row", "col")) {
    center = match.arg(center)
    scale = match.arg(scale)
    if (center %in% c("row", "both")) {
        x = sweep(x, 1, STATS=rowMeans(x))
    }
    if (center %in% c("column", "both")) {
        x = sweep(x, 2, STATS=colMeans(x))
    }
    if (scale == "row") {
        x = sweep(x, 1, STATS=rowSds(x), FUN=`/`)
    } else if (scale == "column") {
        x = sweep(x, 2, STATS=colSds(x), FUN=`/`)
    }
    out = svd(x)
    dord = order(out$d, decreasing=TRUE)
    out$u = out$u[ , dord]
    rownames(out$u) = rownames(x)
    out$d = out$d[dord]
    out$v = out$v[ , dord]
    rownames(out$v) = colnames(x)
    return(out)
}



ggpca = function(
        x,
        y,
        center = c("both", "column", "row", "none"),
        scale = c("none", "column", "row"),
        rlab = FALSE,
        clab = FALSE,
        cshow = ncol(x),
        rsize = 4,
        csize = 2,
        lsize = 3,
        ralpha = 0.6,
        calpha = 1.0,
        rname = "Sample",
        cname = "Variable",
        lname = "",
        grid = FALSE,
        print = TRUE,
        colscale,
        ...) {
    require(ggplot2)
    center = match.arg(center)
    scale = match.arg(scale)
    if (length(rlab)==1 && is.logical(rlab)) {
        rlab = if (rlab) {rownames(x)} else {""}
    }
    if (length(clab)==1 && is.logical(clab)) {
        clab = if (clab) {colnames(x)} else {""}
    }
    if (!missing(y)) {
        if (is.character(y)) {
            y = factor(y, levels=unique(y))
        }
        if (length(names(y)) == 0) {
            names(y) = rownames(x)
        }
        classLevels = c(cname, levels(y))
        y = structure(as.character(y), names=names(y))
    } else {
        classLevels = c(cname, rname)
    }
    x = x[ , sapply(x, function(z) {!any(is.na(z))}), drop=FALSE]
    xsvd = svdForPca(x, center=center, scale=scale)
    rsf = max(xsvd$u[ , 1]) - min(xsvd$u[ , 1])
    csf = max(xsvd$v[ , 1]) - min(xsvd$v[ , 1])
    sizeRange = sort(c(csize, rsize))
    alphaRange = sort(c(calpha, ralpha))
    ggdata = data.frame(
        PC1 = xsvd$u[ , 1] / rsf,
        PC2 = xsvd$u[ , 2] / rsf,
        label = rlab,
        size = rsize,
        alpha = ralpha,
        stringsAsFactors = FALSE
    )
    if (cshow > 0) {
        cdata = data.frame(
            PC1 = xsvd$v[ , 1] / csf,
            PC2 = xsvd$v[ , 2] / csf,
            label = clab,
            size = csize,
            alpha = calpha,
            stringsAsFactors = FALSE
        )
        if (cshow < ncol(x)) {
            cscores = cdata$PC1^2 + cdata$PC2^2
            names(cscores) = colnames(x)
            keep = names(sort(cscores, decreasing=TRUE)[1:cshow])
            cdata[!colnames(x) %in% keep, "label"] = ""
            cdata[!colnames(x) %in% keep, "alpha"] = 0.1
            alphaRange = c(min(alphaRange[1], 0.1),
                    max(alphaRange[2], 0.1))
        }
        ggdata = rbind(cdata, ggdata)
    }
    cclass = rep(cname, times=if (cshow>0) {nrow(cdata)} else{0})
    if (!missing(y)) {
        ggdata$class = factor(c(cclass, y), levels=classLevels)
    } else {
        ggdata$class = factor(
                c(cclass, rep(rname, times=nrow(x))), levels=classLevels)
    }
    ggobj = ggplot(
        aes(
            x = PC1,
            y = PC2,
            color = class,
            size = size,
            alpha = alpha,
            label = label
        ), 
        data = ggdata
    ) + geom_point() + theme_bw()
    ggobj = ggobj + geom_text(vjust=-1.1, show.legend=FALSE, size=lsize)
    if (missing(colscale) && (length(unique(ggdata$class)) < 8)) {
        colscale = c("gray", "darkslategray", "goldenrod", "lightseagreen",
                "orangered", "dodgerblue2", "darkorchid4")[
                1:length(unique(ggdata$class))]
        if (length(colscale) == 2 && cshow > 0) {colscale = c("darkgray", "black")}
        if (length(colscale) == 2 && cshow == 0) {colscale = c("black", "red")}
        if (length(colscale) == 3) {colscale = c("darkgray", "black", "red")}
    }
    if (all(classLevels %in% names(colscale))) {
        colscale = colscale[classLevels]
    }
    ggobj = ggobj + scale_color_manual(values=colscale, name=lname)
    ggobj = ggobj + scale_size_continuous(guide=FALSE, range=sizeRange)
    ggobj = ggobj + scale_alpha_continuous(guide=FALSE, range=alphaRange)
    ggobj = ggobj + xlab(paste0(
        "PC1 (",
        round(100 * xsvd$d[1]^2 / sum(xsvd$d^2), 1),
        "% explained var.)"
    ))
    ggobj = ggobj + ylab(paste0(
        "PC 2 (",
        round(100 * xsvd$d[2]^2 / sum(xsvd$d^2), 1),
        "% explained var.)"
    ))
    if (!grid) {
        ggobj = ggobj + theme(
            panel.grid.minor = element_blank(),
            panel.grid.major = element_blank(),
            panel.background = element_blank()
        )
    }
    if (print) {print(ggobj)}
    invisible(ggobj)
}



boxstrip = function(x, y, colscale,
        xname="group", nrow, pointAlpha=0.6, boxAlpha=0.5, scales="fixed",
        print=TRUE) {
    require(ggplot2)
    require(reshape2)
    if (length(names(y)) == 0) {
        names(y) = rownames(x)
    }
    xmelt = melt(data.frame(row=rownames(x), x, check.names=FALSE))
    xmelt$row = factor(as.character(xmelt$row),
            levels=unique(xmelt$row))
    xmelt$group = y[as.character(xmelt$row)]
    if (!is.factor(y)) {
        xmelt$group = factor(xmelt$group, levels=unique(y))
    }
    aesArg = if (missing(colscale)) {
        aes(x=group, y=value)
    } else {
        aes(x=group, y=value, color=group)
    }
    ggobj = ggplot(
        aesArg,
        data = xmelt
    ) + geom_point(alpha=pointAlpha) + theme_bw()
    if (!missing(colscale)) {
        if (all(levels(xmelt$group) %in% names(colscale))) {
            colscale = colscale[xmelt$group]
        }
        ggobj = ggobj + scale_color_manual(name=xname, values=colscale)
    }
    facetArgs = list(facets=~variable, scales=scales)
    if (!missing(nrow)) {
        facetArgs$nrow = nrow
    }
    ggobj = ggobj + do.call(facet_wrap, args=facetArgs)
    ggobj = ggobj + geom_boxplot(outlier.size=0, alpha=boxAlpha)
    ggobj = ggobj + theme(
            axis.text.x=element_text(angle=-90, vjust=0.5, hjust=0))
    ggobj = ggobj + xlab("")
    if (print) {
        print(ggobj)
    }
    invisible(ggobj)
}



altprint.ggpairs = function(x, leftWidthProportion = 0.2, bottomHeightProportion = 0.1, 
    spacingProportion = 0.03, showStrips = NULL, ...) {
    require(ggplot2)
    require(GGally)
    require(grid)
    require(gtable)
    plotObj <- x
    if (identical(plotObj$axisLabels, "internal")) {
        v1 <- viewport(y = unit(0.5, "npc") - unit(0.5, "lines"), 
            width = unit(1, "npc") - unit(1, "lines"), height = unit(1, 
                "npc") - unit(2, "lines"))
    }
    else {
        v1 <- viewport(width = unit(1, "npc") - unit(3, "lines"), 
            height = unit(1, "npc") - unit(3, "lines"))
    }
    numCol <- length(plotObj$columns)
    if (identical(plotObj$axisLabels, "show")) {
        showLabels <- TRUE
        viewPortWidths <- c(leftWidthProportion, 1, rep(c(spacingProportion, 
            1), numCol - 1))
        viewPortHeights <- c(rep(c(1, spacingProportion), numCol - 
            1), 1, bottomHeightProportion)
    }
    else {
        showLabels <- FALSE
        viewPortWidths <- c(1, rep(c(spacingProportion, 1), numCol - 
            1))
        viewPortHeights <- c(rep(c(1, spacingProportion), numCol - 
            1), 1)
    }
    viewPortCount <- length(viewPortWidths)
    v2 <- viewport(layout = grid.layout(viewPortCount, viewPortCount, 
        widths = viewPortWidths, heights = viewPortHeights))
    grid.newpage()
    if (plotObj$title != "") {
        pushViewport(viewport(height = unit(1, "npc") - unit(0.4, 
            "lines")))
        grid.text(plotObj$title, x = 0.5, y = 1, just = c(0.5, 
            1), gp = gpar(fontsize = 15))
        popViewport()
    }
    if (!identical(plotObj$axisLabels, "internal")) {
        pushViewport(viewport(width = unit(1, "npc") - unit(2, 
            "lines"), height = unit(1, "npc") - unit(3, "lines")))
        pushViewport(viewport(layout = grid.layout(viewPortCount, 
            viewPortCount, widths = viewPortWidths, heights = viewPortHeights)))
        for (i in 1:numCol) {
            grid.text(plotObj$columnLabels[i], 0, 0.5, rot = 90, 
                just = c("centre", "centre"), vp = GGally:::vplayout(as.numeric(i) * 
                  2 - 1, 1), vjust=-0.25) ## 150227 DW: added vjust
        }
        popViewport()
        popViewport()
        pushViewport(viewport(width = unit(1, "npc") - unit(3, 
            "lines"), height = unit(1, "npc") - unit(2, "lines")))
        pushViewport(viewport(layout = grid.layout(viewPortCount, 
            viewPortCount, widths = viewPortWidths, heights = viewPortHeights)))
        for (i in 1:numCol) {
            grid.text(plotObj$columnLabels[i], 0.5, 0, just = c("centre", 
                "centre"), vp = GGally:::vplayout(ifelse(showLabels, 2 * 
                numCol, 2 * numCol - 1), ifelse(showLabels, 2 * 
                i, 2 * i - 1)), vjust=1.25) ## 150227 DW: added vjust
        }
        popViewport()
        popViewport()
    }
    pushViewport(v1)
    pushViewport(v2)
    for (rowPos in 1:numCol) {
        for (columnPos in 1:numCol) {
            p <- getPlot(plotObj, rowPos, columnPos)
            if (GGally:::is_blank_plot(p)) {
                next
            }
            pGtable <- ggplot_gtable(ggplot_build(p))
            if (columnPos == 1 && showLabels) {
                if (identical(plotObj$verbose, TRUE)) {
                  print("trying left axis")
                }
                pAxisLabels <- gtable_filter(pGtable, "axis-l")
                grobLength <- length(pAxisLabels$grobs)
                leftAxisLayoutHeight <- rep(c(0.1, 1), grobLength)[-1]
                leftAxisLayoutHeightUnits <- rep(c("lines", "null"), 
                  grobLength)[-1]
                vpLAxis <- viewport(layout = grid.layout(nrow = 2 * 
                  grobLength - 1, ncol = 1, widths = unit(1, 
                  "null"), heights = unit(leftAxisLayoutHeight, 
                  leftAxisLayoutHeightUnits)))
                pushViewport(GGally:::vplayout(rowPos * 2 - 1, 1))
                pushViewport(vpLAxis)
                for (lAxisPos in 1:grobLength) {
                  pushViewport(GGally:::vplayout(lAxisPos * 2 - 1, 1))
                  grid.draw(pAxisLabels$grobs[[lAxisPos]])
                  popViewport()
                }
                popViewport()
                popViewport()
            }
            if (rowPos == numCol && showLabels) {
                if (identical(plotObj$verbose, TRUE)) {
                  print("trying bottom axis")
                }
                pAxisLabels <- gtable_filter(pGtable, "axis-b")
                grobLength <- length(pAxisLabels$grobs)
                botAxisLayoutWidth <- rep(c(0.1, 1), grobLength)[-1]
                botAxisLayoutWidthUnits <- rep(c("lines", "null"), 
                  grobLength)[-1]
                vpBAxis <- viewport(layout = grid.layout(nrow = 1, 
                  ncol = 2 * grobLength - 1, heights = unit(1, 
                    "null"), widths = unit(botAxisLayoutWidth, 
                    botAxisLayoutWidthUnits)))
                pushViewport(GGally:::vplayout(2 * numCol, 2 * columnPos))
                pushViewport(vpBAxis)
                for (bAxisPos in 1:grobLength) {
                  pushViewport(GGally:::vplayout(1, bAxisPos * 2 - 1))
                  grid.draw(pAxisLabels$grobs[[bAxisPos]])
                  popViewport()
                }
                popViewport()
                popViewport()
            }
            layoutNames <- c("panel")
            allLayoutNames <- c("panel", "strip-right", "strip-top")
            if (is.null(showStrips)) {
                pShowStrips <- (!is.null(p$type)) && (!is.null(p$subType))
                if (pShowStrips) {
                  if (columnPos == numCol) {
                    layoutNames <- c(layoutNames, "strip-right")
                  }
                  if (rowPos == 1) {
                    layoutNames <- c(layoutNames, "strip-top")
                  }
                }
            }
            else if (showStrips) {
                layoutNames <- allLayoutNames
            }
            if (!is.null(p$axisLabels)) {
                if (p$axisLabels %in% c("internal", "none")) {
                  layoutNames <- allLayoutNames
                }
            }
            layoutRows <- pGtable$layout$name %in% layoutNames
            layoutInfo <- pGtable$layout[layoutRows, ]
            layoutTB <- layoutInfo[, c("t", "b")]
            layoutLR <- layoutInfo[, c("l", "r")]
            pPanel <- pGtable[min(layoutTB):max(layoutTB), min(layoutLR):max(layoutLR)]
            pushViewport(GGally:::vplayout(2 * rowPos - 1, ifelse(showLabels, 
                2 * columnPos, 2 * columnPos - 1)))
            suppressMessages(suppressWarnings(grid.draw(pPanel)))
            popViewport()
        }
    }
    popViewport()
    popViewport()
}

ggpairs0 = function(data, ...,
        ## lower=list(continuous="smooth"),
        lower=list(continuous="points"),
        upper=list(continuous="density"),
        colscale, alpha=0.6, print=TRUE) {
    ggobj = ggpairs(data, lower=lower, upper=upper, alpha=I(alpha), ...)
    ggobj = ggobj + theme_bw()
    ggobj = ggobj + theme(
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        panel.background = element_blank()
    )
    ggobj$plots = lapply(X=ggobj$plots, FUN=function(code) {
        if (length(grep("ggally_points", code)) > 0) {
            code = paste0(code, "+stat_smooth(deg=1, span=1.5)")
        }
        return(code)
    })
    if (!missing(colscale)) {
        colorCode = paste(colscale, collapse="', '")
        ggobj$plots = lapply(X=ggobj$plots, FUN=function(code) {
            code = paste0(code, "+scale_color_manual(values=c('",
                    colorCode, "'))")
            code = paste0(code, "+scale_fill_manual(values=c('",
                    colorCode, "'))")
            return(code)
        })
    }
    if (print) {
        altprint.ggpairs(ggobj)
    }
    invisible(ggobj)
}



gramSchmidtSelect = function(x, y, g=NULL) {
    dx = as.matrix(sweep(
        x = x,
        MARGIN = 2,
        STATS = colMeans(x),
        FUN = `-`
    ))
    y = as.numeric(y)
    dy = y - mean(y)
    pgtotal = diag(1, nrow(x))
    for (gel in g) {
        dxg = as.numeric(scale(as.numeric(pgtotal %*% dx[ , gel])))
        pg = diag(1, nrow(x)) - (outer(dxg, dxg) / sum(dxg^2))
        pgtotal = pg %*% pgtotal
    }
    pgdx = pgtotal %*% dx
    pgdy = pgtotal %*% matrix(dy, nrow=length(dy))
    compCors = as.numeric(t(scale(pgdy)) %*% scale(pgdx)) / (length(y)-1)
    names(compCors) = colnames(x)
    return(compCors)
}

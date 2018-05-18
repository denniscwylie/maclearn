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
        center=c("col", "row", "both", "none"),
        scale=c("none", "row", "col")) {
    center = match.arg(center)
    scale = match.arg(scale)
    if (center %in% c("row", "both")) {
        x = sweep(x, 1, STATS=rowMeans(x))
    }
    if (center %in% c("col", "both")) {
        x = sweep(x, 2, STATS=colMeans(x))
    }
    if (scale == "row") {
        x = sweep(x, 1, STATS=rowSds(x), FUN=`/`)
    } else if (scale == "col") {
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
        center = c("col", "row", "both", "none"),
        scale = c("none", "col", "row"),
        rlab = FALSE,
        clab = FALSE,
        labrepel = FALSE,
        cshow = ncol(x),
        rsize = 4,
        csize = 2,
        lsize = 3,
        ralpha = 0.6,
        calpha = 1.0,
        clightalpha = 0.1,
        rname = "Sample",
        cname = "Variable",
        lname = "",
        grid = FALSE,
        print = TRUE,
        xsvd = NULL,
        invert1 = FALSE,
        invert2 = FALSE,
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
    if (length(xsvd) == 0) {
        xsvd = svdForPca(x, center=center, scale=scale)
    }
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
            cdata[!colnames(x) %in% keep, "alpha"] = clightalpha
            alphaRange = c(min(alphaRange[1], clightalpha),
                    max(alphaRange[2], clightalpha))
        }
        ggdata = rbind(cdata, ggdata)
    }
    if (invert1) {ggdata$PC1 = -ggdata$PC1}
    if (invert2) {ggdata$PC2 = -ggdata$PC2}
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
    )
    ggobj = ggobj + geom_hline(yintercept=0, color='lightgray', alpha=0.5)
    ggobj = ggobj + geom_vline(xintercept=0, color='lightgray', alpha=0.5)
    ggobj = ggobj + geom_point() + theme_bw()
    if (!labrepel) {
        ggobj = ggobj + geom_text(vjust=-1.1, show.legend=FALSE, size=lsize)
    } else {
        require(ggrepel)
        ## ggdata[ggdata$label == '', 'label'] = NA
        ggobj = ggobj + geom_text_repel(show.legend=FALSE, size=lsize)
    }
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
    ggobj = ggobj + theme(axis.ticks.x = element_blank(),
                          axis.ticks.y = element_blank(),
                          axis.text.x = element_blank(),
                          axis.text.y = element_blank())
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

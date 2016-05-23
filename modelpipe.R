rowSds = function(x, na.rm=FALSE) {
    n = ncol(x)
    return(sqrt((n/(n-1)) * (
            rowMeans(x*x, na.rm=na.rm) - rowMeans(x, na.rm=na.rm))))
}
colSds = function(x, na.rm=FALSE) {
    n = nrow(x)
    return(sqrt((n/(n-1)) * (
            colMeans(x*x, na.rm=na.rm) - colMeans(x, na.rm=na.rm))))
}


XTransformer = function(f) {
    f = f
    learner = function(x, y, ...) {list(transform=f)}
    class(learner) = c("XTransformer", "ModelPipe", class(learner))
    return(learner)
}

PValueSelector = function(f=NULL, threshold=0.05, fdr="fdr", 
        nFeat=NULL, vectorize=TRUE) {
    f = f
    threshold = threshold
    fdr = fdr
    nFeat = nFeat
    vectorize = vectorize
    if (length(f) == 0) {
        vectorize = FALSE
        f = function(x, y, ...) {
            yFeat = y
            if (is.factor(yFeat)) {
                yFeat = as.integer(yFeat)
            }
            apply(X=x, MARGIN=2, FUN=function(z) {
                t.test(z[yFeat==1], z[yFeat==2], ...)[["p.value"]]
            })
        }
    }
    learner = function(x, y, ...) {
        featurePVals = if (vectorize) {
            apply(X=x, MARGIN=2, FUN=function(z) {
                f(z, y, ...)
            })
        } else {
            f(x, y, ...)
        }
        selectedFeatures = if (length(nFeat) > 0) {
            names(sort(featurePVals))[1:nFeat]
        } else {
            if (length(fdr) > 0) {
                featurePVals = p.adjust(featurePVals, method=fdr)
            }
            names(featurePVals[featurePVals < threshold])
        }
        return(list(
            selectedFeatures = selectedFeatures,
            p = featurePVals,
            transform = function(x, ...) {
                x[ , selectedFeatures, drop=FALSE]
            }
        ))
    }
    class(learner) = c("PValueSelector", "ModelPipe", class(learner))
    return(learner)
}

Multiselector = function(...) {
    subLearners = list(...)
    learner = function(x, y, ...) {
        subLearned = lapply(X=subLearners,
                FUN=function(f) {f(x, y, ...)})
        selectedFeatures = Reduce(f=union, x=lapply(
                X=subLearned, FUN=function(z) {z$selectedFeatures}))
        return(list(
            selectedFeatures = selectedFeatures,
            transform = function(x, ...) {
                x[ , selectedFeatures, drop=FALSE]
            },
            components = subLearned
        ))
    }
}

ModelFitter = function(f,
        predictor=predict, predictionProcessor=identity) {
    predictor = predictor
    predictionProcessor = predictionProcessor
    learner = function(x, y, ...) {
        fit = f(x, y, ...)
        fitOut = list(
            fit = fit,
            predict = function(x, ...) {
                predictionProcessor(predictor(fit, x, ...))
            }
        )
        class(fitOut) = "ModelFit"
        return(fitOut)
    }
    class(learner) = c("ModelFitter", "ModelPipe", class(learner))
    return(learner)
}

SolderedPipeFitter = function(...) {
    piping = list(...)
    learner = function(x, y, ...) {
        fit = list()
        for (i in 1:length(piping)) {
            fit[[i]] = (piping[[i]])(x, y, ...)
            if (is(fit[[i]], "SolderedPipe")) {
                x = transform(fit[[i]], x, ...)
            } else if ("transform" %in% names(fit[[i]])) {
                x = fit[[i]]$transform(x, y, ...)
            }
        }
        class(fit) = c("SolderedPipe", "ModelPipe")
        return(fit)
    }
    class(learner) = c("SolderedPipeFitter", "ModelFitter",
            "SolderedPipe", "ModelPipe", class(learner))
    return(learner)
}
SolderedPipe = SolderedPipeFitter
solder = SolderedPipeFitter

transform.ModelPipe = function(obj, x, ...) {
    if (!is(obj, "SolderedPipe")) {
        obj = solder(obj)
    }
    for (subobj in obj) {
        if (is(subobj, "SolderedPipe")) {
            x = transform(subobj, x, ...)
        } else if ("transform" %in% names(subobj)) {
            x = subobj$transform(x, ...)
        }
    }
    return(x)
}

predict.ModelPipe = function(obj, x, ..., level=0) {
    if (level == 0) {
        x = transform(obj, x, ...)
    }
    if (!is(obj, "SolderedPipe")) {
        obj = solder(obj)
    }
    for (i in length(obj):1) {
        if (is(obj[[i]], "SolderedPipe")) {
            out = predict(obj[[i]], x, ..., level=level+1)
            if (!all(is.na(out))) {
                return(out)
            }
        } else if ("predict" %in% names(obj[[i]])) {
            return(obj[[i]]$predict(x, ...))
        }
    }
    return(NA)
    ## stop("This ModelPipe object does not support predict.")
}

predict.ModelFit = function(obj, x, ...) {obj$predict(x, ...)}


caretize = function(fitpipe, lev=NULL, threshold=0.5, ...,
        type=NULL, library=NULL, loop=NULL,
        parameters=NULL, grid=NULL) {
    lev = lev
    threshold = threshold
    if (length(type) == 0) {
        type = ifelse(length(lev)==0, "Regression", "Classification")
    }
    if (type == "Classification") {
        caretPredict = function(modelFit, newdata, ...) {
            ifelse(
                predict(modelFit, newdata) < threshold,
                lev[1],
                lev[2]
            )
        }
    } else {
        caretPredict = function(modelFit, newdata, ...) {
            predict(modelFit, newdata)
        }
    }
    if (length(parameters) == 0) {
        parameters = data.frame(
            parameter = "ignored",
            class = "numeric",
            label = "Ignored"
        )
    }
    if (length(grid) == 0) {
        grid = function(x, y, len=NULL, ...) {data.frame(ignored=0)}
    }
    return(list(
        library = library,
        type = type,
        loop = loop,
        parameters = parameters,
        grid = grid,
        fit = fitpipe,
        predict = caretPredict,
        prob = function(modelFit, newdata, ...) {
            preds = predict(modelFit, newdata)
            out = data.frame(
                lev1 = 1 - preds,
                lev2 = preds
            )
            colnames(out) = lev
            return(out)
        }
    ))
}

train.ModelFitter = function(
        fitpipe,
        x,
        y,
        threshold = 0.5,
        ...,
        method = "repeatedcv",
        number = 10,
        repeats = 1,
        trControl = trainControl(method, number, repeats),
        tuneGrid = NULL,
        parameters = NULL,
        grid = NULL,
        type = NULL,
        library = NULL,
        loop = NULL) {
    caretizedPipe = caretize(
        fitpipe,
        lev = levels(y),
        threshold = threshold,
        type = type,
        library = library,
        loop = loop,
        parameters = parameters,
        grid = grid
    )
## browser()
    return(train(
        x = x,
        y = y,
        method = caretizedPipe,
        trControl = trControl,
        tuneGrid = tuneGrid,
        ...
    ))
}

## caretTrain = function(
##         fitpipe,
##         x,
##         y,
##         threshold = 0.5,
##         ...,
##         method = "repeatedcv",
##         number = 10,
##         repeats = 1,
##         trControl = trainControl(method, number, repeats),
##         tuneGrid = data.frame(ignored=0)) {
##     caretizedPipe = caretize(fitpipe, lev=levels(y), threshold=threshold)
##     return(train(
##         x = x,
##         y = y,
##         method = caretizedPipe,
##         trControl = trControl,
##         tuneGrid = tuneGrid,
##         ...
##     ))
## }



FastTSelector = function(threshold=0.05, fdr="fdr", nFeat=NULL) {
    threshold = threshold
    fdr = fdr
    nFeat = nFeat
    require(genefilter)
    selector = PValueSelector(
        f = function(x, y, ...) {colttests(as.matrix(x), y)$p.value},
        threshold = threshold,
        fdr = fdr,
        nFeat = nFeat
    )
    class(selector) = c("FastTSelector", class(selector))
    return(selector)
}


PearsonSelector = function(nFeat, type="abs") {
    nFeat = nFeat
    type = type
    selector = function(x, y, ...) {
        x = scale(x)
        y = as.numeric(scale(as.numeric(y)))
        xycors = as.vector(y %*% x) / (length(y)-1)
        names(xycors) = colnames(x)
        xycors = get(type)(xycors)
        selectedFeatures = head(
            names(sort(xycors, decreasing=TRUE)),
            n = nFeat
        )
        return(list(
            selectedFeatures = selectedFeatures,
            transform = function(x, ...) {
                x[ , selectedFeatures, drop=FALSE]
            }
        ))
    }
    class(selector) = c("PearsonSelector", "ModelPipe", class(selector))
    return(selector)
}



PcaExtractor = function(k, center=c("both", "column", "row", "none"),
        scale=c("none", "column", "row")) {
    k = k
    center = match.arg(center)
    scale = match.arg(scale)
    learner = function(x, ...) {
        colAvStore = NA
        colSdStore = NA
        if (center %in% c("row", "both")) {
            x = sweep(x, 1, STATS=rowMeans(x))
        }
        if (center %in% c("column", "both")) {
            colAvStore = colMeans(x)
            x = sweep(x, 2, STATS=colAvStore)
        }
        if (scale == "row") {
            x = sweep(x, 1, STATS=rowSds(x), FUN=`/`)
        } else if (scale == "column") {
            colSdStore = colSds(x)
            x = sweep(x, 2, STATS=colSdStore, FUN=`/`)
        }
        xsvd = svd(x)
        v = xsvd$v[ , order(xsvd$d, decreasing=TRUE)]
        v = v[ , 1:k, drop=FALSE]
        rownames(v) = colnames(x)
        return(list(
            k = k,
            center = center,
            scale = scale,
            v = v,
            colAvStore = colAvStore,
            colSdStore = colSdStore,
            transform = function(x, ...) {
                if (center %in% c("row", "both")) {
                    x = sweep(x, 1, STATS=rowMeans(x))
                }
                if (center %in% c("column", "both")) {
                    x = sweep(x, 2, STATS=colAvStore)
                }
                if (scale == "row") {
                    x = sweep(x, 1, STATS=rowSds(x), FUN=`/`)
                } else if (scale == "column") {
                    x = sweep(x, 2, STATS=colSdStore, FUN=`/`)
                }
                return(data.frame(as.matrix(x) %*% v, check.names=FALSE))
            }
        ))
    }
    class(learner) = c("PcaExtractor", "ModelPipe", class(learner))
    return(learner)
}


logRpm = function(x, ..., offset=1) {
    x = 1e6 * sweep(
        x = x,
        MARGIN = 1,
        STATS = rowSums(x),
        FUN = `/`
    )
    return(log2(x+offset))
}
logRpmTransformer = XTransformer(logRpm)


logUq = function(x, ..., MARGIN=1, scale=100) {
    ## geneDetected = (apply(x, 3-MARGIN, sum) > 0)
    geneDetected = if (MARGIN == 1) {
        colSums(x) > 0
    } else if (MARGIN == 2) {
        rowSums(x) > 0
    }
    x = scale * sweep(
        x = x,
        MARGIN = MARGIN,
        STATS = apply(X=x, MARGIN=MARGIN,
                FUN=function(z) {quantile(z[geneDetected], 0.75)}),
        FUN = `/`
    )
    return(log2(x+1))
}
LogUqFitter = function(scale=100) {
    scale = scale
    learner = function(x, y, ...) {
        geneDetected = (colSums(x) > 0)
        return(list(
            transform = function(x, ...) {
                log2(scale * sweep(
                    X = x[ , geneDetected],
                    MARGIN = 1,
                    FUN = quantile,
                    probs = 0.75
                ) + 1)
            }
        ))
    }
    class(learner) = c("LogUqFitter", "ModelPipe", class(learner))
    return(learner)
}


KnnFitter = function(k=5) {
    k = k
    out = ModelFitter(
        f = function(x, y, ...) {
            list(k=k, x=x, y=y)
        },
        predictor = function(obj, x, ...) {
            require(class)
            knnObj = knn(train=obj$x, test=x, cl=obj$y, k=obj$k,
                    prob=TRUE, use.all=TRUE)
            predictions = sign(as.numeric(knnObj) - 1.5)
            predictions = predictions * (attr(knnObj, "prob") - 0.5)
            predictions = predictions + 0.5
            names(predictions) = rownames(x)
            return(predictions)
        }
    )
    class(out) = c("KnnFitter", class(out))
    return(out)
}


dldaFitter = ModelFitter(
    f = function(x, y, ...) {
        require(sparsediscrim)
        return(dlda(x, y, prior=c(0.5, 0.5)))
    },
    predictionProcessor = function(discriminants) {
        exponentiatedDiscriminants = exp(sweep(
            x = -discriminants$scores,
            MARGIN = 2,
            STATS = apply(-discriminants$scores, 2, max),
            FUN = `-`
        ))
        probs = sweep(
            x = exponentiatedDiscriminants,
            MARGIN = 2,
            STATS = colSums(exponentiatedDiscriminants),
            FUN = `/`
        )
        return(probs[2, ])
    }
)


LdaFitter = function(...) {
    ldaArgs = list(...)
    out = ModelFitter(
        f = function(x, y, ...) {
            require(MASS)
            do.call(lda, c(list(x, y), ldaArgs))
        },
        predictor = function(obj, x, ...) {
            predictions = predict(obj, x, ...)$posterior[ , 2]
            names(predictions) = rownames(x)
            return(predictions)
        }
    )
    class(out) = c("LdaFitter", class(out))
    return(out)
}


QdaFitter = function(...) {
    qdaArgs = list(...)
    out = ModelFitter(
        f = function(x, y, ...) {
            require(MASS)
            do.call(qda, c(list(x=x, y=y), qdaArgs))
        },
        predictor = function(obj, x, ...) {
            predictions = predict(obj, x, ...)$posterior[ , 2]
            names(predictions) = rownames(x)
            return(predictions)
        }
    )
    class(out) = c("QdaFitter", class(out))
    return(out)
}


GlmFitter = function(fam="binomial", alpha=0, lambda=NULL) {
    fam = fam
    al = alpha
    lam = lambda
    out = ModelFitter(
        f = function(x, y, ..., lambda=lam) {
            require(glmnet)
            if (length(lambda) == 0) {
                cvOut = cv.glmnet(as.matrix(x), y,
                        family=fam, alpha=al)
                lambda = cvOut$lambda.min
            }
            out = glmnet(as.matrix(x), y,
                    family=fam, alpha=al, lambda=lam)
            out$lambda.min = lambda
            return(out)
        },
        predictor = function(obj, x, ..., lambda) {
            if (missing(lambda)) {
                lambda = obj$lambda.min
            }
            return(predict(obj, as.matrix(x),
                    s=lambda, type="response")[ , 1])
        }
    )
    class(out) = c("GlmFitter", class(out))
    return(out)
}


SvmFitter = function(...) {
    svmArgs = list(...)
    svmArgs$probability = TRUE
    out = ModelFitter(
        f = function(x, y, ...) {
            require(e1071)
            if (!all(unique(as.character(y)) == levels(y))) {
                ord = order(y)
                x = x[ord, , drop=FALSE]
                y = y[ord]
            }
            do.call(svm, c(list(x=x, y=y), svmArgs))
        },
        predictor = function(obj, x, ...) {
            predictions = attr(predict(obj, x,
                    probability=TRUE, ...), "probabilities")[ , 2]
            names(predictions) = rownames(x)
            return(predictions)
        }
    )
    class(out) = c("SvmFitter", class(out))
    return(out)
}


RandomForestFitter = function(...) {
    rfArgs = list(...)
    out = ModelFitter(
        f = function(x, y, ...) {
            require(randomForest)
            do.call(randomForest, c(list(x=x, y=y), rfArgs))
        },
        predictor = function(obj, x, ...) {
            ## x = data.frame(x)
            predictions = predict(obj, x, type="prob")[ , 2]
            names(predictions) = rownames(x)
            return(predictions)
        }
    )
    class(out) = c("RandomForestFitter", class(out))
    return(out)
}


AdaFitter = function(...) {
    adaArgs = list(...)
    out = ModelFitter(
        f = function(x, y, ...) {
            require(ada)
            do.call(ada, c(list(x=x, y=y), adaArgs))
        },
        predictor = function(obj, x, ...) {
            x = data.frame(x)
            predictions = predict(obj, x, type="probs")[ , 2]
            names(predictions) = rownames(x)
            return(predictions)
        }
    )
    class(out) = c("AdaFitter", class(out))
    return(out)
}

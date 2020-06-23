bindArgs = function(f, ...) {
    args = list(...)
    return(function(...) {do.call(f, args=c(list(...), args))})
}

## -----------------------------------------------------------------------------
selectByTTest = function(x, y, m) {
    ## use genefilter package for efficient repeated t-test functions
    require(genefilter)
    ## assume x is samples-in-rows, genes-in-columns format!
    p = colttests(as.matrix(x), y)$p.value
    ## sort genes by order of p, return first m
    return(colnames(x)[order(p)[1:m]])
}

predict.FeatureSelectedFitModel = function(object, x, ...) {
    x = x[ , object$features, drop=FALSE]
    return(predict(object$fit, data.frame(x, check.names=FALSE), ...))
}

featSelFit = function(x, y, selector, fitter) {
    ## use selector function to select features using data x, y:
    features = selector(x, y)
    ## retain only selected features in x for fitting knn model:
    x = x[ , features, drop=FALSE]
    ## fit the desired using the selected feature set:
    fit = fitter(x, y)
    ## declare this list to be a FeatureSelectedFitModel object:
    out = list(features=features, fit=fit)
    ## declare this list to be a FeatureSelectedFitModel object:
    class(out) = "FeatureSelectedFitModel"
    return(out)
}

## -----------------------------------------------------------------------------
extractPCs = function(x, m=min(dim(x)), ...) {
    ## assume x is samples-in-rows, genes-in-columns format!
    pca = prcomp(x, center=TRUE, scale.=FALSE)
    mu = pca$center   ## training-set estimated mean expression of each gene
    ## extract matrix needed to project new data onto first m extracted PCs:
    projection = pca$rotation[ , 1:m, drop=FALSE]
    ## define extraction function to extract features from new data:
    extractor = function(newdata) {
        ## sweep out gene means (as estimated from training data, not newdata!):
        newdata = sweep(newdata, 2, mu, `-`)
        return(as.matrix(newdata) %*% projection)
    }
    ## return the function "extractor" which can be applied to newdata;
    ## this function yields coordinates of samples in newdata in PC-space
    ## learned from the training data passed in as x argument.
    return(extractor)
}

## write a function telling R how to make predictions from a
## FeatureExtractedFitModel object:
predict.FeatureExtractedFitModel = function(object, x, ...) {
    ## first extract the features using object$extractor:
    x = object$extractor(x)
    ## now predict using object$fit on the extracted features:
    return(predict(object$fit, data.frame(x, check.names=FALSE), ...))
}

featExtFit = function(x, y, extractionLearner, fitter) {
    ## use extractionLearner function to learn extractor using data x, y:
    extractor = extractionLearner(x, y)
    ## extract features from x for fitting knn model:
    x = extractor(x)
    ## fit the desired model using the selected feature set:
    fit = fitter(x, y)
    ## package results in list; need to remember extractor and fit:
    out = list(extractor=extractor, fit=fit)
    ## declare this list to be a FeatureExtractedFitModel object:
    class(out) = "FeatureExtractedFitModel"
    return(out)
}


## -----------------------------------------------------------------------------
regularizedGLM = function(x, y, family=gaussian, alpha=0, lambda=NULL, ...) {
    require(glmnet)
    if (length(lambda) == 0) {
        glmnetCV = cv.glmnet(as.matrix(x), y, family=family, alpha=alpha, ...)
        lambda = glmnetCV$lambda.min
        out = glmnetCV$glmnet.fit
        out$lambda.min = glmnetCV$lambda.min
    } else {
        out = glmnet(as.matrix(x), y,
                     family=family, alpha=alpha, lambda=lambda, ...)
        out$lambda.min = lambda
    }
    class(out) = c("regularizedGLM", class(out))
    return(out)
}

coef.regularizedGLM = function(modelFit, ...) {
    predict.glmnet(modelFit, s=modelFit$lambda.min,
                   type="coefficient", ...)[ , 1]
}

predict.regularizedGLM = function(modelFit, newdata, type="response", ...) {
    out = predict.glmnet(modelFit, as.matrix(newdata),
                         s=modelFit$lambda.min, type="response", ...)[ , 1]
    if (type == "class") {
        return(1 + (out >= 0))
    } else if (type == "response") {
        return(out)
    } else if (type == "prob") {
        return(1 / (1+exp(-out)))
    }
}

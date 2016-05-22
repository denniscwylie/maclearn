from collections import OrderedDict
import copy
import numpy
from numpy import mean
import pandas
from pandas import DataFrame
from pandas import Series
import scipy
import sklearn
import sklearn.cross_validation
from sklearn.cross_validation import ShuffleSplit
import sklearn.feature_selection
import sklearn.pipeline
import sklearn.svm

import MaclearnUtilities

import RestrictedData
xs = RestrictedData.xs
xnorms = RestrictedData.xnorms
annots = RestrictedData.annots
ys = RestrictedData.ys
ynums = RestrictedData.ynums

cvSchedules = {k : ShuffleSplit(len(ys[k]),
                                n_iter = 5,
                                test_size = 0.2,
                                random_state = 123)
               for k in xnorms}


def pandaize(f):
    def pandaized(estimator, X, y, **kwargs):
        return f(estimator, array(X), y, **kwargs)
    return pandaized

@pandaize
def cross_val_score_pd(estimator, X, y, **kwargs):
    return sklearn.cross_validation.cross_val_score(
            estimator, X, y, **kwargs)

def fitModelWithNFeat(fitter, n, setname, cv=None):
    if cv is None:
        cv = cvSchedules[setname]
    if n > xnorms[setname].shape[1]:
        return None
    fsFitter = sklearn.pipeline.Pipeline([
        ('featsel', sklearn.feature_selection.SelectKBest(
                sklearn.feature_selection.f_regression, k=n)),
        ('classifier', fitter)
    ])
    return mean(cross_val_score_pd(estimator = fsFitter,
                                   X = xnorms[setname],
                                   y = ynums[setname],
                                   cv = cv))


svmLinAccs = {
    s : fitModelWithNFeat(
        fitter = sklearn.svm.SVC(kernel="linear", C=1),
        n = 10,
        setname = s
    )
    for s in xnorms
}

svmRadAccs = {
    s : fitModelWithNFeat(
        fitter = sklearn.svm.SVC(kernel="rbf", C=1), # use default gamma
        n = 10,
        setname = s
    )
    for s in xnorms
}

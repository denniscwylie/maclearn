from collections import OrderedDict
import numpy
import pandas
from pandas import DataFrame
from pandas import Series
import sklearn
import sklearn.cross_validation
from sklearn.cross_validation import ShuffleSplit
import sklearn.feature_selection
import sklearn.neighbors
import sklearn.pipeline

import pcaextractor
import MaclearnUtilities
from MaclearnUtilities import safeFactorize

import RestrictedData
xs = RestrictedData.xs
xnorms = RestrictedData.xnorms
annots = RestrictedData.annots
ys = RestrictedData.ys
ynums = RestrictedData.ynums

## Note sklearn has some nice built-in capabilities
## for tuning model parameters over a grid of potential values...
## Here doing things manually instead for more explicit illustration.


def pandaize(f):
    def pandaized(estimator, X, y, **kwargs):
        return f(estimator, array(X), safeFactorize(y), **kwargs)
    return pandaized

@pandaize
def cross_val_score_pd(estimator, X, y, **kwargs):
    return sklearn.cross_validation.cross_val_score(
            estimator, X, y, **kwargs)


def fsKnnFitterGenerator(k):
    return sklearn.pipeline.Pipeline([
        ('featsel', sklearn.feature_selection.SelectKBest(
                sklearn.feature_selection.f_regression, k=10)),
        ('classifier', sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=k))
    ])

cvSchedules = {k : ShuffleSplit(len(ys[k]),
                                n_iter = 5,
                                test_size = 0.2,
                                random_state = 123)
               for k in xnorms}

ks = [3, 5, 9, 15]
knnModels = [
    OrderedDict([
        (s, mean(cross_val_score_pd(
            estimator = fsKnnFitterGenerator(k),
            X = xnorms[s],
            y = ys[s],
            cv = cvSchedules[s])))
        for s in xnorms
    ])
    for k in ks
]

knnCvAccs = DataFrame(knnModels, index=ks)


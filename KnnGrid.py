from collections import OrderedDict
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import sklearn as sk
import sklearn.model_selection as model_selection
from sklearn.model_selection import ShuffleSplit
import sklearn.feature_selection as feature_selection
import sklearn.neighbors as neighbors
import sklearn.pipeline as pipeline

import warnings
warnings.filterwarnings("ignore")

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
        return f(estimator, np.array(X), safeFactorize(y), **kwargs)
    return pandaized

@pandaize
def cross_val_score_pd(estimator, X, y, **kwargs):
    return model_selection.cross_val_score(
            estimator, X, y, **kwargs)


def fsKnnFitterGenerator(k):
    return pipeline.Pipeline([
        ('featsel', feature_selection.SelectKBest(
                feature_selection.f_regression, k=10)),
        ('classifier', neighbors.KNeighborsClassifier(
                n_neighbors=k))
    ])

cvSchedules = {k : ShuffleSplit(n_splits = 5,
                                test_size = 0.2,
                                random_state = 123)
               for k in xnorms}

ks = [3, 5, 9, 15]
knnModels = [
    OrderedDict([
        (s, np.mean(cross_val_score_pd(
            estimator = fsKnnFitterGenerator(k),
            X = xnorms[s],
            y = ys[s],
            cv = cvSchedules[s].split(xnorms[s]))))
        for s in xnorms
    ])
    for k in ks
]

knnCvAccs = DataFrame(knnModels, index=ks)


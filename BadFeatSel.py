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

import SimData


def pandaize(f):
    def pandaized(estimator, X, y, **kwargs):
        return f(estimator, array(X), safeFactorize(y), **kwargs)
    return pandaized

@pandaize
def cross_val_score_pd(estimator, X, y, **kwargs):
    return model_selection.cross_val_score(
            estimator, X, y, **kwargs)


fsKnnFitter = pipeline.Pipeline([
    ('featsel', feature_selection.SelectKBest(
            feature_selection.f_regression, k=10)),
    ('classifier', neighbors.KNeighborsClassifier(
            n_neighbors=3))
])

simData = SimData.simulate2Group(n=40, p=1000, effect=[0]*1000)
x = simData['x']
y = simData['y']

simSelBad = feature_selection.SelectKBest(
        feature_selection.f_regression, k=10).fit(x, y)
xbad = simSelBad.transform(x)
cvbad = np.mean(model_selection.cross_val_score(
    estimator = neighbors.KNeighborsClassifier(n_neighbors=3),
    X = xbad,
    y = y,
    cv = 5
))

cvgood = np.mean(model_selection.cross_val_score(
    estimator = fsKnnFitter,
    X = x,
    y = y,
    cv = 5
))

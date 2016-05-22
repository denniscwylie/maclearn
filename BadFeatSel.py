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

import SimData


def pandaize(f):
    def pandaized(estimator, X, y, **kwargs):
        return f(estimator, array(X), safeFactorize(y), **kwargs)
    return pandaized

@pandaize
def cross_val_score_pd(estimator, X, y, **kwargs):
    return sklearn.cross_validation.cross_val_score(
            estimator, X, y, **kwargs)


fsKnnFitter = sklearn.pipeline.Pipeline([
    ('featsel', sklearn.feature_selection.SelectKBest(
            sklearn.feature_selection.f_regression, k=10)),
    ('classifier', sklearn.neighbors.KNeighborsClassifier(
            n_neighbors=3))
])

simData = SimData.simulate2Group(n=40, p=1000, effect=[0]*1000)
x = simData['x']
y = simData['y']

simSelBad = sklearn.feature_selection.SelectKBest(
        sklearn.feature_selection.f_regression, k=10).fit(x, y)
xbad = simSelBad.transform(x)
cvbad = mean(sklearn.cross_validation.cross_val_score(
    estimator = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3),
    X = xbad,
    y = y,
    cv = 5
))

cvgood = mean(sklearn.cross_validation.cross_val_score(
    estimator = fsKnnFitter,
    X = x,
    y = y,
    cv = 5
))

from collections import OrderedDict
import copy
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import sklearn as sk
import sklearn.model_selection as model_selection
from sklearn.model_selection import ShuffleSplit
import sklearn.ensemble as ensemble
import sklearn.feature_selection as feature_selection
import sklearn.pipeline as pipeline

import warnings
warnings.filterwarnings("ignore")

import MaclearnUtilities
from MaclearnUtilities import bhfdr, colcor

import RestrictedData
xs = RestrictedData.xs
xnorms = RestrictedData.xnorms
annots = RestrictedData.annots
ys = RestrictedData.ys
ynums = RestrictedData.ynums

cvSchedules = {k : ShuffleSplit(n_splits = 5,
                                test_size = 0.2,
                                random_state = 123)
               for k in xnorms}


def pandaize(f):
    def pandaized(estimator, X, y, **kwargs):
        return f(estimator, np.array(X), y, **kwargs)
    return pandaized

@pandaize
def cross_val_score_pd(estimator, X, y, **kwargs):
    return model_selection.cross_val_score(estimator, X, y, **kwargs)

def fitModelWithNFeat(fitter, n, setname, cv=None):
    if cv is None:
        cv = cvSchedules[setname]
    if n > xnorms[setname].shape[1]:
        return None
    fsFitter = pipeline.Pipeline([
        ('featsel', feature_selection.SelectKBest(
                feature_selection.f_regression, k=n)),
        ('classifier', fitter)
    ])
    return np.mean(cross_val_score_pd(estimator = fsFitter,
                                      X = xnorms[setname],
                                      y = ynums[setname],
                                      cv = cv.split(xnorms[setname])))


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

fsAda50Accs = {
    s : fitModelWithNFeat(
        fitter = AdaBoostClassifier(
            base_estimator = DecisionTreeClassifier(
                max_depth = 3,
                min_samples_split = 20,
                min_samples_leaf = 7
            ),
            learning_rate = 0.05,
            n_estimators = 50
        ),
        n = 10,
        setname = s
    )
    for s in xnorms
}

fsAda100Accs = {
    s : fitModelWithNFeat(
        fitter = AdaBoostClassifier(
            base_estimator = DecisionTreeClassifier(
                max_depth = 3,
                min_samples_split = 20,
                min_samples_leaf = 7
            ),
            learning_rate = 0.05,
            n_estimators = 100
        ),
        n = 10,
        setname = s
    )
    for s in xnorms
}

fsAda250Accs = {
    s : fitModelWithNFeat(
        fitter = AdaBoostClassifier(
            base_estimator = DecisionTreeClassifier(
                max_depth = 3,
                min_samples_split = 20,
                min_samples_leaf = 7
            ),
            learning_rate = 0.05,
            n_estimators = 250
        ),
        n = 10,
        setname = s
    )
    for s in xnorms
}

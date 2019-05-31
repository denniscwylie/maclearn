from collections import OrderedDict
import copy
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import sklearn as sk
import sklearn.model_selection as model_selection
from sklearn.model_selection import ShuffleSplit
import sklearn.ensemble as ensemble
import sklearn.feature_selection as feature_selection
import sklearn.discriminant_analysis as discriminant_analysis
import sklearn.linear_model as linear_model
import sklearn.naive_bayes as naive_bayes
import sklearn.neighbors as neighbors
import sklearn.pipeline as pipeline
import sklearn.tree as tree

import warnings
warnings.filterwarnings("ignore")

import MaclearnUtilities

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

fitters = {
    "knn5" : neighbors.KNeighborsClassifier(n_neighbors=5),
    "knn9" : neighbors.KNeighborsClassifier(n_neighbors=9),
    "logistic" : linear_model.LogisticRegression(C=1e10),
    "l1" : linear_model.LogisticRegression(penalty="l1", C=1),
    "l2" : linear_model.LogisticRegression(penalty="l2", C=1),
    "lda" : discriminant_analysis.LinearDiscriminantAnalysis(),
    "nb_gauss" : naive_bayes.GaussianNB(),
    "rf" : ensemble.RandomForestClassifier(n_estimators=500),
    "ada" : ensemble.AdaBoostClassifier(
        base_estimator = tree.DecisionTreeClassifier(
            max_depth = 3,
            min_samples_split = 20,
            min_samples_leaf = 7
        ),
        learning_rate = 0.05,
        n_estimators = 100
    )
}


modelFits10 = {}
for s in xnorms:
    modelFits10[s] = {f : fitModelWithNFeat(fitters[f],
                                            n = 10,
                                            setname = s)
                      for f in fitters}

modelFits10 = DataFrame(modelFits10)


modelFits20 = {}
for s in xnorms:
    modelFits20[s] = {f : fitModelWithNFeat(fitters[f],
                                           n = 20,
                                           setname = s)
                      for f in fitters}

modelFits20 = DataFrame(modelFits20)


modelFits50 = {}
for s in xnorms:
    modelFits50[s] = {f : fitModelWithNFeat(fitters[f],
                                            n = 50,
                                            setname = s)
                      for f in fitters}

modelFits50 = DataFrame(modelFits50)

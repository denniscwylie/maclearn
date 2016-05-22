from collections import OrderedDict
import copy
import numpy
import pandas
from pandas import DataFrame
from pandas import Series
import scipy
import sklearn
import sklearn.cross_validation
from sklearn.cross_validation import ShuffleSplit
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.discriminant_analysis
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.pipeline

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

fitters = {
    "knn5" : sklearn.neighbors.KNeighborsClassifier(n_neighbors=5),
    "knn9" : sklearn.neighbors.KNeighborsClassifier(n_neighbors=9),
    "logistic" : sklearn.linear_model.LogisticRegression(C=1e10),
    "l1" : sklearn.linear_model.LogisticRegression(penalty="l1", C=1),
    "l2" : sklearn.linear_model.LogisticRegression(penalty="l2", C=1),
    "lda" : sklearn.discriminant_analysis.LinearDiscriminantAnalysis(),
    "nb_gauss" : sklearn.naive_bayes.GaussianNB(),
    "rf" : sklearn.ensemble.RandomForestClassifier(n_estimators=500),
    "ada" : sklearn.ensemble.AdaBoostClassifier(
        base_estimator = sklearn.tree.DecisionTreeClassifier(
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

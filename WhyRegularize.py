from collections import OrderedDict
import copy
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import scipy
import sklearn as sk
import sklearn.model_selection as model_selection
from sklearn.model_selection import ShuffleSplit
import sklearn.feature_selection as feature_selection
import sklearn.linear_model as linear_model
import sklearn.pipeline as pipeline

import MaclearnUtilities
from MaclearnUtilities import bhfdr, colcor

def pandaize(f):
    def pandaized(estimator, X, y, **kwargs):
        return f(estimator, np.array(X), y, **kwargs)
    return pandaized

@pandaize
def cross_val_score_pd(estimator, X, y, **kwargs):
    return model_selection.cross_val_score(estimator, X, y, **kwargs)


## -----------------------------------------------------------------
## linear regression simulated example
## -----------------------------------------------------------------
x = np.random.randn(15, 4)
x[:, 1] = x[:, 0] + 0.01 * x[:, 1]

y = x[:, 3] + np.random.randn(15)

linmod = linear_model.LinearRegression().fit(x, y)
linmod.coef_

l2mod = linear_model.Ridge(alpha=15*0.1).fit(x, y)
l2mod.coef_

l1mod = linear_model.Lasso(alpha=0.1).fit(x, y)
l1mod.coef_


## -----------------------------------------------------------------
## load Hess data
## -----------------------------------------------------------------
def readTab(file):
    return pd.read_csv(file, sep="\t", header=0, index_col=0)

x = readTab("microarray/Hess/HessTrainingData.tsv.gz").transpose()
annot = readTab("microarray/Hess/HessTrainingAnnotation.tsv")
y = MaclearnUtilities.safeFactorize(annot.pCRtxt)

logisticFitter = pipeline.Pipeline([
    ('featsel', feature_selection.SelectKBest(
            feature_selection.f_regression, k=4)),
    ('classifier', linear_model.LogisticRegression(C=1e15))
])
logisticFit = copy.deepcopy(logisticFitter).fit(x, y)
logisticCoef = logisticFit.get_params()['classifier'].coef_


## -----------------------------------------------------------------
## regularized models
## -----------------------------------------------------------------
l2Fitter = pipeline.Pipeline([
    ('featsel', feature_selection.SelectKBest(
            feature_selection.f_regression, k=4)),
    ('classifier', linear_model.LogisticRegression(
            C=20.0/len(y), penalty="l2", intercept_scaling=100))
])
l2Fit = copy.deepcopy(l2Fitter).fit(x, y)
l2Coef = l2Fit.get_params()['classifier'].coef_

l1Fitter = pipeline.Pipeline([
    ('featsel', feature_selection.SelectKBest(
            feature_selection.f_regression, k=4)),
    ('classifier', linear_model.LogisticRegression(
            C=20.0/len(y), penalty="l1", intercept_scaling=100))
])
l1Fit = copy.deepcopy(l1Fitter).fit(x, y)
l1Coef = l1Fit.get_params()['classifier'].coef_


## -----------------------------------------------------------------
## 
## -----------------------------------------------------------------
cvSchedule = ShuffleSplit(n_splits=5, test_size=0.2, random_state=123)

cvLogisticAcc = np.mean(cross_val_score_pd(estimator = logisticFitter,
                                           X = x,
                                           y = y,
                                           cv = cvSchedule.split(x)))
cvLogisticAcc

cvL1Acc = np.mean(cross_val_score_pd(estimator = l1Fitter,
                                     X = x,
                                     y = y,
                                     cv = cvSchedule.split(x)))
cvL1Acc

cvL2Acc = np.mean(cross_val_score_pd(estimator = l2Fitter,
                                     X = x,
                                     y = y,
                                     cv = cvSchedule.split(x)))
cvL2Acc

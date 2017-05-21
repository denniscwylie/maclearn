from collections import OrderedDict
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import sklearn as sk
import sklearn.cross_validation as cross_validation
from sklearn.cross_validation import ShuffleSplit
import sklearn.feature_selection as feature_selection
import sklearn.linear_model as linear_model
import sklearn.pipeline as pipeline

plt.ion()
plt.style.use('fivethirtyeight')

import MaclearnUtilities
from MaclearnUtilities import bhfdr, colcor

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
        return f(estimator, np.array(X), y, **kwargs)
    return pandaized

@pandaize
def cross_val_score_pd(estimator, X, y, **kwargs):
    return cross_validation.cross_val_score(
            estimator, X, y, **kwargs)

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
                                      cv = cv))

def accPlot(accsByNFeats):
    ax = plt.subplot(111)
    for s in accsByNFeats:
        plotdata = pd.concat([DataFrame({"p" : p,
                                         "acc" : accsByNFeats[s][p]},
                                        index = [str(p)])
                              for p in accsByNFeats[s]],
                             axis = 0)
        plotdata.plot(x = "p",
                      y = "acc",
                      ax = ax,
                      logx = True,
                      label = s)


nFeatures = [2, 5, 10, 20, 50, 100, 200, 500,
             1000, 2000, 5000, 10000]


## -----------------------------------------------------------------
## no (err...very little) regularization
## -----------------------------------------------------------------
def fitLogisticWithNFeat(**kwargs):
    fitter = linear_model.LogisticRegression(penalty="l2", C=1e10)
    return fitModelWithNFeat(fitter=fitter, **kwargs)

nFeatNoReg = [2, 5, 10, 20, 50, 100, 200]
accsByNFeats = OrderedDict([(s,
                             OrderedDict([(
                                 n,
                                 fitLogisticWithNFeat(n=n, setname=s))
                                          for n in nFeatNoReg]))
                            for s in xnorms])
for s in accsByNFeats:
    for n in accsByNFeats[s]:
        if n > xnorms[s].shape[0]:
            accsByNFeats[s][n] = None

plt.clf()
accPlot(accsByNFeats)


## -----------------------------------------------------------------
## L2 regularization
## -----------------------------------------------------------------
def fitL2LogisticWithNFeat(**kwargs):
    fitter = linear_model.LogisticRegression(penalty="l2", C=1)
    return fitModelWithNFeat(fitter=fitter, **kwargs)

accsByNFeatsL2 = OrderedDict([(s,
                               OrderedDict([(
                                   n,
                                   fitL2LogisticWithNFeat(n=n, setname=s))
                                            for n in nFeatures]))
                              for s in xnorms])

plt.clf()
accPlot(accsByNFeatsL2)



## -----------------------------------------------------------------
## L1 regularization
## -----------------------------------------------------------------
def fitL1LogisticWithNFeat(**kwargs):
    fitter = linear_model.LogisticRegression(penalty="l1", C=1)
    return fitModelWithNFeat(fitter=fitter, **kwargs)

accsByNFeatsL1 = OrderedDict([(s,
                               OrderedDict([(
                                   n,
                                   fitL1LogisticWithNFeat(n=n, setname=s))
                                            for n in nFeatures]))
                              for s in xnorms])

plt.clf()
accPlot(accsByNFeatsL1)

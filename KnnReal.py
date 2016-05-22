from collections import OrderedDict
import numpy
from numpy import mean
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

plt.style.use("fivethirtyeight")

import RestrictedData
xs = RestrictedData.xs
xnorms = RestrictedData.xnorms
annots = RestrictedData.annots
ys = RestrictedData.ys
ynums = RestrictedData.ynums


def pandaize(f):
    def pandaized(estimator, X, y, **kwargs):
        return f(estimator, array(X), safeFactorize(y), **kwargs)
    return pandaized

@pandaize
def cross_val_score_pd(estimator, X, y, **kwargs):
    return sklearn.cross_validation.cross_val_score(
            estimator, X, y, **kwargs)


ys = {
    'bottomly' : annots['bottomly'].strain,
    'patel' : annots['patel'].SubType,
    'montastier' : annots['montastier'].Time,
    'hess' : annots['hess'].pCRtxt
}

knnClass = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
cvSchedules = {k : ShuffleSplit(len(ys[k]),
                                n_iter = 5,
                                test_size = 0.2,
                                random_state = 123)
               for k in xnorms}
knnCvAccs = {k : mean(cross_val_score_pd(estimator = knnClass,
                                         X = xnorms[k],
                                         y = ys[k],
                                         cv = cvSchedules[k]))
             for k in xnorms}


## -----------------------------------------------------------------
## try with univariate filter feature selection
## -----------------------------------------------------------------
fsKnnFitter = sklearn.pipeline.Pipeline([
    ('featsel', sklearn.feature_selection.SelectKBest(
            sklearn.feature_selection.f_regression, k=10)),
    ('classifier', sklearn.neighbors.KNeighborsClassifier(
            n_neighbors=3))
])

fsKnnCvAccs = {k : mean(cross_val_score_pd(estimator = fsKnnFitter,
                                           X = xnorms[k],
                                           y = ys[k],
                                           cv = cvSchedules[k]))
               for k in xnorms}


## -----------------------------------------------------------------
## vary number of features used
## -----------------------------------------------------------------
nFeatures = [1, 2, 5, 10, 20, 50, 100, 200, 500,
             1000, 2000, 5000, 10000]
def fitKnnWithNFeat(n, setname, cv=None):
    if cv is None:
        cv = cvSchedules[setname]
    if n > xnorms[setname].shape[1]:
        return None
    fsKnnFitter = sklearn.pipeline.Pipeline([
        ('featsel', sklearn.feature_selection.SelectKBest(
                sklearn.feature_selection.f_regression, k=n)),
        ('classifier', sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=3))
    ])
    return mean(cross_val_score_pd(estimator = fsKnnFitter,
                                   X = xnorms[setname],
                                   y = ys[setname],
                                   cv = cv))

accsByNFeats = OrderedDict([(s, OrderedDict([(n, fitKnnWithNFeat(n, s))
                                             for n in nFeatures]))
                            for s in xnorms])

plotData = pandas.concat([DataFrame({"set" : s,
                                     "p" : p,
                                     "acc" : accsByNFeats[s][p]},
                                    index = [s + "_" + str(p)])
                          for s in accsByNFeats
                          for p in accsByNFeats[s]],
                         axis = 0)
plt.clf()
ax = plt.subplot(111)
for s in plotData['set'].unique():
    plotData.ix[plotData['set']==s].plot(x = "p",
                                         y = "acc",
                                         logx = True,
                                         ax = ax,
                                         label = s)

# plotData.to_csv("KnnRealAccuracyByNFeat.tsv",
#                 sep = "\t",
#                 index = False,
#                 header = True)


## -----------------------------------------------------------------
## use PCA feature extraction
## -----------------------------------------------------------------
feKnnFitter = sklearn.pipeline.Pipeline([
    ('featextr', pcaextractor.PcaExtractor(k=3)),
    ('classifier', sklearn.neighbors.KNeighborsClassifier(
            n_neighbors=3))
])

xmod = feKnnFitter.fit(array(xnorms['patel']), array(ys['patel']))
xcv = cross_val_score_pd(feKnnFitter, xnorms['patel'], ys['patel'],
                         cv=cvSchedules['patel'])

feKnnCvAccs = {k : mean(cross_val_score_pd(estimator = feKnnFitter,
                                           X = xnorms[k],
                                           y = ys[k],
                                           cv = cvSchedules[k]))
               for k in xnorms}


## -----------------------------------------------------------------
## test with varying number of principal components
## -----------------------------------------------------------------
npcs = [1, 2, 5, 10, 20, 50, 100, 200]
def fitKnnWithNPcs(n, setname, cv=None):
    if cv is None:
        cv = cvSchedules[setname]
    if n > min(xnorms[setname].shape):
        return None
    feKnnFitter = sklearn.pipeline.Pipeline([
        ('featextr', pcaextractor.PcaExtractor(k=n)),
        ('classifier', sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=3))
    ])
    return mean(cross_val_score_pd(estimator = feKnnFitter,
                                   X = xnorms[setname],
                                   y = ys[setname],
                                   cv = cv))
accsByNPcs = OrderedDict([(s, OrderedDict([(n, fitKnnWithNPcs(n, s))
                                           for n in npcs]))
                            for s in xnorms])

plotData = pandas.concat([DataFrame({"set" : s,
                                     "p" : p,
                                     "acc" : accsByNPcs[s][p]},
                                    index = [s + "_" + str(p)])
                          for s in accsByNPcs
                          for p in accsByNPcs[s]],
                         axis = 0)
plt.clf()
ax = plt.subplot(111)
for s in plotData['set'].unique():
    plotData.ix[plotData['set']==s].plot(x = "p",
                                         y = "acc",
                                         logx = True,
                                         ax = ax,
                                         label = s)

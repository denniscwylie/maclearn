from collections import OrderedDict
import copy
import matplotlib.pyplot as plt
import numpy
from numpy import mean
import pandas
from pandas import DataFrame
from pandas import Series
import scipy
import sklearn
import sklearn.cross_validation
from sklearn.cross_validation import ShuffleSplit
import sklearn.feature_selection
import sklearn.linear_model
import sklearn.pipeline

import MaclearnUtilities
from MaclearnUtilities import bhfdr
from MaclearnUtilities import colcor

plt.style.use("fivethirtyeight")

def pandaize(f):
    def pandaized(estimator, X, y, **kwargs):
        return f(estimator, array(X), y, **kwargs)
    return pandaized

@pandaize
def cross_val_score_pd(estimator, X, y, **kwargs):
    return sklearn.cross_validation.cross_val_score(
            estimator, X, y, **kwargs)


## -----------------------------------------------------------------
## load Patel data
## -----------------------------------------------------------------
def readTab(file):
    return pandas.read_csv(file, sep="\t",
                           header=0, index_col=0)

x = readTab("rnaseq/GSE57872/GSE57872_DataMatrixMapped.tsv.gz").transpose()
y = x.BRCA1
x0 = x[ x.columns[x.columns != "BRCA1"] ]

cvSched = ShuffleSplit(len(y), n_iter=10, test_size=0.1, random_state=123)

corPVals = colcor(x0, y)['p']
corQVals = bhfdr(corPVals)
corQVals.sort_values(inplace=False).head()

plt.clf()
ax = plt.subplot(111)
x.plot.scatter(x="CDK1", y="BRCA1", ax=ax)


## -----------------------------------------------------------------
## unregularized linear regression
## -----------------------------------------------------------------
nFeats = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
brca1Modelers = OrderedDict([
    (n, sklearn.pipeline.Pipeline([
        ('featsel', sklearn.feature_selection.SelectKBest(
                sklearn.feature_selection.f_regression, k=n)),
        ('regressor', sklearn.linear_model.LinearRegression())
    ]))
    for n in nFeats
])
            
brca1Model20 = copy.deepcopy(brca1Modelers[20]).fit(x0, y)
brca1Preds = brca1Model20.predict(x0)
scipy.stats.pearsonr(brca1Preds, y)[0]

brca1Model1000 = copy.deepcopy(brca1Modelers[1000]).fit(x0, y)
brca1Preds = brca1Model1000.predict(x0)
scipy.stats.pearsonr(brca1Preds, y)[0]

cvR2s_unreg = Series(OrderedDict([
    (n, mean(cross_val_score_pd(copy.deepcopy(brca1Modelers[n]),
                                X = x0,
                                y = y,
                                cv = cvSched)))
    for n in nFeats
]))


## -----------------------------------------------------------------
## L2-regularized linear regression
## -----------------------------------------------------------------
brca1Modelers2 = OrderedDict([
    (n, sklearn.pipeline.Pipeline([
        ('featsel', sklearn.feature_selection.SelectKBest(
                sklearn.feature_selection.f_regression, k=n)),
        ('regressor', sklearn.linear_model.Ridge(
                alpha=len(y)*(1.5 + 0.034*n)))
    ]))
    for n in nFeats
])
            
cvR2s_L2 = Series(OrderedDict([
    (n, mean(cross_val_score_pd(copy.deepcopy(brca1Modelers2[n]),
                                X = x0,
                                y = y,
                                cv = cvSched)))
    for n in nFeats
]))


## -----------------------------------------------------------------
## L1-regularized linear regression
## -----------------------------------------------------------------
brca1Modelers1 = OrderedDict([
    (n, sklearn.pipeline.Pipeline([
        ('featsel', sklearn.feature_selection.SelectKBest(
                sklearn.feature_selection.f_regression, k=n)),
        ('regressor', sklearn.linear_model.Lasso(
                alpha=max(0, (0.0235*numpy.log(n)-0.0157))))
    ]))
    for n in nFeats
])
            
cvR2s_L1 = Series(OrderedDict([
    (n, mean(cross_val_score_pd(copy.deepcopy(brca1Modelers1[n]),
                                X = x0,
                                y = y,
                                cv = cvSched)))
    for n in nFeats
]))


## -----------------------------------------------------------------
## plot results
## -----------------------------------------------------------------
plotdata = DataFrame({
    "Number Potential Features" : nFeats * 3,
    "Rsquared" : pandas.concat([cvR2s_unreg, cvR2s_L2, cvR2s_L1]),
    "Regularization" : (['none']*len(nFeats) +
                        ['L2/ridge']*len(nFeats) +
                        ['L1/lasso']*len(nFeats))
})
plotdata.index = (plotdata["Number Potential Features"].apply(str) + "_" +
                plotdata["Regularization"])
plotdata = plotdata.ix[plotdata.Rsquared > 0]

plt.clf()
ax = plt.subplot(111)
regStyles = {'none' : '-', 'L2/ridge' : '--', 'L1/lasso' : ':'}
for reg in plotdata["Regularization"].unique():
    regdata = plotdata.ix[plotdata["Regularization"] == reg]
    regdata.sort_values("Number Potential Features")
    regdata.plot(x = "Number Potential Features",
                 y = "Rsquared",
                 color = "black",
                 style = regStyles[reg],
                 label = reg,
                 logx = True,
                 ax = ax)
                                                  

from collections import OrderedDict
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import plotnine as gg
import scipy as sp
import scipy.stats as stats
import sklearn as sk
import sklearn.model_selection as model_selection
from sklearn.model_selection import ShuffleSplit
import sklearn.feature_selection as feature_selection
import sklearn.linear_model as linear_model
import sklearn.pipeline as pipeline

import MaclearnUtilities
from MaclearnUtilities import bhfdr, colcor

plt.ion()

def pandaize(f):
    def pandaized(estimator, X, y, **kwargs):
        return f(estimator, np.array(X), y, **kwargs)
    return pandaized

@pandaize
def cross_val_score_pd(estimator, X, y, **kwargs):
    return model_selection.cross_val_score(estimator, X, y, **kwargs)


## -----------------------------------------------------------------
## load Patel data
## -----------------------------------------------------------------
def readTab(file):
    return pd.read_csv(file, sep="\t", header=0, index_col=0)

x = readTab("rnaseq/GSE57872/GSE57872_DataMatrixMapped.tsv.gz").transpose()
y = x.BRCA1
x0 = x[ x.columns[x.columns != "BRCA1"] ]

cvSched = ShuffleSplit(n_splits=10, test_size=0.1, random_state=123)

corPVals = colcor(x0, y)['p']
corQVals = bhfdr(corPVals)
corQVals.sort_values(inplace=False).head()

plt.close()
ax = plt.subplot(111)
x.plot.scatter(x="CDK1", y="BRCA1", ax=ax)


## -----------------------------------------------------------------
## unregularized linear regression
## -----------------------------------------------------------------
nFeats = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
brca1Modelers = OrderedDict([
    (n, pipeline.Pipeline([
        ('featsel', feature_selection.SelectKBest(
                feature_selection.f_regression, k=n)),
        ('regressor', linear_model.LinearRegression())
    ]))
    for n in nFeats
])
            
brca1Model20 = copy.deepcopy(brca1Modelers[20]).fit(x0, y)
brca1Preds = brca1Model20.predict(x0)
stats.pearsonr(brca1Preds, y)[0]

brca1Model1000 = copy.deepcopy(brca1Modelers[1000]).fit(x0, y)
brca1Preds = brca1Model1000.predict(x0)
stats.pearsonr(brca1Preds, y)[0]

cvR2s_unreg = Series(OrderedDict([
    (n, np.mean(cross_val_score_pd(copy.deepcopy(brca1Modelers[n]),
                                   X = x0,
                                   y = y,
                                   cv = cvSched.split(x0))))
    for n in nFeats
]))


## -----------------------------------------------------------------
## L2-regularized linear regression
## -----------------------------------------------------------------
brca1Modelers2 = OrderedDict([
    (n, pipeline.Pipeline([
        ('featsel', feature_selection.SelectKBest(
                feature_selection.f_regression, k=n)),
        ('regressor', linear_model.Ridge(
                alpha=len(y)*(1.5 + 0.034*n)))
    ]))
    for n in nFeats
])
            
cvR2s_L2 = Series(OrderedDict([
    (n, np.mean(cross_val_score_pd(copy.deepcopy(brca1Modelers2[n]),
                                   X = x0,
                                   y = y,
                                   cv = cvSched)))
    for n in nFeats
]))


## -----------------------------------------------------------------
## L1-regularized linear regression
## -----------------------------------------------------------------
brca1Modelers1 = OrderedDict([
    (n, pipeline.Pipeline([
        ('featsel', feature_selection.SelectKBest(
                feature_selection.f_regression, k=n)),
        ('regressor', linear_model.Lasso(
                alpha=max(0, (0.0235*np.log(n)-0.0157))))
    ]))
    for n in nFeats
])
            
cvR2s_L1 = Series(OrderedDict([
    (n, np.mean(cross_val_score_pd(copy.deepcopy(brca1Modelers1[n]),
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
    "Rsquared" : pd.concat([cvR2s_unreg, cvR2s_L2, cvR2s_L1]),
    "Regularization" : (['-']*len(nFeats) +
                        ['L2/ridge']*len(nFeats) +
                        ['L1/lasso']*len(nFeats))
})
plotdata.index = (plotdata["Number Potential Features"].apply(str) + "_" +
                plotdata["Regularization"])
plotdata = plotdata.loc[plotdata.Rsquared > 0]

plt.close()
ggo = gg.ggplot(plotdata, gg.aes(x = 'Number Potential Features',
                                 y = 'Rsquared',
                                 linetype = 'Regularization'))
ggo += gg.geom_line()
ggo += gg.theme_bw()
ggo += gg.scale_x_log10()
print(ggo)


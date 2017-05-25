from collections import OrderedDict
# import ggplot
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier

plt.ion()

import SimData


x2_train = SimData.simulate2Group(n = 100,
                                  p = 2,
                                  effect = [1.25] * 2)
knnFit = KNeighborsClassifier(n_neighbors=3)
knnFit.fit(np.array(x2_train['x']), np.array(x2_train['y']))
knnResub = Series(knnFit.predict(x2_train['x']),
                  index = x2_train['y'].index)
sum(np.diag(pd.crosstab(knnResub, x2_train['y'])))
x2_test = SimData.simulate2Group(n = 100,
                                 p = 2,
                                 effect = [1.25] * 2)
knnTest = Series(knnFit.predict(x2_test['x']),
                 index = x2_test['y'].index)
sum(np.diag(pd.crosstab(knnTest, x2_test['y'])))


def expandGrid(od):
    cartProd = list(itertools.product(*od.values()))
    return DataFrame(cartProd, columns=od.keys())

parVals = OrderedDict()
parVals['n'] = [100]
parVals['p'] = [2, 5, 10, 25, 100, 500]
parVals['k'] = [3, 5, 10, 25]
parGrid = expandGrid(parVals)
parGrid['effect'] = 2.5
parGrid['effect'] = parGrid['effect'] / np.sqrt(parGrid['p'])


def knnSimulate(param):
    trainSet = SimData.simulate2Group(
        n = int(param['n']),
        p = int(param['p']),
        effect = [param['effect']] * int(param['p'])
    )
    knnFit = KNeighborsClassifier(n_neighbors=int(param['k']))
    knnFit.fit(np.array(trainSet['x']), np.array(trainSet['y']))
    testSet = SimData.simulate2Group(
        n = int(param['n']),
        p = int(param['p']),
        effect = [param['effect']] * int(param['p'])
    )
    out = OrderedDict()
    out['p'] = int(param['p'])
    out['k'] = int(param['k'])
    out['train'] = trainSet
    out['test'] = testSet
    out['resubPreds'] = knnFit.predict(trainSet['x'])
    out['resubProbs'] = knnFit.predict_proba(trainSet['x'])
    out['testPreds'] = knnFit.predict(testSet['x'])
    out['testProbs'] = knnFit.predict_proba(testSet['x'])
    out['resubTable'] = pd.crosstab(
        Series(out['resubPreds'], index=trainSet['y'].index),
        trainSet['y']
    )
    out['resubAccuracy'] = (np.sum(np.diag(out['resubTable'])) /
                            (1.0 * np.sum(np.sum(out['resubTable']))))
    out['testTable'] = pd.crosstab(
        Series(out['testPreds'], index=testSet['y'].index),
        testSet['y']
    )
    out['testAccuracy'] = (np.sum(np.diag(out['testTable'])) /
                           (1.0 * np.sum(np.sum(out['testTable']))))
    return out


knnResults = [knnSimulate(parGrid.ix[i])
              for i in range(parGrid.shape[0])]


repeatedKnnResults = []
for r in range(10):
    repeatedKnnResults.extend(knnSimulate(parGrid.ix[i])
                              for i in range(parGrid.shape[0]))

knnResultsSimplified = DataFrame([(x['p'],
                                   x['k'],
                                   x['resubAccuracy'],
                                   x['testAccuracy'])
                                  for x in repeatedKnnResults],
                                 columns = ['p',
                                            'k',
                                            'resubAccuracy',
                                            'testAccuracy'])

ggdata = pd.concat(
    [DataFrame({'log10(p)' : np.log10(knnResultsSimplified.p),
                'k' : knnResultsSimplified.k.apply(int),
                'type' : 'resub',
                'Accuracy' : knnResultsSimplified.resubAccuracy}),
     DataFrame({'log10(p)' : np.log10(knnResultsSimplified.p),
                'k' : knnResultsSimplified.k.apply(int),
                'type' : 'test',
                'Accuracy' : knnResultsSimplified.testAccuracy})],
    axis = 0
)

plt.clf()
plotIndex = 1
for k in ggdata['k'].unique():
    ax = plt.subplot(int("22"+str(plotIndex)))
    kdata = ggdata.ix[ggdata['k'] == k]
    kdata.ix[kdata.type=="resub"].plot.scatter(x = "log10(p)",
                                               y = "Accuracy",
                                               # label = "resub",
                                               color = (1, 0, 0, 0.7),
                                               edgecolors = "none",
                                               ax = ax)
    kdata.ix[kdata.type=="test"].plot.scatter(x = "log10(p)",
                                              y = "Accuracy",
                                              # label = "test",
                                              color = (0.2, 0.2, 0.2, 0.7),
                                              edgecolors = "none",
                                              ax = ax)
    plt.title(str(k))
    if plotIndex < 3:
        plt.xlabel("")
    if plotIndex in [2, 4]:
        plt.ylabel("")
    plotIndex += 1


# ggobj = ggplot.ggplot(
#     data = ggdata,
#     aesthetics = ggplot.aes(x='log10(p)', y='Accuracy',
#                             color='type', group='type', linetype='type')
# )
# ggobj += ggplot.theme_bw()
# # ggobj += ggplot.scale_x_log()
# ggobj += ggplot.geom_point(alpha=0.6)
# ggobj += ggplot.stat_smooth()
# ggobj += ggplot.facet_wrap('k') 
# print(ggobj)

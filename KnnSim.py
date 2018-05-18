from collections import OrderedDict
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import plotnine as gg
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
np.sum(np.diag(pd.crosstab(knnResub, x2_train['y'])))

x2_test = SimData.simulate2Group(n = 100,
                                 p = 2,
                                 effect = [1.25] * 2)
knnTest = Series(knnFit.predict(x2_test['x']),
                 index = x2_test['y'].index)
np.sum(np.diag(pd.crosstab(knnTest, x2_test['y'])))


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


knnResults = [knnSimulate(parGrid.iloc[i])
              for i in range(parGrid.shape[0])]


repeatedKnnResults = []
for r in range(10):
    repeatedKnnResults.extend(knnSimulate(parGrid.iloc[i])
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
    [DataFrame({'p' : knnResultsSimplified.p,
                'k' : knnResultsSimplified.k.apply(int),
                'type' : 'resub',
                'Accuracy' : knnResultsSimplified.resubAccuracy}),
     DataFrame({'p' : knnResultsSimplified.p,
                'k' : knnResultsSimplified.k.apply(int),
                'type' : 'test',
                'Accuracy' : knnResultsSimplified.testAccuracy})],
    axis = 0
)

plt.close()
ggo = gg.ggplot(ggdata, gg.aes(x='p', y='Accuracy',
                               color='type', group='type', linetype='type'))
ggo += gg.facet_wrap('~ k')
ggo += gg.scale_x_log10()
ggo += gg.geom_point(alpha=0.6)
ggo += gg.stat_smooth()
ggo += gg.theme_bw()
print(ggo)

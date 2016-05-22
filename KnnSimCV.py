import collections
import ggplot
import itertools
import numpy
import pandas
from pandas import DataFrame
from pandas import Series
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import sklearn.cross_validation
from sklearn.cross_validation import cross_val_score

import SimData


x2_train = SimData.simulate2Group(n = 100,
                                  p = 2,
                                  effect = [1.25] * 2)
knnClass = KNeighborsClassifier(n_neighbors=3)
cvAccs = cross_val_score(estimator = knnClass,
                         X = array(x2_train['x']),
                         y = array(x2_train['y']),
                         cv = 5)
cvAccEst = mean(cvAccs)
knnClass.fit(array(x2_train['x']), array(x2_train['y']))
x2_test = SimData.simulate2Group(n = 100,
                                 p = 2,
                                 effect = [1.25] * 2)
knnTest = Series(knnClass.predict(x2_test['x']),
                 index = x2_test['y'].index)
testAccEst = (sum(diag(pandas.crosstab(knnTest, x2_test['y']))) /
              (1.0 * sum(sum(pandas.crosstab(knnTest, x2_test['y'])))))

def expandGrid(od):
    cartProd = list(itertools.product(*od.values()))
    return DataFrame(cartProd, columns=od.keys())

parVals = collections.OrderedDict()
parVals['n'] = [100]
parVals['p'] = [2, 5, 10, 25, 100, 500]
parVals['k'] = [3, 5, 10, 25]
parGrid = expandGrid(parVals)
parGrid['effect'] = 2.5
parGrid['effect'] = parGrid['effect'] / parGrid['p']


def knnSimulate(param, nFold=5):
    trainSet = SimData.simulate2Group(
        n = int(param['n']),
        p = int(param['p']),
        effect = [param['effect']] * int(param['p'])
    )
    knnClass = KNeighborsClassifier(n_neighbors=int(param['k']))
    cvAccs = cross_val_score(estimator = knnClass,
                             X = array(trainSet['x']),
                             y = array(trainSet['y']),
                             cv = nFold)
    knnClass.fit(array(trainSet['x']), array(trainSet['y']))
    testSet = SimData.simulate2Group(
        n = int(param['n']),
        p = int(param['p']),
        effect = [param['effect']] * int(param['p'])
    )
    out = collections.OrderedDict()
    out['p'] = param['p']
    out['k'] = param['k']
    out['train'] = trainSet
    out['test'] = testSet
    out['testPreds'] = knnClass.predict(testSet['x'])
    out['testProbs'] = knnClass.predict_proba(testSet['x'])
    out['cvAccuracy'] = mean(cvAccs)
    out['testTable'] = pandas.crosstab(
        Series(out['testPreds'], index=testSet['y'].index),
        testSet['y']
    )
    out['testAccuracy'] = (sum(diag(out['testTable'])) /
                           (1.0 * sum(sum(out['testTable']))))
    return out


repeatedKnnResults = []
for r in xrange(5):
    repeatedKnnResults.extend(knnSimulate(parGrid.ix[i])
                              for i in xrange(parGrid.shape[0]))
knnResultsSimplified = DataFrame([(x['p'],
                                   x['k'],
                                   x['cvAccuracy'],
                                   x['testAccuracy'])
                                  for x in repeatedKnnResults],
                                 columns = ['p',
                                            'k',
                                            'cvAccuracy',
                                            'testAccuracy'])


ggdata = pandas.concat(
    [DataFrame({'log10(p)' : log10(knnResultsSimplified.p),
                'k' : knnResultsSimplified.k.apply(int),
                'type' : 'cv',
                'Accuracy' : knnResultsSimplified.cvAccuracy}),
     DataFrame({'log10(p)' : log10(knnResultsSimplified.p),
                'k' : knnResultsSimplified.k.apply(int),
                'type' : 'test',
                'Accuracy' : knnResultsSimplified.testAccuracy})],
    axis = 0
)

ggobj = ggplot.ggplot(
    data = ggdata,
    aesthetics = ggplot.aes(x='log10(p)', y='Accuracy',
                            color='type', group='type', linetype='type')
)
ggobj += ggplot.theme_bw()
# ggobj += ggplot.scale_x_log()
ggobj += ggplot.geom_point(alpha=0.6)
ggobj += ggplot.stat_smooth()
ggobj += ggplot.facet_wrap('k') 
print ggobj

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
import sklearn as sk
import sklearn.preprocessing as preprocessing

plt.ion()
plt.style.use('fivethirtyeight')

import MaclearnUtilities
from MaclearnUtilities import bhfdr, colttests, gramSchmidtSelect

import RestrictedData
xs = RestrictedData.xs
xnorms = RestrictedData.xnorms
annots = RestrictedData.annots
ys = RestrictedData.ys
ynums = RestrictedData.ynums


## -----------------------------------------------------------------
## ttest example (using equal variance t test)
## -----------------------------------------------------------------
shengene = xnorms['shen']['NM_008161']
shengene_nervous = shengene[ys['shen']]
shengene_other = shengene[~ys['shen']]
tout = stats.ttest_ind(shengene_nervous.values,
                       shengene_other.values,
                       equal_var = True)

stats.pearsonr(shengene, ynums['shen'])


## -----------------------------------------------------------------
## t tests for all genes in shen set
## -----------------------------------------------------------------
tShenAll = colttests(xnorms['shen'], ynums['shen'])
tShenAll['q'] = bhfdr(tShenAll.p)
## let's try something else...
xscShen = preprocessing.scale(xnorms['shen'])
xscShen = pd.DataFrame(xscShen,
                      index = xnorms['shen'].index,
                      columns = xnorms['shen'].columns)
xscShen.mean(axis=0)
xscShen.std(axis=0)

yscShen = preprocessing.scale(ynums['shen'].astype('float'))
tShenAll['pearson'] = np.dot(yscShen, xscShen) / len(yscShen)
## sort by p
tShenAll.sort_values('p', inplace=True)


## -----------------------------------------------------------------
## t tests for all genes in each set
## -----------------------------------------------------------------
def tTestPlus(x, y):
    out = colttests(x, y)
    out['q'] = bhfdr(out['p'])
    out['pearson'] = np.dot(
        preprocessing.scale(y.astype('float')),
        preprocessing.scale(x)
    ) / len(y)
    out.sort_values('p', inplace=True)
    return out

tTestResults = {k : tTestPlus(xnorms[k], ynums[k]) for k in xnorms}


## -----------------------------------------------------------------
## generate fancy p vs pearson plot
## -----------------------------------------------------------------
plt.clf()
ax = plt.subplot(111)
colors = {
    "montastier" : "black",
    "patel" : "darkgray",
    "hess" : "red",
    "shen" : "darkred"
}
for s in tTestResults:
    plotdata = pd.DataFrame({'gene' : tTestResults[s].index,
                             'set' : s + " (" + str(xnorms[s].shape[0]) + ")",
                             'p' : tTestResults[s].p,
                             'pearson' : tTestResults[s].pearson})
    plotdata.sort_values("pearson", inplace=True)
    plotdata.plot(x = "pearson",
                  y = "p",
                  color = colors[s],
                  logy = True,
                  label = s + " (" + str(xnorms[s].shape[0]) + ")",
                  ax = ax)


compResults = gramSchmidtSelect(x = xnorms['patel'],
                                y = ynums['patel'],
                                g = 'NAMPT')
compResults = compResults.loc[
        compResults.abs().sort_values(ascending=False).index]


import matplotlib.pyplot as plt
import numpy
import pandas
import scipy
import scipy.stats
import sklearn
import sklearn.cross_validation
import sklearn.feature_selection
import sklearn.neighbors
import sklearn.pipeline

import MaclearnUtilities
from MaclearnUtilities import bhfdr
from MaclearnUtilities import colttests
from MaclearnUtilities import gramSchmidtSelect

import RestrictedData
xs = RestrictedData.xs
xnorms = RestrictedData.xnorms
annots = RestrictedData.annots
ys = RestrictedData.ys
ynums = RestrictedData.ynums


## -----------------------------------------------------------------
## ttest example (using equal variance t test)
## -----------------------------------------------------------------
botgene = xnorms['bottomly']['ENSMUSG00000027855']
botgene_C57BL = botgene[ys['bottomly'] == 'C57BL/6J']
botgene_DBA = botgene[ys['bottomly'] == 'DBA/2J']
tout = scipy.stats.ttest_ind(array(botgene_C57BL),
                             array(botgene_DBA),
                             equal_var = True)

scipy.stats.pearsonr(botgene, ynums['bottomly'])


## -----------------------------------------------------------------
## t tests for all genes in bottomly set
## -----------------------------------------------------------------
tBotAll = colttests(xnorms['bottomly'], ynums['bottomly'])
tBotAll['q'] = bhfdr(tBotAll.p)
## let's try something else...
xscBot = sklearn.preprocessing.scale(xnorms['bottomly'])
xscBot = pandas.DataFrame(xscBot,
                          index = xnorms['bottomly'].index,
                          columns = xnorms['bottomly'].columns)
xscBot.mean(axis=0)
xscBot.std(axis=0)

yscBot = sklearn.preprocessing.scale(ynums['bottomly'].astype('float'))
tBotAll['pearson'] = numpy.dot(yscBot, xscBot) / len(yscBot)
## sort by p
tBotAll.sort_values('p', inplace=True)


## -----------------------------------------------------------------
## t tests for all genes in each set
## -----------------------------------------------------------------
def tTestPlus(x, y):
    out = colttests(x, y)
    out['q'] = bhfdr(out['p'])
    out['pearson'] = numpy.dot(
        sklearn.preprocessing.scale(y.astype('float')),
        sklearn.preprocessing.scale(x)
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
    "bottomly" : "darkred"
}
for s in tTestResults:
    plotdata = pandas.DataFrame({'gene' : tTestResults[s].index,
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
compResults = compResults.ix[
        compResults.abs().sort_values(ascending=False).index]


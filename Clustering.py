import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import scipy
import scipy.cluster
import seaborn as sns
import sklearn
import sklearn.cluster

import SimData

plt.ion()
plt.style.use("fivethirtyeight")

simDat = SimData.simulate2Group(n=40, p=2, effect=[1, 0.75])
xsim = simDat['x']
ysim = simDat['y']


## -----------------------------------------------------------------
## k-means clustering
## -----------------------------------------------------------------
k2Clusterer = sklearn.cluster.KMeans(n_clusters=2)
kmSim = k2Clusterer.fit(xsim)
kmSimClusts = kmSim.predict(xsim)

def kmplot(xy):
    x = xy['x']
    y = xy['y']
    km = sklearn.cluster.KMeans(n_clusters=2).fit(x)
    plotdata = x.copy()
    clust = km.predict(x)
    if sum((clust-0.5) * (y-0.5)) < 0:
        clust = 1 - clust
    isright = np.sign((clust-0.5) * (y-0.5))
    plotdata['group'] = (1-isright) + y
    ax = plotdata.ix[plotdata["group"]==0].plot.scatter(
            x="g0", y="g1", color="black")
    plotdata.ix[plotdata["group"]==1].plot.scatter(
            x="g0", y="g1", color="goldenrod", ax=ax)
    if len(plotdata["group"].unique()) > 2:
        plotdata.ix[plotdata["group"]==2].plot.scatter(
                x="g0", y="g1", color="lightgray", ax=ax)
        plotdata.ix[plotdata["group"]==3].plot.scatter(
                x="g0", y="g1", color="red", ax=ax)
    return(plotdata)

plt.close()
kmplot(SimData.simulate2Group(n=40, p=2, effect=[10, 0]))

plt.close()
kmplot(SimData.simulate2Group(n=40, p=2, effect=[1, 0.75]))


## -----------------------------------------------------------------
## hierarchical clustering
## -----------------------------------------------------------------
simData2 = SimData.simulate2Group(n=40, p=20, effect=[2, 1, 1])
xsim2 = simData2['x']
ysim2 = simData2['y']

plt.close()
xdist = scipy.spatial.distance.pdist(xsim, metric="euclidean")
ihcSim = scipy.cluster.hierarchy.average(xdist)
idendrout = scipy.cluster.hierarchy.dendrogram(ihcSim,
                                               orientation = "right")

plt.close()
gdist = scipy.spatial.distance.pdist(xsim2.transpose(), metric="euclidean")
ghcSim = scipy.cluster.hierarchy.average(gdist)
gdendrout = scipy.cluster.hierarchy.dendrogram(ghcSim,
                                               orientation = "right")


## -----------------------------------------------------------------
## clustered heatmap
## -----------------------------------------------------------------
heatColors = pd.Series(['#000000']*xsim2.shape[0], index=xsim2.index)
heatColors.ix[ysim2 == 1] = '#FF0066'
plt.close()
sns.clustermap(xsim2.transpose(), method='complete', col_colors=heatColors)


## HERE

## -----------------------------------------------------------------
## on real data...
## -----------------------------------------------------------------
import RestrictedData
xs = RestrictedData.xs
xnorms = RestrictedData.xnorms
annots = RestrictedData.annots
ys = RestrictedData.ys
ynums = RestrictedData.ynums

shenHighVar = xnorms['shen'].columns[xnorms['shen'].std() > 2]
heatX = xnorms['shen'][shenHighVar].transpose()
## remove overall gene-means from data for more useful plot
heatX = heatX.subtract(heatX.mean(axis=1), axis=0)
## pay attention to changes around mean, not far from it
maxLogFoldChange = 2.5
heatX[heatX > maxLogFoldChange] = maxLogFoldChange
heatX[heatX < -maxLogFoldChange] = -maxLogFoldChange

# heatColors = pd.Series(['#000000']*heatX.shape[1], index=heatX.columns)
# heatColors.ix[ys['shen'] == 'DBA/2J'] = '#FF0066'
# heatColors = pd.DataFrame({'Mouse Strain' : heatColors})

heatColors = pd.Series({
    'circulatory' : 'firebrick',
    'digestive/excretory' : 'goldenrod',
    'lymphatic' : 'lightseagreen',
    'nervous' : 'darkorchid',
    'other' : 'darkslategray',
    'respiratory' : 'dodgerblue'
}).reindex(annots['shen']['System'].values)
heatColors.index = annots['shen'].index
heatColors = pd.DataFrame({'System' : heatColors})
heatColors = heatColors.reindex(heatX.columns)

plt.close()
cm = sns.clustermap(heatX, method='complete', col_colors=heatColors, figsize=(10, 10))
garbage = plt.setp(cm.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
garbage = plt.setp(cm.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)

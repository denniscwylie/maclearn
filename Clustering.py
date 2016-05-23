import numpy
import pandas
from pandas import DataFrame
from pandas import Series
import scipy
import scipy.cluster
import sklearn
import sklearn.cluster

import SimData

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
    isright = numpy.sign((clust-0.5) * (y-0.5))
    plotdata['group'] = (1-isright) + y
    ax = plotdata.ix[plotdata["group"]==0].plot.scatter(
            x="g0", y="g1", color="black")
    plotdata.ix[plotdata["group"]==1].plot.scatter(
            x="g0", y="g1", color="gold", ax=ax)
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


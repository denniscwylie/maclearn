import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

import MaclearnUtilities
from MaclearnUtilities import safeFactorize

plt.ion()
plt.style.use("fivethirtyeight")

import RestrictedData
xs = RestrictedData.xs
xnorms = RestrictedData.xnorms
annots = RestrictedData.annots
ys = RestrictedData.ys
ynums = RestrictedData.ynums


def svdForPca(x, center="both", scale="none"):
    if min(x.std(axis=0)) == 0:
        return None
    xhere = x.copy()
    if center in ['row', 'both']:
        xRowAvs = xhere.mean(axis=1)
        xhere = xhere.add(-xRowAvs, axis=0)
    if center in ['col', 'both']:
        xColAvs = xhere.mean(axis=0)
        xhere = xhere.add(-xColAvs, axis=1)
    if scale == 'row':
        rowSds = xhere.std(axis=1)
        xhere = xhere.divide(rowSds, axis=0)
    elif scale == 'col':
        colSds = xhere.std(axis=0)
        xhere = xhere.divide(colSds, axis=1)
    xsvd = np.linalg.svd(xhere, full_matrices=False)
    return xsvd


def pca(x, y=None, ylev=None,
        nlab=0, lsize=10, lalpha=1,
        center="both", scale="none",
        legend=True, cname="variable",
        color=None):
    if type(color) != type({}):
        color = None
    xForSvd = x.ix[:, x.std(axis=0) > 0]
    xsvd = svdForPca(xForSvd, center, scale)
    svdRowPlot = DataFrame(
        xsvd[0][:, 0:2],
        index = xForSvd.index,
        columns = ["PC1", "PC2"]
    )
    svdRowPlot = svdRowPlot.divide(svdRowPlot.max(axis=0) -
                                   svdRowPlot.min(axis=0), axis=1)
    svdColPlot = DataFrame(
        np.transpose(xsvd[2][0:2, :]),
        index = xForSvd.columns,
        columns = ["PC1", "PC2"]
    )
    svdColPlot = svdColPlot.divide(svdColPlot.max(axis=0) -
                                   svdColPlot.min(axis=0), axis=1)
    if nlab > 0:
        svdColPlotMag = (svdColPlot**2).sum(axis=1)
        svdColPlotMag.sort_values(ascending=False, inplace=True)
        svdColPlot = svdColPlot.ix[svdColPlotMag.index]
        svdColPlot["label"] = ""
        svdColPlot.ix[0:nlab, "label"] = \
                svdColPlot.ix[0:nlab].index.to_series()
    if legend:
        ax = plt.subplot(111)
    plt.plot(svdColPlot["PC1"], svdColPlot["PC2"],
             "o", color=(0, 0, 0, 0.1), markersize=5,
             label=cname)
    if nlab > 0:
        for i in range(nlab):
            plt.text(svdColPlot.ix[i, "PC1"],
                     svdColPlot.ix[i, "PC2"],
                     svdColPlot.ix[i, "label"],
                     fontsize = lsize,
                     color = (0, 0, 0, lalpha),
                     label = None)
    if y is not None:
        if ylev is None:
            ylev = y.unique()
        for level in ylev:
            if color is not None and level in color.keys():
                plt.plot(svdRowPlot.ix[y == level, 0],
                         svdRowPlot.ix[y == level, 1],
                         "o",
                         markersize = 8,
                         label = level,
                         color = color[level])
            else:
                plt.plot(svdRowPlot.ix[y == level, 0],
                         svdRowPlot.ix[y == level, 1],
                         "o",
                         markersize = 8,
                         label = level)
    else:
        plt.plot(svdRowPlot["PC1"], svdRowPlot["PC2"],
                 "o", markersize=8)
    if legend:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), numpoints=1)
    plt.show()


plt.close()
pca(xnorms['bottomly'], ys['bottomly'], nlab=10)

plt.close()
pca(xnorms['patel'], ys['patel'], nlab=10)

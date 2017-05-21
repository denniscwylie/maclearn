import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series

import LoadData

xs = LoadData.xs
annots = LoadData.annots


def uqnormalize(x, axis=0, scale=100):
    geneDetected = (x.sum(axis=axis) > 0)
    if axis == 0:
        xgd = x[ x.columns[geneDetected] ]
    elif axis == 1:
        xgd = x.ix[geneDetected]
    normfacs = np.percentile(xgd, q=75, axis=1-axis)
    return scale * x.divide(normfacs, axis=axis)

xnorms = {}
xnorms['bottomly'] = np.log2(uqnormalize(xs['bottomly']) + 1)

## patel set already normalized
xnorms['patel'] = xs['patel'].copy()

def meanCenter(x, axis=0):
    geneHasNans = (np.isnan(x).sum(axis=axis) > 0)
    if axis == 0:
        xnonans = x[ x.columns[~geneHasNans] ]
    elif axis == 1:
        xnonans = x.ix[~geneHasNans]
    means = xnonans.mean(axis=1-axis)
    return x.add(-means, axis=axis)

def meanCenterAndImpute(x, axis=0, imputeAt=None):
    if imputeAt is None:
        imputeAt = np.ceil(x.max().max())
    geneHasNans = (np.isnan(x).sum(axis=axis) > 0)
    if axis == 0:
        xnonans = x[ x.columns[~geneHasNans] ]
    elif axis == 1:
        xnonans = x.ix[~geneHasNans]
    means = xnonans.mean(axis=1-axis)
    out = x.copy()
    out[np.isnan(out)] = imputeAt
    return out.add(-means, axis=axis)

xnorms['montastier'] = meanCenterAndImpute(xs['montastier'])

## hess set already normalized
xnorms['hess'] = xs['hess']

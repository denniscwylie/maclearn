import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series

import LoadData

xs = LoadData.xs
annots = LoadData.annots


def rleSizeFactors(x):
    xno0 = x.loc[:, x.min(axis=0) > 0]
    geoMeans = np.exp(np.log(xno0).mean(axis=0))
    sizeFactors = xno0.divide(geoMeans, axis=1).median(axis=1)
    return sizeFactors

xnorms = {}

## shen set already normalized
xnorms['shen'] = xs['shen'].copy()

## patel set already normalized
xnorms['patel'] = xs['patel'].copy()

def meanCenter(x, axis=0):
    geneHasNans = (np.isnan(x).sum(axis=axis) > 0)
    if axis == 0:
        xnonans = x[ x.columns[~geneHasNans] ]
    elif axis == 1:
        xnonans = x.loc[~geneHasNans]
    means = xnonans.mean(axis=1-axis)
    return x.add(-means, axis=axis)

def meanCenterAndImpute(x, axis=0, imputeAt=None):
    if imputeAt is None:
        imputeAt = np.ceil(x.max().max())
    geneHasNans = (np.isnan(x).sum(axis=axis) > 0)
    if axis == 0:
        xnonans = x[ x.columns[~geneHasNans] ]
    elif axis == 1:
        xnonans = x.loc[~geneHasNans]
    means = xnonans.mean(axis=1-axis)
    out = x.copy()
    out[np.isnan(out)] = imputeAt
    return out.add(-means, axis=axis)

xnorms['montastier'] = meanCenterAndImpute(xs['montastier'])

## hess set already normalized
xnorms['hess'] = xs['hess'].copy()

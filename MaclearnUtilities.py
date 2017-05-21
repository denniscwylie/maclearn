from collections import OrderedDict
import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize as optimize
import scipy.special as special
import scipy.stats as stats
import sklearn as sk
import sklearn.preprocessing as preprocessing


def safeFactorize(series):
    if "factorize" in dir(series):
        return series.factorize()[0]
    else:
        uniqSer = series.unique()
        out = pd.Series(np.zeros(len(series)))
        out.index = series.index
        for i in range(1, len(uniqSer)):
            out.ix[series == uniqSer[i]] = i
        return out


def colcor(x, y, **kwargs):
    xmeans = x.mean(axis=0)
    dx = x.add(-xmeans, axis=1)
    dy = y - np.mean(y)
    dxnorms = np.sqrt((dx*dx).mean(axis=0))
    ynorm = np.sqrt(sum(dy*dy))
    rhos = np.dot(dy, dx) / ynorm
    rhos = rhos / dxnorms
    tgs = np.sign(rhos) * (np.sqrt(len(y) - 2) *
                           np.sqrt(rhos**2 / (1 - rhos**2)))
    out = OrderedDict()
    out['t'] = tgs
    out['p'] = 2 * stats.t.cdf(-abs(out['t']), len(y)-2.0)
    return pd.DataFrame(out, index=x.columns)


def colttests(x, y, **kwargs):
    y0mean = np.dot(1 - y, x) / sum(1 - y)
    y1mean = np.dot(y, x) / sum(y)
    y0sse = np.dot(1 - y, x**2) - (sum(1 - y) * (y0mean**2))
    y1sse = np.dot(y, x**2) - (sum(y) * (y1mean**2))
    tnumerator = y0mean - y1mean
    tdenominator = np.sqrt((y0sse + y1sse) / (len(y) - 2))
    tdenominator *= np.sqrt((1.0/sum(1-y)) + (1.0/sum(y)))
    out = OrderedDict()
    out['t'] = tnumerator / tdenominator
    out['p'] = 2 * stats.t.cdf(-abs(out['t']), len(y)-2.0)
    return pd.DataFrame(out, index=x.columns)


def bhfdr(pvals):
    p = len(pvals) - sum(np.isnan(pvals))
    psort = pvals.sort_values(ascending = False,
                              inplace = False,
                              na_position = 'first')
    qvals = psort * (p / (1.0 * np.arange(len(pvals), 0, -1)))
    qvals = qvals.cummin()
    qvals.ix[qvals > 1] = 1
    return qvals.ix[pvals.index]


def ebayes(x, y):
    p = 1.0 * x.shape[1]
    dg = 1.0 * (x.shape[0] - 2)
    y0mean = np.dot(1 - y, x) / sum(1 - y)
    y1mean = np.dot(y, x) / sum(y)
    y0sse = np.dot(1 - y, x**2) - (sum(1 - y) * (y0mean**2))
    y1sse = np.dot(y, x**2) - (sum(y) * (y1mean**2))
    sg2 = (y0sse + y1sse) / (len(y) - 2)
    vgj = (1.0/sum(1-y)) + (1.0/sum(y))
    zg = log(sg2)
    eg = zg - special.digamma(dg/2) + log(dg/2)
    ebar = np.mean(eg)
    rhs1 = np.mean((((eg-ebar)**2) * p / (p-1)) -
                   special.polygamma(1, dg/2))
    def objective(df):
        return special.polygamma(1, df/2.0) - rhs1
    d0 = optimize.brentq(objective, 0.0, 10.0*x.shape[0])
    s0 = np.sqrt(np.exp(ebar
                        + special.digamma(d0/2)
                        - np.log(d0/2)))
    stilde = np.sqrt( ((d0*(s0**2)) + (dg*(sg2)))
                         / (d0 + dg) )
    tmod = (y1mean - y0mean) / (stilde * sqrt(vgj))
    out = pd.DataFrame({'t' : tmod}, index=x.columns)
    out['p'] = 2 * stats.t.cdf(-abs(tmod), dg + d0)
    return out
    
    
def gramSchmidtSelect(x, y, g=[]):
    if isinstance(g, str) or "__len__" not in dir(g):
        g = [g]
    xmeans = x.mean(axis=0)
    dx = x.add(-xmeans, axis=1)
    dy = y - y.mean()
    pgtotal = np.eye(x.shape[0])
    for gel in g:
        dxg = preprocessing.scale(np.dot(pgtotal, dx[gel]))
        pg = np.eye(x.shape[0]) - (np.outer(dxg, dxg) / sum(dxg**2))
        pgtotal = np.dot(pg, pgtotal)
    pgdx = np.dot(pgtotal, dx)
    pgdy = np.dot(pgtotal, dy)
    compCors = (np.dot(preprocessing.scale(pgdy),
                       preprocessing.scale(pgdx)) /
                len(y))
    return pd.Series(compCors, index=x.columns)

    

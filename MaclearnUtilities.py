from collections import OrderedDict
import numpy
import pandas
import scipy
import scipy.optimize
import scipy.special
import scipy.stats
import sklearn
import sklearn.preprocessing

def safeFactorize(series):
    if "factorize" in dir(series):
        return series.factorize()[0]
    else:
        uniqSer = series.unique()
        out = pandas.Series(numpy.zeros(len(series)))
        out.index = series.index
        for i in range(1, len(uniqSer)):
            out.ix[series == uniqSer[i]] = i
        return out


def colcor(x, y, **kwargs):
    xmeans = x.mean(axis=0)
    dx = x.add(-xmeans, axis=1)
    dy = y - numpy.mean(y)
    dxnorms = numpy.sqrt((dx*dx).mean(axis=0))
    ynorm = numpy.sqrt(sum(dy*dy))
    rhos = numpy.dot(dy, dx) / ynorm
    rhos = rhos / dxnorms
    tgs = numpy.sign(rhos) * (numpy.sqrt(len(y) - 2) *
                              numpy.sqrt(rhos**2 / (1 - rhos**2)))
    out = OrderedDict()
    out['t'] = tgs
    out['p'] = 2 * scipy.stats.t.cdf(-abs(out['t']), len(y)-2.0)
    return pandas.DataFrame(out, index=x.columns)


def colttests(x, y, **kwargs):
    y0mean = numpy.dot(1 - y, x) / sum(1 - y)
    y1mean = numpy.dot(y, x) / sum(y)
    y0sse = numpy.dot(1 - y, x**2) - (sum(1 - y) * (y0mean**2))
    y1sse = numpy.dot(y, x**2) - (sum(y) * (y1mean**2))
    tnumerator = y0mean - y1mean
    tdenominator = numpy.sqrt((y0sse + y1sse) / (len(y) - 2))
    tdenominator *= numpy.sqrt((1.0/sum(1-y)) + (1.0/sum(y)))
    out = OrderedDict()
    out['t'] = tnumerator / tdenominator
    out['p'] = 2 * scipy.stats.t.cdf(-abs(out['t']), len(y)-2.0)
    return pandas.DataFrame(out, index=x.columns)


def bhfdr(pvals):
    p = len(pvals) - sum(numpy.isnan(pvals))
    psort = pvals.sort_values(ascending = False,
                              inplace = False,
                              na_position = 'first')
    qvals = psort * (p / (1.0 * numpy.arange(len(pvals), 0, -1)))
    qvals = qvals.cummin()
    qvals.ix[qvals > 1] = 1
    return qvals.ix[pvals.index]


def ebayes(x, y):
    p = 1.0 * x.shape[1]
    dg = 1.0 * (x.shape[0] - 2)
    y0mean = numpy.dot(1 - y, x) / sum(1 - y)
    y1mean = numpy.dot(y, x) / sum(y)
    y0sse = numpy.dot(1 - y, x**2) - (sum(1 - y) * (y0mean**2))
    y1sse = numpy.dot(y, x**2) - (sum(y) * (y1mean**2))
    sg2 = (y0sse + y1sse) / (len(y) - 2)
    vgj = (1.0/sum(1-y)) + (1.0/sum(y))
    zg = log(sg2)
    eg = zg - scipy.special.digamma(dg/2) + log(dg/2)
    ebar = numpy.mean(eg)
    rhs1 = numpy.mean((((eg-ebar)**2) * p / (p-1)) -
                      scipy.special.polygamma(1, dg/2))
    def objective(df):
        return scipy.special.polygamma(1, df/2.0) - rhs1
    d0 = scipy.optimize.brentq(objective, 0.0, 10.0*x.shape[0])
    s0 = numpy.sqrt(numpy.exp(ebar
                              + scipy.special.digamma(d0/2)
                              - numpy.log(d0/2)))
    stilde = numpy.sqrt( ((d0*(s0**2)) + (dg*(sg2)))
                         / (d0 + dg) )
    tmod = (y1mean - y0mean) / (stilde * sqrt(vgj))
    out = pandas.DataFrame({'t' : tmod}, index=x.columns)
    out['p'] = 2 * scipy.stats.t.cdf(-abs(tmod), dg + d0)
    return out
    
    
def gramSchmidtSelect(x, y, g=[]):
    if isinstance(g, basestring) or "__len__" not in dir(g):
        g = [g]
    xmeans = x.mean(axis=0)
    dx = x.add(-xmeans, axis=1)
    dy = y - y.mean()
    pgtotal = numpy.eye(x.shape[0])
    for gel in g:
        dxg = sklearn.preprocessing.scale(numpy.dot(pgtotal, dx[gel]))
        pg = numpy.eye(x.shape[0]) - (numpy.outer(dxg, dxg) / sum(dxg**2))
        pgtotal = numpy.dot(pg, pgtotal)
    pgdx = numpy.dot(pgtotal, dx)
    pgdy = numpy.dot(pgtotal, dy)
    compCors = (numpy.dot(sklearn.preprocessing.scale(pgdy),
                          sklearn.preprocessing.scale(pgdx)) /
                len(y))
    return pandas.Series(compCors, index=x.columns)

    

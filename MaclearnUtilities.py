from collections import OrderedDict
import numpy as np
import pandas as pd
import plotnine as gg
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
            out.loc[series == uniqSer[i]] = i
        return out

    
def colcor(x, y, **kwargs):
    if isinstance(y, pd.Series) and np.any(x.index != y.index):
        y = y.copy().loc[x.index]
    xmeans = x.mean(axis=0)
    dx = x.add(-xmeans, axis=1)
    dy = y - np.mean(y)
    dxnorms = np.sqrt((dx*dx).sum(axis=0))
    ynorm = np.sqrt(np.sum(dy*dy))
    rhos = np.dot(dy, dx) / ynorm
    rhos = rhos / dxnorms
    tgs = np.sign(rhos) * (np.sqrt(len(y) - 2) *
                           np.sqrt(rhos**2 / (1 - rhos**2)))
    out = OrderedDict()
    out['rho'] = rhos
    out['t'] = tgs
    out['p'] = 2 * stats.t.cdf(-abs(out['t']), len(y)-2.0)
    return pd.DataFrame(out, index=x.columns)
    

def colttests(x, y, **kwargs):
    if isinstance(y, pd.Series) and np.any(x.index != y.index):
        y = y.copy().loc[x.index]
    y0mean = np.dot(1 - y, x) / np.sum(1 - y)
    y1mean = np.dot(y, x) / np.sum(y)
    y0sse = np.dot(1 - y, x**2) - (np.sum(1 - y) * (y0mean**2))
    y1sse = np.dot(y, x**2) - (np.sum(y) * (y1mean**2))
    tnumerator = y0mean - y1mean
    tdenominator = np.sqrt((y0sse + y1sse) / (len(y) - 2))
    tdenominator *= np.sqrt((1.0/np.sum(1-y)) + (1.0/np.sum(y)))
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
    qvals.loc[qvals > 1] = 1
    return qvals.loc[pvals.index]


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

    
def svdForPca(x, center="col", scale="none", pandaize=True):
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
    if pandaize:
        xsvd = (
            pd.DataFrame(xsvd[0], index=x.index),
            pd.Series(xsvd[1]),
            pd.DataFrame(xsvd[2], columns=x.columns)
        )
    return xsvd


def ggpca(x, y=None, center='col', scale='none',
          rlab=False, clab=False, cshow=None,
          rsize=4, csize=2, lsize=10, lnudge=0.03,
          ralpha=0.6, calpha=1.0, clightalpha=0,
          rname='sample', cname='variable', lname='',
          grid=True, printit=False, xsvd=None,
          invert1=False, invert2=False, colscale=None,
          **kwargs):
    if cshow is None:
        cshow = x.shape[1]
    if rlab is not None and isinstance(rlab, bool):
        rlab = x.index if rlab else ''
    if clab is not None and isinstance(clab, bool):
        clab = x.columns if clab else ''
    if y is not None:
        pass
    x = x.loc[:, x.isnull().sum(axis=0) == 0]
    if xsvd is None:
        xsvd = svdForPca(x, center, scale)
    rsf = np.max(xsvd[0].iloc[:, 0]) - np.min(xsvd[0].iloc[:, 0])
    csf = np.max(xsvd[2].iloc[0, :]) - np.min(xsvd[2].iloc[0, :])
    sizeRange = sorted([csize, rsize])
    alphaRange = sorted([calpha, ralpha])
    ggd = pd.DataFrame({
        'PC1' : xsvd[0].iloc[:, 0] / rsf,
        'PC2' : xsvd[0].iloc[:, 1] / rsf,
        'label' : rlab,
        'size' : rsize,
        'alpha' : ralpha
    })
    cclass = []
    if cshow > 0:
        cdata = pd.DataFrame({
            'PC1' : xsvd[2].iloc[0, :] / csf,
            'PC2' : xsvd[2].iloc[1, :] / csf,
            'label' : clab,
            'size' : csize,
            'alpha' : calpha
        })
        if cshow < x.shape[1]:
            cscores = cdata['PC1']**2 + cdata['PC2']**2
            keep = cscores.sort_values(ascending=False).head(cshow).index
            if clightalpha > 0:
                cdata.loc[~cdata.index.isin(keep), 'label'] = ''
                cdata.loc[~cdata.index.isin(keep), 'alpha'] = clightalpha
                alphaRange = [np.min([alphaRange[0], clightalpha]),
                              np.max([alphaRange[1], clightalpha])]
            else:
                cdata = cdata.loc[cdata.index.isin(keep)]
        ggd = pd.concat([cdata, ggd])
        cclass = [cname] * cdata.shape[0]
    if invert1:
        ggd['PC1'] = -ggd['PC1']
    if invert2:
        ggd['PC2'] = -ggd['PC2']
    if y is not None:
        ggd['class'] = cclass + list(y.loc[x.index])
    else:
        ggd['class'] = cclass + ([rname] *  x.shape[0])
    ggo = gg.ggplot(ggd, gg.aes(
        x = 'PC1',
        y = 'PC2',
        color = 'class',
        size = 'size',
        alpha = 'alpha',
        label = 'label'
    ))
    ggo += gg.geom_hline(yintercept=0, color='gray')
    ggo += gg.geom_vline(xintercept=0, color='gray')
    ggo += gg.geom_point()
    ggo += gg.theme_bw()
    ggo += gg.geom_text(nudge_y=lnudge, size=lsize, show_legend=False)
    if colscale is None and len(ggd['class'].unique()) < 8:
        colscale = ['darkslategray', 'goldenrod', 'lightseagreen',
                    'orangered', 'dodgerblue', 'darkorchid']
        colscale = colscale[0:(len(ggd['class'].unique())-1)] + ['gray']
        if len(colscale) == 2 and cshow > 0:
            colscale = ['black', 'darkgray']
        if len(colscale) == 2 and cshow == 0:
            colscale = ['black', 'red']
        if len(colscale) == 3:
            colscale = ['black', 'red', 'darkgray']
    ggo += gg.scale_color_manual(values=colscale, name=lname)
    ggo += gg.scale_size_continuous(guide=False, range=sizeRange)
    ggo += gg.scale_alpha_continuous(guide=False, range=alphaRange)
    ggo += gg.xlab('PC1 (' +
                   str(np.round(100*xsvd[1][0]**2 / ((xsvd[1]**2).sum()), 1)) +
                   '% explained var.)')
    ggo += gg.ylab('PC2 (' +
                   str(np.round(100*xsvd[1][1]**2 / ((xsvd[1]**2).sum()), 1)) +
                   '% explained var.)')
    if not grid:
        ggo += gg.theme(panel_grid_minor = gg.element_blank(),
                        panel_grid_major = gg.element_blank(),
                        panel_background = gg.element_blank())
    ggo += gg.theme(axis_ticks = gg.element_blank(),
                    axis_text_x = gg.element_blank(),
                    axis_text_y = gg.element_blank())
    if printit:
        print(ggo)
    return ggo

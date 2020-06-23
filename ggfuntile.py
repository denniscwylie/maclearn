from collections import OrderedDict
import numpy as np
import pandas as pd
import plotnine
from plotnine import ggplot, aes, geom_tile, scale_fill_gradientn, xlab, ylab, geom_point, ggtitle, theme, theme_classic, scale_shape_manual

from maclearn_utils_2020 import expandGrid

def ggfuntile(f, d,
              xrng=(0, 1), yrng=(0, 1), limits=(0, 1),
              density=51,
              xlab="x", ylab="y", zlab="f",
              breaks=None, **kwargs):
    od = OrderedDict()
    od[xlab] = np.arange(xrng[0], xrng[1],
                        (xrng[1]-xrng[0]) / (density-1.0))
    od[ylab] = np.arange(yrng[0], yrng[1],
                        (yrng[1]-yrng[0]) / (density-1.0))
    ggdata = expandGrid(od)
    ggdata["z"] = [f(ggdata.iloc[i, 0], ggdata.iloc[i, 1])
                   for i in range(ggdata.shape[0])]
    gg = ggplot(ggdata, aes(x=xlab, y=ylab))
    gg += geom_tile(aes(fill="z"))
    gg += scale_fill_gradientn(
        colors = ["black", "#202020", "#404040", "#808080", "white",
                  "dodgerblue", "blue", "darkblue", "midnightblue"],
        name = zlab,
        limits = limits
    )
    gg += theme_classic()
    gg += geom_point(
        data = d,
        mapping = aes(shape="class"),
        color="red", size=2, alpha=0.8
    )
    gg += scale_shape_manual(values=["x", "^"])
    return gg

def predictionContour(fit, data, y, title, density=51):
    data = data.copy()
    y = y.copy().astype(str)
    def predictor(g, h):
        dfgh = pd.DataFrame({data.columns[0] : [g]})
        dfgh[data.columns[1]] = [h]
        return fit.predict_proba(dfgh)[0, 1]
    data["class"] = y
    xrng = (0.5 * np.floor(2.0*min(data.iloc[:, 0])),
            0.5 * np.ceil(2.0*max(data.iloc[:, 0])))
    yrng = (0.5 * np.floor(2.0*min(data.iloc[:, 1])),
            0.5 * np.ceil(2.0*max(data.iloc[:, 1])))
    out =  ggfuntile(predictor, data,
                     xrng=xrng, yrng=yrng, density=density,
                     xlab=data.columns[0], ylab=data.columns[1],
                     zlab="P(Y=1)", breaks=[-np.inf, 0.5, np.inf])
    out += ggtitle(title)
    return out
        
        

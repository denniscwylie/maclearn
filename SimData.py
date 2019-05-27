import numpy
import numpy.random
import pandas
from pandas import DataFrame
from pandas import Series

def simulate2Group(n=100, p=1000, n1=None, effect=None):
    if n1 is None:
        n1 = int(numpy.ceil(0.5 * n))
    if effect is None:
        effect = [1] * 10
    x = DataFrame(numpy.random.randn(n, p))
    y = Series(([0] * n1) + ([1] * (n-n1)))
    x.columns = ["g"+str(g) for g in range(p)]
    x.index = ["i"+str(i) for i in range(n)]
    y.index = x.index
    for i in range(len(effect)):
        x.loc[y==1, x.columns[i]] = x.loc[y==1, x.columns[i]] + effect[i]
    return {"x":x, "y":y}


    
    
    
    

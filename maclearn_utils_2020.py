import itertools
import numpy as np
import pandas as pd
import sklearn as sk

def expandGrid(od):
    cartProd = list(itertools.product(*od.values()))
    return pd.DataFrame(cartProd, columns=od.keys())

def extractPCs(mat, m=None, *args):
    ## assume x is samples-in-rows, genes-in-columns format!
    if m is None:
        m = np.min(mat.shape)    
    mu = mat.mean(axis=0)  ## training-set estimated mean expression per gene
    ## use singular value decomposition (SVD) to compute PCs:
    svdOut = np.linalg.svd(mat - mu, full_matrices=False)
    x = svdOut[0] * svdOut[1]  ## same as R's prcomp out$x
    rotation = svdOut[2].T     ## same as R's prcomp out$rotation
    sdev = svdOut[1] / np.sqrt(len(svdOut[1])-1)  ## same as R's prcomp out$sdev
    extractor = lambda newdata : np.dot(newdata-mu, rotation[:, 0:m])
    extractor.sdev = sdev
    extractor.rotation = rotation    
    extractor.center = mu
    extractor.x = x
    extractor.m = m
    ## return the function "extractor" which can be applied to newdata;
    ## this function yields coordinates of samples in newdata in PC-space
    ## learned from the training data passed in as x argument.
    return extractor

class PcaExtractor(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """Transforms data set into first m principal components"""
    def __init__(self, m):
        self.m = m
    def fit(self, X, y=None):
        self.extractor = extractPCs(X, m=self.m)
        return self
    def transform(self, X):
        return self.extractor(X)





        

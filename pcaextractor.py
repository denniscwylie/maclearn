import numpy
import pandas
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

class PcaExtractor(BaseEstimator, TransformerMixin):
    """Transforms data set into first k principal components."""

    def __init__(self, k=2, center="both", scale="none", small=1e-10):
        self.k = k
        self.center = center
        self.scale = scale
        self.small = small

    def fit(self, X, y=None):
        xhere = pandas.DataFrame(X.copy())
        if self.center in ['row', 'both']:
            xRowAvs = xhere.mean(axis=1)
            xhere = xhere.add(-xRowAvs, axis=0)
        if self.center in ['col', 'both']:
            self.colAvs_ = xhere.mean(axis=0)
            xhere = xhere.add(-self.colAvs_, axis=1)
        colSds = xhere.std(axis=0)
        xhere.ix[:, colSds==0] += (self.small *
                                   numpy.random.randn(xhere.shape[0],
                                                      sum(colSds==0)))
        if self.scale == 'row':
            rowSds = xhere.std(axis=1)
            xhere = xhere.divide(rowSds, axis=0)
        elif self.scale == 'col':
            self.colSds_ = xhere.std(axis=0)
            xhere = xhere.divide(self.colSds_, axis=1)
        xsvd = numpy.linalg.svd(xhere, full_matrices=False)
        self.v_ = numpy.transpose(xsvd[2])[:, 0:self.k]
        return self

    def transform(self, X):
        xhere = pandas.DataFrame(X.copy())
        if self.center in ['row', 'both']:
            xRowAvs = xhere.mean(axis=1)
            xhere = xhere.add(-xRowAvs, axis=0)
        if self.center in ['col', 'both']:
            xhere = xhere.add(-self.colAvs_, axis=1)
        if self.scale == 'row':
            rowSds = xhere.std(axis=1)
            xhere = xhere.divide(rowSds, axis=0)
        elif self.scale == 'col':
            xhere = xhere.divide(self.colSds_, axis=1)
        return numpy.dot(xhere, self.v_)

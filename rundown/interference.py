__author__ = "Tyler D. Hoffman cause@tdhoffman.com"

"""
Spatial interference adjustments for causal inference.
Basically just include a lag of the treatment variables
This should be a transformer!!!!
"""

import numpy as np
from libpysal.weights import W as WeightsType
from sklearn.base import BaseEstimator, TransformerMixin

class InterferenceAdj(BaseEstimator, TransformerMixin):
    def __init__(self, w=None):
        self.w = w

    def fit(self):
        return self

    def transform(self, Z):
        # Input checks
        if len(Z.shape) < 2:  # ensure column vector
            Z = Z.reshape(-1, 1)

        if type(self.w) == WeightsType:
            weights = self.w.full()[0]
        else:
            weights = self.w

        return np.hstack((Z, np.dot(weights, Z)))

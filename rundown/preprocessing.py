__author__ = "Tyler D. Hoffman cause@tdhoffman.com"

"""
Adjustments for spatial causal inference.
CONTENTS:
- Spatial interference adjustment for causal inference. 
  Adds a lag of the treatment variables according to the specified interference matrix.
- First stage propensity score estimator. Include the results of this in a second
  stage estimator for the outcome.
"""

import numpy as np
from libpysal.weights import W as WeightsType
from sklearn.base import BaseEstimator, TransformerMixin


class InterferenceAdj(BaseEstimator, TransformerMixin):
    """
    Spatial lag transformer: adjoins a lag of the covariates
    """

    def __init__(self, w=None):
        self.w = w

    def fit(self, X, y=None):
        return self  # nothing to fit

    def transform(self, X, y=None):
        # Input checks
        if len(X.shape) < 2:  # ensure column vector
            X = X.reshape(-1, 1)

        if type(self.w) == WeightsType:
            weights = self.w.full()[0]
        else:
            weights = self.w

        return np.hstack((X, np.dot(weights, X)))


class PropEst(BaseEstimator, TransformerMixin):
    """
    Estimates propensity scores prior to fitting a model.
    """

    def __init__(self, w=None, fit_intercept=False):
        self.w = w
        self.fit_intercept = fit_intercept

    def fit(self, X, Z=None):
        if len(X.shape) < 2:  # ensure column vector
            X = X.reshape(-1, 1)

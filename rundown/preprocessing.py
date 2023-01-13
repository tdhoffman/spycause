__author__ = "Tyler D. Hoffman cause@tdhoffman.com"

"""
Adjustments for spatial causal inference.
CONTENTS:
- Spatial interference adjustment for causal inference. 
  Adds a lag of the treatment variables according to the specified interference matrix.
- First stage propensity score estimator. Include the results of this in a second
  stage estimator for the outcome.
"""

import os
import stan
import numpy as np
from libpysal.weights import W as WeightsType
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from patsy import bs

_package_directory = os.path.dirname(os.path.abspath(__file__))


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
    Must be used prior to an interference adjustment.
    """

    def __init__(self, w=None, fit_intercept=False, bs_df=None):
        self.w = w
        self.fit_intercept = fit_intercept
        self.bs_df = bs_df

    def fit(self, X, Z, nchains=1, nsamples=1000, nwarmup=1000, save_warmup=True):
        if len(Z.shape) > 1:
            Z = Z.flatten()

        N, D = X.shape
        if self.fit_intercept:
            X = np.hstack((np.ones((N, 1)), X))
            D += 1

        if type(self.w) == WeightsType:
            node1 = self.w.to_adjlist()['focal'].values + 1
            node2 = self.w.to_adjlist()['neighbor'].values + 1
            N_edges = len(node1)
            self._stanf = os.path.join(_package_directory, "stan", "spatial_logit.stan")
            model_data = {"N": N, "D": D, "X": X, "Z": Z,
                        "N_edges": N_edges, "node1": node1, "node2": node2}
        elif type(self.w) == np.ndarray:
            raise ValueError("w must be libpysal.weights.W in order to access adjacency lists")
        else:
            self._stanf = os.path.join(_package_directory, "stan", "logit.stan")
            model_data = {"N": N, "D": D, "X": X, "Z": Z}

        with open(self._stanf, "r") as f:
            model_code = f.read()

        posterior = stan.build(model_code, data=model_data)
        self.stanfit_ = posterior.sample(num_chains=nchains,
                                         num_samples=nsamples,
                                         num_warmup=nwarmup,
                                         save_warmup=save_warmup)
        self.results_ = self.stanfit_.to_frame()
        return self

    def transform(self):
        check_is_fitted(self)

        pi_hat = self.stanfit_['pi_hat'].mean(1)
        if self.bs_df is not None:
            pi_hat = bs(pi_hat, df=self.bs_df)
        return pi_hat

    def fit_transform(self, X, Z, **kwargs):
        self.fit(X, Z, **kwargs)
        return self.transform()

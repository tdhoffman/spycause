__author__ = "Tyler D. Hoffman cause@tdhoffman.com"

"""
Spatial confounding adjustments for causal inference.
TODO:
- reformat all models to accept treatment vector separately and return separate effects, etc
- reformat all models to accept possible lags of treatment vector (i.e., InterferenceAdjuster was used
  prior to a confounding adjusted model)
- read up on IVs and square it up with Reich
  - SAR as written might be well adapted to becoming the IV model class
  - ...and then we default to CAR in all other scenarios
- add nonlinear capabilities with small NNs (possibly another file)
- check https://github.com/reich-group/SpatialCausalReview for their code
- rectify priors with Reich's priors on pg. 17
"""

import numpy as np
import libpysal.weights as weights
from ..utils import set_endog
from sklearn.base import RegressorMixin
from sklearn.base.linear_model import LinearModel
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from scipy.stats import pearsonr


class SAR(RegressorMixin, LinearModel):
    def __init__(self, w=None, fit_intercept=True):
        self.w = w
        self.fit_intercept = fit_intercept

    def _decision_function(self, X):
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse=True, reset=False)
        return safe_sparse_dot(
            np.linalg.inv(np.eye(self.w.n) - self.indir_coef_ * self.w.full()[0]),
            safe_sparse_dot(X, self.coef_.T, dense_output=True), dense_output=True) + \
            self.intercept_

    def fit(self, X, y, yend=None, q=None, w_lags=1, lag_q=True):
        """
        Fit spatial lag model using generalized method of moments.

        Parameters
        ----------
        X               : array
                          nxk array of covariates
        y               : array
                          nx1 array of dependent variable
        yend            : array
                          nxp array of endogenous variables (default None)
        q               : array
                          nxp array of external endogenous variables to use as instruments
                          (should not contain any variables in X; default None)
        w_lags          : integer
                          orders of W to include as instruments for the spatially
                          lagged dependent variable. For example, w_lags=1, then
                          instruments are WX; if w_lags=2, then WX, WWX; and so on.
                          (default 1)
        epsilon         : float
                          tolerance to use for fitting maximum likelihood models
                          (default 1e-7)

        Returns
        -------
        self            : SAR
                          fitted spreg.sklearn.Lag object
        """

        # Input validation
        X, y = self._validate_data(X, y, accept_sparse=True, y_numeric=True)
        y = y.reshape(-1, 1)  # ensure vector TODO FORMALIZE THIS

        if self.w is None:
            raise ValueError("w must be libpysal.weights.W object")

        if self.fit_intercept:
            X = np.insert(X, 0, np.ones((X.shape[0],)), axis=1)
        else:
            self.intercept_ = 0

        yend2, q2 = set_endog(y, X[:, 1:], self.w,
                              yend, q, w_lags, lag_q)  # assumes constant in first column

        # including exogenous and endogenous variables
        z = np.hstack(X, yend2)
        h = np.hstack(X, q2)
        # k = number of exogenous variables and endogenous variables
        hth = np.dot(h.T, h)
        hthi = np.linalg.inv(hth)
        zth = np.dot(z.T, h)
        hty = np.dot(h.T, y)

        factor_1 = np.dot(zth, hthi)
        factor_2 = np.dot(factor_1, zth.T)
        # this one needs to be in cache to be used in AK
        varb = np.linalg.inv(factor_2)
        factor_3 = np.dot(varb, factor_1)
        params_ = np.dot(factor_3, hty)

        if self.fit_intercept:
            self.coef_ = params_[1:-1].T
            self.intercept_ = params_[0]
        else:
            self.coef_ = params_[:-1].T
        self.indir_coef_ = params_[-1]

        return self

    def score(self, X, y):
        """
        Computes pseudo R2 for the spatial lag model.
        """

        y_pred = self.predict(X)
        return float(pearsonr(y.flatten(), y_pred.flatten())[0]**2)


class SpSmoothing:
    pass

class ModeledPropScore:
    def __init__(self):
        self._stanf = "../stan/prop_score.stan"

class TwoStagePropScore:
    pass

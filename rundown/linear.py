__author__ = "Tyler D. Hoffman cause@tdhoffman.com"

"""
Spatial confounding adjustments for causal inference.
All classes implement linear models with explicit treatment variable.
CONTENTS:
- Bayesian OLS estimation via BayesOLS
- Exact sparse CAR model via CAR
- Intrinsic CAR model via ICAR
- Joint outcome and treatment model via Joint
- Spatial instrumental variables via SpatialIV (UNFINISHED)
"""

import os
import stan
import numpy as np
import arviz as az
from libpysal.weights import W as WeightsType
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from scipy.stats import pearsonr

_package_directory = os.path.dirname(os.path.abspath(__file__))


class BayesOLS(RegressorMixin, LinearModel):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self._stanf = os.path.join(_package_directory, "stan", "ols.stan")

    def predict(self, X, Z):
        # This is all predict is in sklearn.linear_model
        return self._decision_function(X, Z)

    def _decision_function(self, X, Z):
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse=True, reset=False)

        Z = self._validate_data(Z, accept_sparse=True, reset=False)
        base = safe_sparse_dot(X, self.coef_.T, dense_output=True) + \
            safe_sparse_dot(Z, self.ate_.T, dense_output=True)
        if self.fit_intercept:
            base += self.intercept_
        return base

    def fit(self, X, y, Z, nchains=1, nsamples=1000, nwarmup=1000, save_warmup=True):
        N, D = X.shape
        if len(Z.shape) < 1:
            Z = Z.reshape(-1, 1)
        K = Z.shape[1]

        if len(y.shape) > 1:
            y = y.flatten()

        if self.fit_intercept:
            X = np.hstack((np.ones((N, 1)), X))
            D += 1

        with open(self._stanf, "r") as f:
            model_code = f.read()

        model_data = {"N": N, "D": D, "K": K, "X": X, "y": y, "Z": Z}
        posterior = stan.build(model_code, data=model_data)
        self.stanfit_ = posterior.sample(num_chains=nchains,
                                         num_samples=nsamples,
                                         num_warmup=nwarmup,
                                         save_warmup=save_warmup)
        self.results_ = self.stanfit_.to_frame()

        # Get posterior means
        if self.fit_intercept:
            self.intercept_ = self.results_["beta.1"].mean()
            self.coef_ = self.results_[[f"beta.{d+1}" for d in range(1, D)]].mean()
        else:
            self.coef_ = self.results_[[f"beta.{d+1}" for d in range(D)]].mean()
        self.ate_ = self.results_[[f"tau.{i+1}" for i in range(K)]].mean()
        return self

    def score(self, X, y, Z):
        """
        Computes pseudo R2 for the model.
        """

        y_pred = self.predict(X, Z)
        return float(pearsonr(y.flatten(), y_pred.flatten())[0]**2)

    def waic(self):
        """
        Computes WAIC for the model.
        """
        check_is_fitted(self)

        return az.waic(self.stanfit_)


class CAR(RegressorMixin, LinearModel):
    """
    Fits an exact sparse CAR model.
    W must have zeros on the diagonal
    """

    def __init__(self, w=None, fit_intercept=True):
        self.w = w
        self.fit_intercept = fit_intercept
        self._stanf = os.path.join(_package_directory, "stan", "car.stan")

    def predict(self, X, Z):
        # This is all predict is in sklearn.linear_model
        return self._decision_function(X, Z)

    def _decision_function(self, X, Z):
        # TODO REPLACE THIS WITH A CAR PREDICTION
        # IS IT THE SAME AS FOR SAR?
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse=True, reset=False)
        base = safe_sparse_dot(
               np.linalg.inv(np.eye(self.w.n) - self.indir_coef_ * self.w.full()[0]),
               safe_sparse_dot(X, self.coef_.T, dense_output=True), dense_output=True)
        if self.fit_intercept:
            base += self.intercept_
        return base

    def fit(self, X, y, Z, nchains=1, nsamples=1000, nwarmup=1000, save_warmup=True):
        N, D = X.shape
        if len(Z.shape) < 1:
            Z = Z.reshape(-1, 1)
        K = Z.shape[1]

        if len(y.shape) > 1:
            y = y.flatten()

        if self.fit_intercept:
            X = np.hstack((np.ones((N, 1)), X))

        # Process weights matrix
        if type(self.w) == WeightsType:
            w = self.w.full()[0]
        else:
            w = self.w
        W_n = w[np.triu_indices(N)].sum(dtype=np.int64)  # number of adjacent region pairs

        with open(self._stanf, "r") as f:
            model_code = f.read()

        model_data = {"N": N, "D": D, "K": K, "X": X, "y": y, "Z": Z,
                      "W": w, "W_n": W_n}
        posterior = stan.build(model_code, data=model_data)
        self.stanfit_ = posterior.sample(num_chains=nchains,
                                         num_samples=nsamples,
                                         num_warmup=nwarmup,
                                         save_warmup=save_warmup)
        self.results_ = self.stanfit_.to_frame()

        # Get posterior means
        if self.fit_intercept:
            self.intercept_ = self.results_["beta.1"].mean()
            self.coef_ = self.results_[[f"beta.{d+1}" for d in range(1, D)]].mean()
        else:
            self.coef_ = self.results_[[f"beta.{d+1}" for d in range(D)]].mean()
        self.ate_ = self.results_[[f"tau.{i+1}" for i in range(K)]].mean()
        self.indir_coef_ = self.results_["rho"].mean()
        return self

    def score(self, X, y, Z):
        """
        Computes pseudo R2 for the model.
        """

        y_pred = self.predict(X, Z)
        return float(pearsonr(y.flatten(), y_pred.flatten())[0]**2)


class ICAR(RegressorMixin, LinearModel):
    def __init__(self, w=None, fit_intercept=True):
        self.w = w
        self.fit_intercept = fit_intercept
        self._stanf = os.path.join(_package_directory, "stan", "icar.stan")

    def predict(self, X, Z):
        # This is all predict is in sklearn.linear_model
        return self._decision_function(X, Z)

    def _decision_function(self, X, Z):
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse=True, reset=False)
        eps = np.random.random(size=(X.shape[0],))
        # eps = np.random.normal(size=(X.shape[0],))
        U = safe_sparse_dot(np.linalg.inv(np.eye(self.w.n) - self.w.full()[0]), eps, dense_output=True)
        base = safe_sparse_dot(X, self.coef_.T, dense_output=True) + \
            safe_sparse_dot(Z, self.ate_.T, dense_output=True) + U
        if self.fit_intercept:
            base += self.intercept_
        return base

    def fit(self, X, y, Z, nchains=1, nsamples=1000, nwarmup=1000, save_warmup=True):
        N, D = X.shape
        if len(Z.shape) < 1:
            Z = Z.reshape(-1, 1)
        K = Z.shape[1]

        if len(y.shape) > 1:
            y = y.flatten()

        if self.fit_intercept:
            X = np.hstack((np.ones((N, 1)), X))

        # Process weights matrix
        if type(self.w) == WeightsType:
            node1 = self.w.to_adjlist()['focal'].values + 1
            node2 = self.w.to_adjlist()['neighbor'].values + 1
            N_edges = len(node1)
        else:
            raise ValueError("w must be libpysal.weights.W in order to access adjacency lists")

        with open(self._stanf, "r") as f:
            model_code = f.read()

        model_data = {"N": N, "D": D, "K": K, "X": X, "y": y, "Z": Z,
                      "N_edges": N_edges, "node1": node1, "node2": node2}
        posterior = stan.build(model_code, data=model_data)
        self.stanfit_ = posterior.sample(num_chains=nchains,
                                         num_samples=nsamples,
                                         num_warmup=nwarmup,
                                         save_warmup=save_warmup)
        self.results_ = self.stanfit_.to_frame()

        # Get posterior means
        if self.fit_intercept:
            self.intercept_ = self.results_["beta.1"].mean().values
            self.coef_ = self.results_[[f"beta.{d+1}" for d in range(1, D)]].mean().values
        else:
            self.coef_ = self.results_[[f"beta.{d+1}" for d in range(D)]].mean().values
        self.ate_ = self.results_[[f"tau.{i+1}" for i in range(K)]].mean().values
        self.indir_coef_ = self.results_["sd_r"].mean()
        return self

    def waic(self):
        """
        Computes WAIC for the model.
        """

        check_is_fitted(self)
        return az.waic(self.stanfit_)

class Joint(RegressorMixin, LinearModel):
    def __init__(self, w=None, fit_intercept=True):
        self.w = w
        self.fit_intercept = fit_intercept
        self._stanf = os.path.join(_package_directory, "stan", "prop_score.stan")

    def predict(self, X, Z):
        # This is all predict is in sklearn.linear_model
        return self._decision_function(X, Z)

    def _decision_function(self, X, Z):
        # TODO REPLACE THIS WITH THE RIGHT PREDICTION
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse=True, reset=False)
        base = safe_sparse_dot(
               np.linalg.inv(np.eye(self.w.n) - self.indir_coef_ * self.w.full()[0]),
               safe_sparse_dot(X, self.coef_.T, dense_output=True), dense_output=True)
        if self.fit_intercept:
            base += self.intercept_
        return base

    def fit(self, X, y, Z, nchains=1, nsamples=1000, nwarmup=1000, save_warmup=True):
        """
        Interference adjustment is going to be tricky here
        Stan doesn't like polymorphism
        """

        N, D = X.shape
        K = Z.shape[1]
        if len(Z.shape) > 1 and Z.shape[1] > 1:
            Zlag = Z[:, 1].flatten()
            Z = Z[:, 0].flatten().astype(int)
        else:
            Z = Z.flatten().astype(int)
            Zlag = np.empty(shape=Z.shape)

        if len(y.shape) > 1:
            y = y.flatten()

        if self.fit_intercept:
            X = np.hstack((np.ones((N, 1)), X))

        # Process weights matrix
        if type(self.w) == WeightsType:
            node1 = self.w.to_adjlist()['focal'].values + 1
            node2 = self.w.to_adjlist()['neighbor'].values + 1
            N_edges = len(node1)
            W = self.w.full()[0]
        else:
            raise ValueError("w must be libpysal.weights.W in order to access adjacency lists")

        with open(self._stanf, "r") as f:
            model_code = f.read()

        model_data = {"N": N, "D": D, "K": K, "X": X, "Y": y, "Z": Z, "W": W,
                      "Zlag": Zlag, "N_edges": N_edges, "node1": node1, "node2": node2}
        posterior = stan.build(model_code, data=model_data)
        self.stanfit_ = posterior.sample(num_chains=nchains,
                                         num_samples=nsamples,
                                         num_warmup=nwarmup,
                                         save_warmup=save_warmup)
        self.results_ = self.stanfit_.to_frame()

        # Get posterior means
        if self.fit_intercept:
            self.intercept_ = self.results_["beta.1"].mean()
            self.coef_ = self.results_[[f"beta.{d+1}" for d in range(1, D)]].mean()
        else:
            self.coef_ = self.results_[[f"beta.{d+1}" for d in range(D)]].mean()
        self.ate_ = self.results_[[f"tau.{i+1}" for i in range(K)]].mean()
        return self

    def waic(self):
        """
        Computes WAIC for the model.
        """

        check_is_fitted(self)
        return az.waic(self.stanfit_)


class SpatialIV(RegressorMixin, LinearModel):
    """
    Fits the spatial instrumental variable model:
    Y = (gamma_hat*A)*tau + X*beta + u + eps_y
    Z = alpha + A*gamma + X*lambda + phi*u + v + eps_z
    where A is an instrument, u and v are CAR terms.
    UNFINISHED AND UNTESTED
    """

    def __init__(self, w=None, fit_intercept=True):
        self.w = w
        self.fit_intercept = fit_intercept
        self._stanf = os.path.join(_package_directory, "stan", "iv.stan")

    def fit(self, X, y, Z, A, nchains=1, nsamples=1000, nwarmup=1000, save_warmup=True):
        N, D = X.shape
        if len(Z.shape) < 1:
            Z = Z.reshape(-1, 1)
        K = Z.shape[1]

        if len(y.shape) > 1:
            y = y.flatten()

        if self.fit_intercept:
            X = np.hstack((np.ones((N, 1)), X))

        # Process weights matrix
        if type(self.w) == WeightsType:
            node1 = self.w.to_adjlist()['focal']
            node2 = self.w.to_adjlist()['neighbor']
            w = self.w.full()[0]
        else:
            raise ValueError("w must be libpysal.weights.W in order to access adjacency lists")

        with open(self._stanf, "r") as f:
            model_code = f.read()

        model_data = {"N": N, "D": D, "K": K, "X": X, "y": y, "Z": Z, "A": A,
                      "W": w, "node1": node1.values + 1, "node2": node2.values + 1}
        posterior = stan.build(model_code, data=model_data)
        self.stanfit_ = posterior.sample(num_chains=nchains,
                                         num_samples=nsamples,
                                         num_warmup=nwarmup,
                                         save_warmup=save_warmup)
        self.results_ = self.stanfit_.to_frame()

        # Get posterior means
        if self.fit_intercept:
            self.intercept_ = self.results_["beta.1"].mean()
            self.coef_ = self.results_[[f"beta.{d+1}" for d in range(1, D)]].mean()
        else:
            self.coef_ = self.results_[[f"beta.{d+1}" for d in range(D)]].mean()
        self.ate_ = self.results_[[f"tau.{i+1}" for i in range(K)]].mean()
        self.indir_coef_ = self.results_["rho"].mean()
        return self

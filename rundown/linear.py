__author__ = "Tyler D. Hoffman cause@tdhoffman.com"

"""
Spatial confounding adjustments for causal inference.
All classes implement linear models with explicit treatment variable.
CONTENTS:
- Bayesian OLS estimation via BayesOLS
- Intrinsic CAR model via ICAR
- Joint outcome and treatment model via Joint
"""

import os
from cmdstanpy import CmdStanModel
import numpy as np
import arviz as az
# from .diagnostics import diagnostics
from libpysal.weights import W as WeightsType
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.utils.validation import check_is_fitted
from scipy.stats import pearsonr

_package_directory = os.path.dirname(os.path.abspath(__file__))


class BayesOLS(RegressorMixin, LinearModel):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self._stanf = os.path.join(_package_directory, "stan", "ols.stan")

    def fit(self, X, y, Z, nchains=1, nsamples=1000, nwarmup=1000, save_warmup=False,
            delta=0.8, max_depth=10):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        N, D = X.shape

        if len(Z.shape) == 1:
            Z = Z.reshape(-1, 1)
        K = Z.shape[1]

        if len(y.shape) > 1:
            y = y.flatten()

        if self.fit_intercept:
            X = np.hstack((np.ones((N, 1)), X))
            D += 1

        model = CmdStanModel(stan_file=self._stanf)

        model_data = {"N": N, "D": D, "K": K, "X": X, "y": y, "Z": Z}
        self.stanfit_ = model.sample(data=model_data,
                                     chains=nchains,
                                     iter_warmup=nwarmup,
                                     iter_sampling=nsamples,
                                     save_warmup=save_warmup,
                                     adapt_delta=delta,
                                     max_treedepth=max_depth,
                                     show_progress=True,
                                     show_console=False)
        self.results_ = self.stanfit_.draws_pd()

        # Get posterior means
        if self.fit_intercept:
            self.intercept_ = self.results_["beta[1]"].mean()
            self.coef_ = self.results_[[f"beta[{d+1}]" for d in range(1, D)]].mean()
        else:
            self.coef_ = self.results_[[f"beta[{d+1}]" for d in range(D)]].mean()
        self.ate_ = self.results_[[f"tau[{i+1}]" for i in range(K)]].mean()
        self.idata_ = az.from_cmdstanpy(posterior=self.stanfit_,
                                        posterior_predictive="y_pred",
                                        log_likelihood="log_likelihood")
        self.max_depth = max_depth
        return self

    def score(self, X, y, Z):
        """
        Computes pseudo R2 for the model using posterior predictive.
        """

        y_pred = self.predict()
        return float(pearsonr(y.flatten(), y_pred.flatten())[0]**2)

    def waic(self):
        """
        Computes WAIC for the model.
        """
        check_is_fitted(self)

        return az.waic(self.idata_)

    def diagnostics(self):
        # diagnostics(self, params=["beta", "tau", "sigma"])
        check_is_fitted(self)

        return self.stanfit_.diagnose()


class ICAR(RegressorMixin, LinearModel):
    def __init__(self, w=None, fit_intercept=True):
        self.w = w
        self.fit_intercept = fit_intercept
        self._stanf = os.path.join(_package_directory, "stan", "icar.stan")

    def fit(self, X, y, Z, nchains=1, nsamples=1000, nwarmup=1000, save_warmup=False,
            delta=0.8, max_depth=10):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        N, D = X.shape

        if len(Z.shape) == 1:
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
            weights = self.w.to_adjlist()['weight'].values
            N_edges = len(node1)
        else:
            raise ValueError("w must be libpysal.weights.W in order to access adjacency lists")

        model = CmdStanModel(stan_file=self._stanf)

        model_data = {"N": N, "D": D, "K": K, "X": X, "y": y, "Z": Z,
                      "N_edges": N_edges, "node1": node1, "node2": node2, "weights": weights}
        self.stanfit_ = model.sample(data=model_data,
                                     chains=nchains,
                                     iter_warmup=nwarmup,
                                     iter_sampling=nsamples,
                                     save_warmup=save_warmup,
                                     adapt_delta=delta,
                                     max_treedepth=max_depth,
                                     show_progress=True,
                                     show_console=False)
        self.results_ = self.stanfit_.draws_pd()

        # Get posterior means
        if self.fit_intercept:
            self.intercept_ = self.results_["beta[1]"].mean().values
            self.coef_ = self.results_[[f"beta[{d+1}]" for d in range(1, D)]].mean().values
        else:
            self.coef_ = self.results_[[f"beta[{d+1}]" for d in range(D)]].mean().values
        self.ate_ = self.results_[[f"tau[{i+1}]" for i in range(K)]].mean().values
        self.idata_ = az.from_cmdstanpy(posterior=self.stanfit_,
                                        posterior_predictive="y_pred",
                                        log_likelihood="log_likelihood")
        self.max_depth = max_depth
        return self

    def score(self, X, y, Z):
        """
        Computes pseudo R2 for the model using posterior predictive.
        """

        y_pred = self.predict()
        return float(pearsonr(y.flatten(), y_pred.flatten())[0]**2)

    def waic(self):
        """
        Computes WAIC for the model.
        """

        check_is_fitted(self)
        return az.waic(self.idata_)

    def diagnostics(self):
        # diagnostics(self, params=["beta", "tau", "sigma", "u"])
        check_is_fitted(self)

        return self.stanfit_.diagnose()

class Joint(RegressorMixin, LinearModel):
    def __init__(self, w=None, fit_intercept=True):
        self.w = w
        self.fit_intercept = fit_intercept
        self._stanf = os.path.join(_package_directory, "stan", "joint.stan")

    def fit(self, X, y, Z, nchains=1, nsamples=1000, nwarmup=1000, save_warmup=False,
            delta=0.8, max_depth=10):
        """
        Interference adjustment is going to be tricky here
        Stan doesn't like polymorphism
        """

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
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
            weights = self.w.to_adjlist()['weight'].values
            N_edges = len(node1)
            W = self.w.full()[0]
        else:
            raise ValueError("w must be libpysal.weights.W in order to access adjacency lists")

        model = CmdStanModel(stan_file=self._stanf)

        model_data = {"N": N, "D": D, "K": K, "X": X, "Y": y, "Z": Z, "W": W,
                      "Zlag": Zlag, "N_edges": N_edges, "node1": node1, "node2": node2, "weights": weights}
        self.stanfit_ = model.sample(data=model_data,
                                     chains=nchains,
                                     iter_warmup=nwarmup,
                                     iter_sampling=nsamples,
                                     save_warmup=save_warmup,
                                     adapt_delta=delta,
                                     max_treedepth=max_depth,
                                     show_progress=True,
                                     show_console=False)
        self.results_ = self.stanfit_.draws_pd()

        # Get posterior means
        if self.fit_intercept:
            self.intercept_ = self.results_["beta[1]"].mean()
            self.coef_ = self.results_[[f"beta[{d+1}]" for d in range(1, D)]].mean()
        else:
            self.coef_ = self.results_[[f"beta[{d+1}]" for d in range(D)]].mean()
        self.ate_ = self.results_[[f"tau[{i+1}]" for i in range(K)]].mean()
        self.idata_ = az.from_cmdstanpy(posterior=self.stanfit_,
                                        posterior_predictive="y_pred",
                                        log_likelihood="log_likelihood")
        self.max_depth = max_depth
        return self

    def score(self, X, y, Z):
        """
        Computes pseudo R2 for the model using posterior predictive.
        """

        y_pred = self.predict()
        return float(pearsonr(y.flatten(), y_pred.flatten())[0]**2)

    def waic(self):
        """
        Computes WAIC for the model.
        """

        check_is_fitted(self)
        return az.waic(self.idata_)

    def diagnostics(self):
        # diagnostics(self, params=["beta", "tau", "sigma", "alpha", "u", "v", "sd_u", "sd_v", "psi"])
        check_is_fitted(self)

        return self.stanfit_.diagnose()

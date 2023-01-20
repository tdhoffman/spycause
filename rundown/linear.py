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
import stan
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
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

    def predict(self):
        """
        Return posterior predictive.
        """
        check_is_fitted(self)
        return self.stanfit_['y_pred'].mean(1)

    def fit(self, X, y, Z, nchains=1, nsamples=1000, nwarmup=1000, save_warmup=True,
            delta=0.8, max_depth=10):
        if len(X.shape) < 1:
            X = X.reshape(-1, 1)
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
                                         save_warmup=save_warmup,
                                         delta=delta,
                                         max_depth=max_depth)
        self.results_ = self.stanfit_.to_frame()

        # Get posterior means
        if self.fit_intercept:
            self.intercept_ = self.results_["beta.1"].mean()
            self.coef_ = self.results_[[f"beta.{d+1}" for d in range(1, D)]].mean()
        else:
            self.coef_ = self.results_[[f"beta.{d+1}" for d in range(D)]].mean()
        self.ate_ = self.results_[[f"tau.{i+1}" for i in range(K)]].mean()
        self.idata_ = az.from_pystan(self.stanfit_, log_likelihood="log_likelihood")
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

    def diagnostics(self, vis=False):
        """
        Return some diagnostics about the posterior convergence.
        """

        # Divergences
        # divergences = self.stanfit_.get_sampler_params()["divergent__"]
        # ndivergent = divergences.sum()
        # nsamples = len(divergences)
        # print(f"{ndivergent} of {nsamples} iterations ended with a divergence ({100*ndivergent/nsamples}%).")
        # if ndivergent > 0:
            # print("Increasing adapt_delta may remove the divergences.")

        # Tree depth
        # treedepths = self.stanfit_.get_sampler_params()["treedepth__"]
        # nmaxdepths = (treedepths == self.max_depth).sum()
        # print(f"{nmaxdepths} of {nsamples} iterations saturated the max tree depth ({100*nmaxdepths/nsamples}%).")
        # if nmaxdepths > 0:
            # print("See https://betanalpha.github.io/assets/case_studies/identifiability.html for more information.")

        # ESS
        self.esses = az.ess(self.idata_)

        # Rhat
        self.rhats = az.rhat(self.idata_)

        # Energy
        self.bfmis = az.bfmi(self.idata_)

        if vis:
            az.plot_trace(self.idata_)
            az.plot_ess(self.idata_)


class ICAR(RegressorMixin, LinearModel):
    def __init__(self, w=None, fit_intercept=True):
        self.w = w
        self.fit_intercept = fit_intercept
        self._stanf = os.path.join(_package_directory, "stan", "icar.stan")

    def predict(self):
        """
        Return posterior predictive.
        """
        check_is_fitted(self)
        return self.stanfit_['y_pred'].mean(1)

    def fit(self, X, y, Z, nchains=1, nsamples=1000, nwarmup=1000, save_warmup=True):
        if len(X.shape) < 1:
            X = X.reshape(-1, 1)
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
            weights = self.w.to_adjlist()['weight'].values
            N_edges = len(node1)
        else:
            raise ValueError("w must be libpysal.weights.W in order to access adjacency lists")

        with open(self._stanf, "r") as f:
            model_code = f.read()

        model_data = {"N": N, "D": D, "K": K, "X": X, "y": y, "Z": Z,
                      "N_edges": N_edges, "node1": node1, "node2": node2, "weights": weights}
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
        self.idata_ = az.from_pystan(self.stanfit_, log_likelihood="log_likelihood")
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

class Joint(RegressorMixin, LinearModel):
    def __init__(self, w=None, fit_intercept=True):
        self.w = w
        self.fit_intercept = fit_intercept
        self._stanf = os.path.join(_package_directory, "stan", "joint.stan")

    def predict(self):
        """
        Return posterior predictive.
        """
        check_is_fitted(self)
        return self.stanfit_['y_pred'].mean(1)

    def fit(self, X, y, Z, nchains=1, nsamples=1000, nwarmup=1000, save_warmup=True):
        """
        Interference adjustment is going to be tricky here
        Stan doesn't like polymorphism
        """

        if len(X.shape) < 1:
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
        self.idata_ = az.from_pystan(self.stanfit_, log_likelihood="log_likelihood")
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

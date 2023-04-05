__author__ = "Tyler D. Hoffman cause@tdhoffman.com"

"""
Spatial confounding adjustments for causal inference.
All classes implement linear models with explicit treatment variable.
CONTENTS:
- Bayesian OLS estimation via BayesOLS
- Intrinsic CAR model via ICAR
- Exact sparse CAR model via CAR
- Joint outcome and treatment model via Joint
"""

import os
from shutil import rmtree
from tempfile import mkdtemp
from cmdstanpy import CmdStanModel
import numpy as np
import arviz as az
from libpysal.weights import W as WeightsType
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.utils.validation import check_is_fitted
from scipy.stats import pearsonr

_package_directory = os.path.dirname(os.path.abspath(__file__))


class BayesOLS(RegressorMixin, LinearModel):
    def __init__(self, fit_intercept=True):
        """
        Initialize model class for ordinary linear regression.

        fit_intercept : bool, defaults to True
                       whether or not to use an intercept
        """

        self.fit_intercept = fit_intercept
        self._stanf = os.path.join(_package_directory, "stan", "ols.stan")

    def fit(
        self,
        X,
        y,
        Z,
        nchains=1,
        nsamples=1000,
        nwarmup=1000,
        save_warmup=False,
        delta=0.8,
        max_depth=10,
        simulation=False,
    ):
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

        output_dir = None
        compile_file = True  # compile as needed if not simulating
        stanexe = None
        if simulation is True:
            output_dir = mkdtemp(dir=".")
            compile_file = False
            stanexe = os.path.join(_package_directory, "stan", "ols")

        model = CmdStanModel(
            stan_file=self._stanf, exe_file=stanexe, compile=compile_file
        )

        model_data = {"N": N, "D": D, "K": K, "X": X, "y": y, "Z": Z}
        self.stanfit_ = model.sample(
            data=model_data,
            chains=nchains,
            iter_warmup=nwarmup,
            iter_sampling=nsamples,
            save_warmup=save_warmup,
            adapt_delta=delta,
            max_treedepth=max_depth,
            show_progress=True,
            show_console=False,
            output_dir=output_dir,
        )
        self.results_ = self.stanfit_.draws_pd()

        # Get posterior medians
        if self.fit_intercept:
            self.intercept_ = self.results_["beta[1]"].median().values
            self.coef_ = (
                self.results_[[f"beta[{d+1}]" for d in range(1, D)]].median().values
            )
        else:
            self.coef_ = (
                self.results_[[f"beta[{d+1}]" for d in range(D)]].median().values
            )
        self.ate_ = self.results_[[f"tau[{i+1}]" for i in range(K)]].median().values
        self.idata_ = az.from_cmdstanpy(
            posterior=self.stanfit_,
            posterior_predictive="y_pred",
            log_likelihood="log_likelihood",
        )
        self.max_depth = max_depth

        if simulation:
            rmtree(output_dir)
        return self

    def predict(self):
        """
        Return posterior predictive.
        """
        check_is_fitted(self)

        return self.results_.filter(regex=r"^y_pred", axis=1).median().values

    def score(self, X, y, Z):
        """
        Computes pseudo R2 for the model using posterior predictive.
        """
        check_is_fitted(self)

        y_pred = self.predict()
        return float(pearsonr(y.flatten(), y_pred.flatten())[0] ** 2)

    def waic(self):
        """
        Computes WAIC for the model.
        """
        check_is_fitted(self)

        return az.waic(self.idata_)

    def diagnostics(self):
        return self.stanfit_.diagnose()


class ICAR(RegressorMixin, LinearModel):
    def __init__(self, w=None, fit_intercept=True):
        self.w = w
        self.fit_intercept = fit_intercept
        self._stanf = os.path.join(_package_directory, "stan", "icar.stan")

    def fit(
        self,
        X,
        y,
        Z,
        nchains=1,
        nsamples=1000,
        nwarmup=1000,
        save_warmup=False,
        delta=0.8,
        max_depth=10,
        simulation=False,
    ):
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
        if isinstance(self.w, WeightsType):
            node1 = self.w.to_adjlist()["focal"].values + 1
            node2 = self.w.to_adjlist()["neighbor"].values + 1
            weights = self.w.to_adjlist()["weight"].values
            N_edges = len(node1)
        else:
            raise ValueError(
                "w must be libpysal.weights.W in order to access adjacency lists"
            )

        output_dir = None
        if simulation is True:
            output_dir = mkdtemp(dir=".")

        model = CmdStanModel(stan_file=self._stanf)

        model_data = {
            "N": N,
            "D": D,
            "K": K,
            "X": X,
            "y": y,
            "Z": Z,
            "N_edges": N_edges,
            "node1": node1,
            "node2": node2,
            "weights": weights,
        }
        self.stanfit_ = model.sample(
            data=model_data,
            chains=nchains,
            iter_warmup=nwarmup,
            iter_sampling=nsamples,
            save_warmup=save_warmup,
            adapt_delta=delta,
            max_treedepth=max_depth,
            show_progress=True,
            show_console=False,
            output_dir=output_dir,
        )
        self.results_ = self.stanfit_.draws_pd()

        # Get posterior medians
        if self.fit_intercept:
            self.intercept_ = self.results_["beta[1]"].median().values
            self.coef_ = (
                self.results_[[f"beta[{d+1}]" for d in range(1, D)]].median().values
            )
        else:
            self.coef_ = (
                self.results_[[f"beta[{d+1}]" for d in range(D)]].median().values
            )
        self.ate_ = self.results_[[f"tau[{i+1}]" for i in range(K)]].median().values
        self.idata_ = az.from_cmdstanpy(
            posterior=self.stanfit_,
            posterior_predictive="y_pred",
            log_likelihood="log_likelihood",
        )
        self.max_depth = max_depth

        if simulation:
            rmtree(output_dir)
        return self

    def predict(self):
        """
        Return posterior predictive.
        """
        check_is_fitted(self)

        return self.results_.filter(regex=r"^y_pred", axis=1).median().values

    def score(self, X, y, Z):
        """
        Computes pseudo R2 for the model using posterior predictive.
        """

        y_pred = self.predict()
        return float(pearsonr(y.flatten(), y_pred.flatten())[0] ** 2)

    def waic(self):
        """
        Computes WAIC for the model.
        """

        check_is_fitted(self)
        return az.waic(self.idata_)

    def diagnostics(self):
        return self.stanfit_.diagnose()


class CAR(RegressorMixin, LinearModel):
    """
    Fits an exact sparse CAR model.
    W must have zeros on the diagonal
    """

    def __init__(self, w=None, fit_intercept=False):
        self.w = w
        self.fit_intercept = fit_intercept
        self._stanf = os.path.join(_package_directory, "stan", "car.stan")

    def fit(
        self,
        X,
        y,
        Z,
        nchains=1,
        nsamples=1000,
        nwarmup=1000,
        save_warmup=True,
        delta=0.8,
        max_depth=10,
        simulation=False,
    ):
        N, D = X.shape
        if len(Z.shape) < 1:
            Z = Z.reshape(-1, 1)
        K = Z.shape[1]

        if len(y.shape) > 1:
            y = y.flatten()

        if self.fit_intercept:
            X = np.hstack((np.ones((N, 1)), X))

        # Process weights matrix
        if isinstance(self.w, WeightsType):
            self.w.transform = "o"
            w = self.w.full()[0]
        else:
            w = self.w
        W_n = w[np.triu_indices(N)].sum(
            dtype=np.int64
        )  # number of adjacent region pairs

        output_dir = None
        compile_file = True  # compile as needed if not simulating
        stanexe = None
        if simulation is True:
            output_dir = mkdtemp(dir=".")
            compile_file = False
            stanexe = os.path.join(_package_directory, "stan", "car")

        model = CmdStanModel(
            stan_file=self._stanf, exe_file=stanexe, compile=compile_file
        )

        model_data = {
            "N": N,
            "D": D,
            "K": K,
            "X": X,
            "y": y,
            "Z": Z,
            "W": w,
            "W_n": W_n,
        }
        self.stanfit_ = model.sample(
            data=model_data,
            chains=nchains,
            iter_warmup=nwarmup,
            iter_sampling=nsamples,
            save_warmup=save_warmup,
            adapt_delta=delta,
            max_treedepth=max_depth,
            show_progress=True,
            show_console=False,
            output_dir=output_dir,
        )
        self.results_ = self.stanfit_.draws_pd()

        # Get posterior medians
        if self.fit_intercept:
            self.intercept_ = self.results_["beta[1]"].median()
            self.coef_ = self.results_[[f"beta[{d+1}]" for d in range(1, D)]].median()
        else:
            self.coef_ = self.results_[[f"beta[{d+1}]" for d in range(D)]].median()
        self.ate_ = self.results_[[f"tau[{i+1}]" for i in range(K)]].median()
        self.indir_coef_ = self.results_["rho"].median()
        self.idata_ = az.from_cmdstanpy(
            posterior=self.stanfit_,
            posterior_predictive="y_pred",
            log_likelihood="log_likelihood",
        )
        self.max_depth = max_depth

        if simulation:
            rmtree(output_dir)
        return self

    def predict(self):
        """
        Return posterior predictive.
        """
        check_is_fitted(self)

        return self.results_.filter(regex=r"^y_pred", axis=1).median().values

    def waic(self):
        """
        Computes WAIC for the model.
        """

        check_is_fitted(self)
        return az.waic(self.idata_)

    def diagnostics(self):
        return self.stanfit_.diagnose()


class Joint(RegressorMixin, LinearModel):
    def __init__(self, w=None, fit_intercept=False):
        self.w = w
        self.fit_intercept = fit_intercept
        self._stanf = os.path.join(_package_directory, "stan", "joint.stan")

    def fit(
        self,
        X,
        Y,
        Z,
        nchains=1,
        nsamples=1000,
        nwarmup=1000,
        save_warmup=False,
        delta=0.8,
        max_depth=10,
        simulation=False,
    ):
        """
        Fits model
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

        if len(Y.shape) > 1:
            Y = Y.flatten()

        if self.fit_intercept:
            X = np.hstack((np.ones((N, 1)), X))

        # Process weights matrix
        if isinstance(self.w, WeightsType):
            self.w.transform = "o"
            w = self.w.full()[0]
        else:
            w = self.w
        W_n = w[np.triu_indices(N)].sum(
            dtype=np.int64
        )  # number of adjacent region pairs

        output_dir = None
        compile_file = True  # compile as needed if not simulating
        stanexe = None
        if simulation is True:
            output_dir = mkdtemp(dir=".")
            compile_file = False
            stanexe = os.path.join(_package_directory, "stan", "joint")

        model = CmdStanModel(
            stan_file=self._stanf, exe_file=stanexe, compile=compile_file
        )

        model_data = {
            "N": N,
            "D": D,
            "K": K,
            "X": X,
            "Y": Y,
            "Z": Z,
            "Zlag": Zlag,
            "W": w,
            "W_n": W_n,
        }
        self.stanfit_ = model.sample(
            data=model_data,
            chains=nchains,
            iter_warmup=nwarmup,
            iter_sampling=nsamples,
            save_warmup=save_warmup,
            adapt_delta=delta,
            max_treedepth=max_depth,
            show_progress=True,
            show_console=False,
            output_dir=output_dir,
        )
        self.results_ = self.stanfit_.draws_pd()

        # Get posterior medians
        if self.fit_intercept:
            self.intercept_ = self.results_["beta[1]"].median().values
            self.coef_ = (
                self.results_[[f"beta[{d+1}]" for d in range(1, D)]].median().values
            )
        else:
            self.coef_ = (
                self.results_[[f"beta[{d+1}]" for d in range(D)]].median().values
            )
        self.ate_ = self.results_[[f"tau[{i+1}]" for i in range(K)]].median().values
        self.idata_ = az.from_cmdstanpy(
            posterior=self.stanfit_,
            posterior_predictive="y_pred",
            log_likelihood="log_likelihood",
        )
        self.max_depth = max_depth

        if simulation:
            rmtree(output_dir)
        return self

    def predict(self):
        """
        Return posterior predictive.
        """
        check_is_fitted(self)

        return self.results_.filter(regex=r"^y_pred", axis=1).median().values

    def score(self, X, y, Z):
        """
        Computes pseudo R2 for the model using posterior predictive.
        """

        y_pred = self.predict()
        return float(pearsonr(y.flatten(), y_pred.flatten())[0] ** 2)

    def waic(self):
        """
        Computes WAIC for the model.
        """

        check_is_fitted(self)
        return az.waic(self.idata_)

    def diagnostics(self):
        return self.stanfit_.diagnose()

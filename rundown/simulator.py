__author__ = "Tyler D. Hoffman cause@tdhoffman.com"

"""
Sets up simulation classes to be reused in experiments.
"""

import numpy as np
import pandas as pd
import libpysal.weights as weights
from spopt.region import RandomRegion
from math import floor

class Simulator:
    def __init__(self, Nlat, D, sp_confound=None, interference=None):
        """
        Initialize self and parameters for simulation.

        Parameters
        ----------
        Nlat         : int
                       side length of the lattice
        D            : int
                       number of covariates to generate
        sp_confound  : matrix, default None
                       matrix specifying the mode of confounding between locations
        interference : matrix, string, or int, default None
                       matrix specifying the mode of interference between locations
                       string options:
                         - "general": interference between all locations
                         - "partial": interference only among locations in clusters
                           generated using RandomRegion. Recommended to select the clusters
                           yourself and manually input the adjacency matrix (block diagonal)
                           as the RandomRegion call takes a while. If int, then generates
                           that number of clusters.
                         - "network": interference among adjacent locations using Queen weights
        """

        self.Nlat = Nlat
        self.N = Nlat**2
        self.D = D
        self.sp_confound = sp_confound

        # Parse interference options
        if type(interference) == str and interference == "general":
            interference = np.ones((self.N, self.N))
        elif type(interference) == str and interference == "network":
            interference = weights.lat2W(Nlat, Nlat, rook=False).full()[0]
        elif type(interference) == int or (type(interference) == str and interference == "partial"):
            W = weights.lat2W(Nlat, Nlat, rook=False)

            if type(interference) == int:
                nregs = interference
            else:
                nregs = np.random.randint(low=4, high=10)
            t1 = RandomRegion(W.id_order, num_regions=nregs, contiguity=W, compact=True)

            source = []
            dest = []
            for region in t1.regions:
                region = set(region)
                for node in region:
                    source.append(node)
                    dest += [i for i in region.difference({node})]
            adjlist = pd.DataFrame(columns=["source", "dest"], data=np.dstack((source, dest)))

            interference = weights.W.from_adjlist(adjlist)
        else:
            raise ValueError("Unknown kind of interference")

        # Enforce row-standardization:
        interference /= interference.sum(1, keepdims=1)
        self.interference = interference

    def simulate(self, x_sd=1, x_sp=0.9, **kwargs):
        """
        Simulate data based on some parameters.

        Returns
        -------
        X            : covariates (NxD)
        Y            : outcomes (Nx1)
        Z            : treatment (Nx1)
        """

        if np.ndim(x_sd) == 0:
            x_sd = np.repeat(x_sd, self.D)

        # Confounders
        means = np.random.choice(np.arange(-2*self.D, 2*self.D + 1, 1, dtype=int),
                                 size=self.D, replace=False)
        X = np.zeros((self.N, self.D))

        W = weights.lat2W(self.Nlat, self.Nlat, rook=True)

        for d in range(self.D):
            X[:, d] = np.random.normal(loc=means[d], scale=x_sd[d], size=(self.N,))
            X[:, d] = np.dot(np.linalg.pinv(np.eye(self.N) - x_sp*W), X[:, d])

        Z = self._create_Z(X, **kwargs)
        Y = self._create_Y(X, Z, **kwargs)
        return X, Y, Z

    def _create_Y(self, X, Z, **kwargs):
        """
        Generate Y based on parameters, confounders X, and treatment Z.
        """

        raise NotImplementedError("Subclasses must define this")

    def _create_Z(self, X, **kwargs):
        """
        Generate Z based on parameters and confounders X.
        """

        raise NotImplementedError("Subclasses must define this")

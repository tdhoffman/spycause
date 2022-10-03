__author__ = "Tyler D. Hoffman cause@tdhoffman.com"

"""
Sets up simulation classes to be reused in experiments.
"""

import numpy as np
import pandas as pd
import libpysal.weights as weights
from spopt.region import RandomRegion

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
                         - "none": no interference, same as passing None
        """

        self.Nlat = Nlat
        self.N = Nlat**2
        self.D = D
        self.sp_confound = sp_confound

        # Parse interference options
        if type(interference) == str:
            interference = interference.lower()

        if interference is not None and interference != "none":
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
        else:
            self.interference = np.eye(self.N)

    def simulate(self, treat=0.5, zconf=0.25, sp_zconf=0.25, yconf=0.5, sp_yconf=0.25,
                 interf=0.8, x_sd=1, x_sp=0.9, eps_sd=0.1, **kwargs):
        """
        Simulate data based on some parameters.
        All the conf and interf parameters could be arrays of size D
        if different variables have different levels of confounding or interference.

        Parameters
        ----------
        treat         : float, default 0.5
                        treatment effect of Z on Y
        zconf         : float, default 0.25
                        effect of nonspatial confounding on Z
        sp_zconf      : float, default 0.25
                        effect of spatial confounding on Z
        yconf         : float, default 0.5
                        effect of nonspatial confounding on Y
        sp_yconf      : float, default 0.25
                        effect of spatial confounding on Y
        interf        : float, default 0.8
                        effect of interference on Y
        x_sd          : float, default 1
                        standard deviation of confounders
        x_sp          : float, default 0.9
                        spatial autocorrelation parameter
        eps_sd        : float, default 0.1
                        SD of nonspatial error term on Y

        Returns
        -------
        X            : covariates (NxD)
        Y            : outcomes (Nx1)
        Z            : treatment (Nx1)
        """

        if np.ndim(x_sd) == 0:
            x_sd = np.repeat(x_sd, self.D)

        # Confounders
        means = np.random.choice(np.arange(-2 * self.D, 2 * self.D + 1, 1, dtype=int),
                                 size=self.D, replace=False)
        X = np.zeros((self.N, self.D))

        W = weights.lat2W(self.Nlat, self.Nlat, rook=False)
        W.transform = "r"
        W = W.full()[0]

        for d in range(self.D):
            X[:, d] = np.random.normal(loc=means[d], scale=x_sd[d], size=(self.N,))
            X[:, d] = np.dot(np.linalg.inv(np.eye(self.N) - x_sp * W), X[:, d])

        Z = np.random.binomial(1, self._create_Z(X, zconf, sp_zconf, **kwargs)).reshape(-1, 1)
        Y = self._create_Y(X, Z, treat, yconf, sp_yconf, interf, eps_sd, **kwargs)
        return X, Y, Z

    def _create_Y(self, X, Z, treat, yconf, sp_yconf, interf, eps_sd, **kwargs):
        """
        Generate Y based on parameters, confounders X, and treatment Z.
        """

        eps_y = np.random.normal(loc=0, scale=eps_sd, size=(self.N, 1))
        Y = np.dot(X, yconf) + treat * Z + eps_y

        if self.sp_confound is not None:
            Y += np.dot(np.dot(self.sp_confound, X), sp_yconf)

        Y += np.dot(np.dot(self.interference, Z), interf)

        return Y

    def _create_Z(self, X, zconf, sp_zconf, **kwargs):
        """
        Generate Z based on parameters and confounders X.
        """

        xvals = X.mean(1) / X.mean(1).max()
        if self.sp_confound is not None:
            xvals += np.dot(np.dot(self.sp_confound, xvals), sp_zconf)

        return np.clip(0.25 + xvals * zconf, 0, 1)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from rundown import Simulator

    Nlat = 30
    D = 2
    sim = Simulator(Nlat, D)
    X, Y, Z = sim.simulate(x_sp=0.9)

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

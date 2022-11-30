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
            if isinstance(interference, str) and interference == "general":
                interference = np.ones((self.N, self.N))
            elif isinstance(interference, str) and interference == "network":
                interference = weights.lat2W(Nlat, Nlat, rook=False).full()[0]
            elif isinstance(interference, int) or (isinstance(interference, str) and interference == "partial"):
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
            elif isinstance(interference, weights.W):
                interference = interference.full()[0]
            elif isinstance(interference, np.ndarray):
                pass
            else:
                raise ValueError("Unknown kind of interference")

            # Enforce row-standardization:
            interference /= interference.sum(1, keepdims=1)
            self.interference = interference
        else:
            self.interference = np.eye(self.N)

    def simulate(self, treat=0.5, zconf=0.25, sp_zconf=0.25, yconf=0.5, sp_yconf=0.25,
                 interf=0, x_sd=1, x_sp=0.9, eps_sd=0.1, **kwargs):
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
        interf        : float, default 0
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
        if np.ndim(sp_zconf) == 0:
            sp_zconf = np.repeat(sp_zconf, self.D)

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

        # Compute treated percentage
        self.treated_pct = (Z == 1).sum() / self.N
        return X, Y, Z

    def _create_Y(self, X, Z, treat, yconf, sp_yconf, interf, eps_sd, **kwargs):
        """
        Generate Y based on parameters, confounders X, and treatment Z.
        """

        eps_y = np.random.normal(loc=0, scale=eps_sd, size=(self.N, 1))

        if np.isscalar(yconf):
            yconf *= np.ones((self.D, 1))
        if np.isscalar(sp_yconf):
            sp_yconf *= np.ones((self.D, 1))

        Y = np.dot(X, yconf) + treat * Z + eps_y

        if self.sp_confound is not None:
            Y += np.dot(np.dot(self.sp_confound, X), sp_yconf)

        Y += np.dot(np.dot(self.interference, Z), interf)

        return Y

    def _create_Z(self, X, zconf, sp_zconf, **kwargs):
        """
        Generate Z based on parameters and confounders X.
        """

        # xvals = (X - X.min()) / (X.max() - X.min())
        xvals = X.mean(1) / X.mean(1).max()
        # xvals = (X - X.mean(0)) / X.std(0)
        if self.sp_confound is not None:
            xvals += np.dot(np.dot(self.sp_confound, xvals), sp_zconf)

        return np.clip(0.25 + xvals * zconf, 0, 1)


class FriedmanSimulator(Simulator):
    def __init__(self, Nlat, D, **kwargs):
        if D < 4:
            D = 4  # minimum of 5 variables required for Friedman's function (Z is one)
        super().__init__(Nlat, D, **kwargs)

    def _create_Y(self, X, Z, treat, yconf, sp_yconf, interf, eps_sd, **kwargs):
        eps_y = np.random.normal(loc=0, scale=eps_sd, size=(self.N, 1))
        Y = 10*np.sin(np.pi*X[:, [0]]*X[:, [1]]) + 20*(X[:, [2]] - 0.5)**2 + 10*X[:, [3]] + treat*Z + eps_y

        if np.isscalar(sp_yconf):
            sp_yconf *= np.ones((self.D, 1))

        if self.sp_confound is not None:
            Y += np.dot(np.dot(self.sp_confound, X), sp_yconf)

        Y += np.dot(np.dot(self.interference, Z), interf)
        return Y


if __name__ == "__main__":
    ## Imports
    import numpy as np
    import matplotlib.pyplot as plt
    from rundown import Simulator, FriedmanSimulator
    from libpysal import weights
    from spopt.region import RegionKMeansHeuristic

    Nlat = 40
    D = 2

    ## Nonspatial linear simulation (scenario 1)
    sim = Simulator(Nlat, D)
    X, Y, Z = sim.simulate()

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

    ## Nonspatial nonlinear simulation (scenario 2)
    sim = FriedmanSimulator(Nlat, D)
    X, Y, Z = sim.simulate()

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

    ## Spatially confounded linear simulation (scenario 3)
    sp_confound = weights.lat2W(Nlat, Nlat, rook=True).full()[0]
    sim = Simulator(Nlat, D, sp_confound=sp_confound)
    X, Y, Z = sim.simulate()

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

    ## Spatially confounded nonlinear simulation (scenario 4)
    sim = FriedmanSimulator(Nlat, D, sp_confound=sp_confound)
    X, Y, Z = sim.simulate()

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

    ## Linear partial spatial interference (scenario 5)
    data = np.vstack((np.hstack((np.ones((Nlat//2, Nlat//2)), 2*np.ones((Nlat//2, Nlat//2)))),
                      np.hstack((3*np.ones((Nlat//2, Nlat//2)), 4*np.ones((Nlat//2, Nlat//2))))))
    interference = np.zeros((Nlat**2, Nlat**2))

    for p in range(Nlat**2):
        i1, j1 = np.unravel_index(p, (Nlat, Nlat))
        for q in range(Nlat**2):
            i2, j2 = np.unravel_index(q, (Nlat, Nlat))
            if data[i1, j1] == data[i2, j2]:
                interference[p, q] = 1

    sim = Simulator(Nlat, D, interference=interference)
    X, Y, Z = sim.simulate(treat=0.2)

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

    ## Nonlinear partial spatial interference simulation (scenario 6)
    sim = FriedmanSimulator(Nlat, D, interference=interference)
    X, Y, Z = sim.simulate()

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

    ## Spatially confounded partial spatial interference (scenario 7)
    sim = Simulator(Nlat, D, sp_confound=sp_confound, interference=interference)
    X, Y, Z = sim.simulate(treat=0.2)

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

    ## Nonlinear spatially confounded partial spatial interference simulation (scenario 8)
    sim = FriedmanSimulator(Nlat, D, sp_confound=sp_confound, interference=interference)
    X, Y, Z = sim.simulate()

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

    ## Linear general spatial interference (scenario 9)
    sim = Simulator(Nlat, D, interference="general")
    X, Y, Z = sim.simulate(treat=0.2)

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

    ## Nonlinear general spatial interference (scenario 10)
    sim = FriedmanSimulator(Nlat, D, interference="general")
    X, Y, Z = sim.simulate(treat=0.5)

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

    ## Spatially confounded linear general spatial interference (scenario 11)
    sim = Simulator(Nlat, D, sp_confound=sp_confound, interference="general")
    X, Y, Z = sim.simulate(treat=0.2)

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

    ## Spatially confounded nonlinear general spatial interference (scenario 12)
    sim = FriedmanSimulator(Nlat, D, sp_confound=sp_confound, interference="general")
    X, Y, Z = sim.simulate(treat=0.2)

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

    ## Linear network spatial interference (scenario 13)
    sim = Simulator(Nlat, D, interference="network")
    X, Y, Z = sim.simulate()

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

    ## Nonlinear network spatial interference (scenario 14)
    sim = FriedmanSimulator(Nlat, D, interference="network")
    X, Y, Z = sim.simulate()

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

    ## Spatially confounded linear network spatial interference (scenario 15)
    sim = Simulator(Nlat, D, sp_confound=sp_confound, interference="network")
    X, Y, Z = sim.simulate()

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

    ## Spatially confounded nonlinear network spatial interference (scenario 16)
    sim = FriedmanSimulator(Nlat, D, sp_confound=sp_confound, interference="network")
    X, Y, Z = sim.simulate()

    _, axes = plt.subplots(ncols=3)
    axes[0].imshow(X[:, 0].reshape(Nlat, Nlat))
    axes[1].imshow(Y.reshape(Nlat, Nlat))
    axes[2].imshow(Z.reshape(Nlat, Nlat))
    plt.show()

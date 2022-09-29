__author__ = "Tyler D. Hoffman cause@tdhoffman.com"

"""
Sets up simulation classes to be reused in experiments.
"""

import numpy as np
import libpysal.weights


class Simulator:
    def __init__(self, N, D, linear=True, sp_confound=False):
        """
        Initialize self and parameters for simulation.
        """

        self.N = N
        self.D = D
        self.linear = linear
        self.sp_confound = sp_confound

    def simulate(self, **kwargs):
        """
        Simulate data based on some parameters.

        Returns X (NxD), Y (Nx1), and Z (Nx1).
        """

        X = self._create_X(**kwargs)
        Z = self._create_Z(X, **kwargs)
        Y = self._create_Y(X, Z, **kwargs)
        return X, Y, Z

    def _create_X(self, **kwargs):
        """
        Generate X based on parameters.
        """

        raise NotImplementedError("Subclasses must define this")

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

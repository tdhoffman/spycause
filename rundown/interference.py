__author__ = "Tyler D. Hoffman cause@tdhoffman.com"

"""
Spatial interference adjustments for causal inference.
Basically just include a lag of the treatment variables
This should be a transformer!!!!
"""

import numpy as np

class InterferenceAdjuster:
    def __init__(self, Z, w):
        Z = np.hstack((w @ Z))

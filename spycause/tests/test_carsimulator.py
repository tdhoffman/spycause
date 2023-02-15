import os
os.chdir('../..')
import rundown
import numpy as np
import libpysal.weights as w
import matplotlib.pyplot as plt


# Set up
weights = w.lat2W(30, 30, rook=False)
sim = rundown.CARSimulator(30, 1, sp_confound=weights)
X, Y, Z = sim.simulate()

_, axes = plt.subplots(ncols=3)
axes[0].imshow(X.reshape(30, 30))
axes[1].imshow(Z.reshape(30, 30))
axes[2].imshow(Y.reshape(30, 30))
plt.show()


# Add interference
intweights = w.lat2W(30, 30)
intsim = rundown.CARSimulator(30, 1, sp_confound=weights, interference=intweights)
X, Y, Z = intsim.simulate()

_, axes = plt.subplots(ncols=3)
axes[0].imshow(X.reshape(30, 30))
axes[1].imshow(Z.reshape(30, 30))
axes[2].imshow(Y.reshape(30, 30))
plt.show()

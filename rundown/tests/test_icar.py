import os
os.chdir('../..')

import numpy as np
import scipy.sparse as sp
import rundown as rd
from libpysal.weights import lat2W

## Set up parameters
Nlat = 30
N = Nlat**2
D = 2
x_sd = 0.75
y_sd = 0.1
beta = np.array([[0.5, -1]]).T
tau = 2
ucar_sd = 1
vcar_sd = 1
rho = 0.9
W = lat2W(Nlat, Nlat)

## Generate data from CARSimulator
sim = rd.CARSimulator(Nlat, D, sp_confound=W)
X, Y, Z = sim.simulate(treat=tau, y_conf=beta, x_sd=x_sd, y_sd=y_sd, ucar_sd=ucar_sd, vcar_sd=vcar_sd,
                       ucar_str=rho, vcar_str=rho)

## Fit model
model = rd.ICAR(w=W, fit_intercept=False)
model = model.fit(X, Y, Z, nsamples=4000)

## Results
print(model.ate_)
print(model.coef_)
print(model.indir_coef_)
print(model.score(X, Y, Z))  # R^2 doesn't work yet

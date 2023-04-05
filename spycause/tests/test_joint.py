import os
os.chdir("../..")

import numpy as np
import scipy.sparse as sp
import spycause as spy
from libpysal.weights import lat2W

## Set up parameters
Nlat = 30
N = Nlat**2
D = 2
x_sd = 0.5
y_sd = 0.1
beta = np.array([[0.5, -1]]).T
tau = 2
ucar_sd = 1
vcar_sd = 1
rho = 0.9
W = lat2W(Nlat, Nlat)
W.transform = "r"

## Generate data from CARSimulator
sim = spy.CARSimulator(Nlat, D, sp_confound=W)
X, Y, Z = sim.simulate(treat=tau, y_conf=beta, x_sd=x_sd, y_sd=y_sd, ucar_sd=ucar_sd, vcar_sd=vcar_sd,
                       ucar_str=rho, vcar_str=rho)
W.transform = "o"

## Fit model
model = spy.Joint(w=W, fit_intercept=False)
model = model.fit(X, Y, Z, nsamples=4000, nwarmup=1000)

## Results
print(model.ate_)
print(model.coef_)
print(model.waic())
print(model.diagnostics())


## Add interference
interf_eff = 10
Wint = lat2W(Nlat, Nlat, rook=False)
Wint.transform = 'r'
intsim = spy.CARSimulator(Nlat, D, interference=Wint)
X, Y, Z = intsim.simulate(treat=tau, y_conf=beta, x_sd=x_sd, y_sd=y_sd, ucar_sd=ucar_sd, vcar_sd=vcar_sd,
                          ucar_str=rho, vcar_str=rho)

## Interference adjustment
intadj = spy.InterferenceAdj(w=Wint)
Zint = intadj.transform(Z)

W.transform = "o"

## Estimate
intmodel = spy.Joint(w=W, fit_intercept=False)
intmodel = intmodel.fit(X, Y, Zint, nsamples=3000, save_warmup=False)
nointmodel = spy.Joint(w=W, fit_intercept=False)
nointmodel = nointmodel.fit(X, Y, Z, nsamples=3000, save_warmup=False)

print(intmodel.ate_)
print(intmodel.coef_)
print(nointmodel.ate_)
print(nointmodel.coef_)

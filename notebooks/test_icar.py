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
x_sd = 0.5
y_sd = 0.1
beta = np.array([[0.5, -1]]).T
tau = 2
ucar_sd = 1
vcar_sd = 0
rho = 0.9
W = lat2W(Nlat, Nlat)
W.transform = "r"

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
print(model.waic())
print(model.diagnostics())


## Add nonspatial prop score preprocessing
propadj = rd.PropEst(bs_df=5)
pi_hat = propadj.fit_transform(X, Z)

## Fit model
propmodel = rd.ICAR(w=W, fit_intercept=False)
propmodel = propmodel.fit(pi_hat, Y, Z, nsamples=4000)

## Results
print(propmodel.ate_)
print(propmodel.coef_)
print(propmodel.waic())

## Add spatial prop score preprocessing
propadj = rd.PropEst(w=W, bs_df=5)
pi_hat = propadj.fit_transform(X, Z)

## Fit model
propmodel = rd.ICAR(w=W, fit_intercept=False)
propmodel = propmodel.fit(pi_hat, Y, Z, nsamples=4000)

## Results
print(propmodel.ate_)
print(propmodel.coef_)
print(propmodel.waic())

## Add interference
# not really doing too much to the problem -- i bet because it's too similar to W
# actually i bet it's because i didn't put it in the signature!
interf_eff = 10
Wint = lat2W(Nlat, Nlat, rook=False)
Wint.transform = 'r'
intsim = rd.CARSimulator(Nlat, D, interference=Wint)
X, Y, Z = intsim.simulate(treat=tau, y_conf=beta, x_sd=x_sd, y_sd=y_sd, ucar_sd=ucar_sd, vcar_sd=vcar_sd,
                          ucar_str=rho, vcar_str=rho)

## Interference adjustment
intadj = rd.InterferenceAdj(w=Wint)
Zint = intadj.transform(Z)

## Estimate
intmodel = rd.ICAR(w=W, fit_intercept=False)
intmodel = intmodel.fit(X, Y, Zint, nsamples=3000, save_warmup=False)
nointmodel = rd.ICAR(w=W, fit_intercept=False)
nointmodel = nointmodel.fit(X, Y, Z, nsamples=3000, save_warmup=False)

print(intmodel.ate_)
print(intmodel.coef_)
print(nointmodel.ate_)
print(nointmodel.coef_)
print(intmodel.waic())
print(nointmodel.waic())

## Add nonspatial prop score preprocessing and interference
propadj = rd.PropEst(bs_df=5)
pi_hat = propadj.fit_transform(X, Z)

## Interference adjustment
intadj = rd.InterferenceAdj(w=Wint)
Zint = intadj.transform(Z)

## Estimate
intmodel = rd.ICAR(w=W, fit_intercept=False)
intmodel = intmodel.fit(X, Y, Zint, nsamples=3000, save_warmup=False)
print(intmodel.ate_)
print(intmodel.coef_)
print(intmodel.waic())

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
beta = np.array([[1, 1]]).T
tau = 0.5
ucar_sd = 1
vcar_sd = 1
rho = 0.9
W = lat2W(Nlat, Nlat)

## Generate data from CARSimulator
sim = spy.CARSimulator(Nlat, D, sp_confound=W)
X, Y, Z = sim.simulate(treat=tau, y_conf=beta, x_sd=x_sd, y_sd=y_sd, ucar_sd=ucar_sd, vcar_sd=vcar_sd,
                       ucar_str=rho, vcar_str=0, balance=0.5)

## Fit model
model = spy.CAR(w=W, fit_intercept=False)
model = model.fit(X, Y, Z, nsamples=4000, nwarmup=1000, save_warmup=False)

## Results
print(model.ate_)
print(model.coef_)
print(model.indir_coef_)
print(model.diagnostics())

## Add nonspatial prop score preprocessing
propadj = spy.PropEst(bs_df=5)
pi_hat = propadj.fit_transform(X, Z)

## Fit model
propmodel = spy.CAR(w=W, fit_intercept=False)
propmodel = propmodel.fit(pi_hat, Y, Z)

## Results
print(propmodel.ate_)
print(propmodel.coef_)
print(propmodel.indir_coef_)
print(propmodel.diagnostics())

## Add interference
interf_eff = 10
Wint = lat2W(Nlat, Nlat, rook=False)
Wint.transform = 'r'
intsim = spy.CARSimulator(Nlat, D, sp_confound=W, interference=Wint)
X, Y, Z = intsim.simulate(treat=tau, y_conf=beta, x_sd=x_sd, y_sd=y_sd, ucar_sd=ucar_sd, vcar_sd=vcar_sd,
                          ucar_str=rho, vcar_str=rho)

W.transform = "o"

## Interference adjustment
intadj = spy.InterferenceAdj(w=Wint)
Zint = intadj.transform(Z)

## Fit model
model = spy.CAR(w=W, fit_intercept=False)
model = model.fit(X, Y, Zint)

## Results
print(model.ate_)
print(model.coef_)
print(model.indir_coef_)
print(model.diagnostics())

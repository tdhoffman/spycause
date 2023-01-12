os.chdir('../..')

import numpy as np
import rundown as rd
from libpysal.weights import lat2W
from sklearn.pipeline import Pipeline

## Set up parameters
Nlat = 30
N = Nlat**2
D = 2
x_sd = 0.75
y_sd = 0.1
beta = np.array([[0.5, -1]]).T
tau = 2

## Generate data
sim = rd.CARSimulator(Nlat, D)
X, Y, Z = sim.simulate(treat=tau, y_conf=beta, x_sd=x_sd, y_sd=y_sd)

## Fit model with covariates
model = rd.BayesOLS(fit_intercept=False)
model = model.fit(X, Y, Z, save_warmup=False)

## Fit model with prop score

## Results
print(model.ate_)
print(model.coef_)
print(model.score(X, Y, Z))
print(model.waic())

## Add interference
interf_eff = 10
W = lat2W(Nlat, Nlat, rook=False)
W.transform = 'r'
intsim = rd.CARSimulator(Nlat, D, interference=W)
X, Y, Z = intsim.simulate(treat=tau, y_conf=beta, x_sd=x_sd, y_sd=y_sd, interf=interf_eff)

## Interference adjustment
# pipe = Pipeline([("interference", rd.InterferenceAdj()), ("model", rd.BayesOLS)])
intadj = rd.InterferenceAdj(w=W)
Zint = intadj.transform(Z)

# Interference is barely being estimated
intmodel = rd.BayesOLS(fit_intercept=False)
intmodel = intmodel.fit(X, Y, Zint, nsamples=3000, save_warmup=False)
nointmodel = rd.BayesOLS(fit_intercept=False)
nointmodel = nointmodel.fit(X, Y, Z, nsamples=3000, save_warmup=False)

## Results
# nointmodel has bias in tau and the betas are definitely off (all by about 0.2)
# intmodel gets everything right
print(intmodel.ate_)
print(intmodel.coef_)
print(nointmodel.ate_)
print(nointmodel.coef_)
print(intmodel.score(X, Y, Zint))
print(nointmodel.score(X, Y, Z))
print(intmodel.waic())
print(nointmodel.waic())

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
sim = rd.Simulator(Nlat, D)
X, Y, Z = sim.simulate(treat=tau, yconf=beta, x_sd=x_sd, eps_sd=y_sd)

## Fit model
model = rd.BayesOLS(fit_intercept=False)
model = model.fit(X, Y, Z, save_warmup=False)

## Results
print(model.ate_)
print(model.coef_)
print(model.score(X, Y, Z))

## Interference adjustment
pipe = Pipeline([("interference", rd.InterferenceAdj()), ("model", rd.BayesOLS)])

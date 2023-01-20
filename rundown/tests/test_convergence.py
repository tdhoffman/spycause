os.chdir('../..')
import numpy as np
import rundown as rd
import arviz as az
from libpysal.weights import lat2W
from sklearn.pipeline import Pipeline

## Set up parameters
Nlat = 30
N = Nlat**2
D = 2
x_sd = 0.5
y_sd = 0.1
beta = np.array([[0.5, -1]]).T
tau = 2
ucar_sd = 0
vcar_sd = 0
ucar_str = 0
vcar_str = 0
balance = 0
W = lat2W(Nlat, Nlat)
W.transform = "r"
interf_eff = 10
WI = lat2W(Nlat, Nlat, rook=False)
WI.transform = 'r'

## ------- OLS MODEL   ------- ##
## Generate data
sim = rd.CARSimulator(Nlat, D)
X, Y, Z = sim.simulate(treat=tau, y_conf=beta, x_sd=x_sd, y_sd=y_sd, ucar_sd=ucar_sd, vcar_sd=vcar_sd,
                       ucar_str=ucar_str, vcar_str=vcar_str, balance=balance)

## Fit model
model = rd.BayesOLS(fit_intercept=False)
model = model.fit(X, Y, Z)

# Report diagnostics
model.diagnostics()

## Add nonspatial prop score preprocessing
propadj = rd.PropEst()
pi_hat = propadj.fit_transform(X, Z)

# Fit model
propmodel = rd.BayesOLS(fit_intercept=False)
propmodel = propmodel.fit(pi_hat, Y, Z)

# Report diagnostics
propmodel.diagnostics()

## Generate interfered data
intsim = rd.CARSimulator(Nlat, D, interference=WI)
X, Y, Z = intsim.simulate(treat=tau, y_conf=beta, x_sd=x_sd, y_sd=y_sd, interf=interf_eff)

# Fit model
intadj = rd.InterferenceAdj(w=WI)
Zint = intadj.transform(Z)
intmodel = rd.BayesOLS(fit_intercept=False)
intmodel = intmodel.fit(X, Y, Zint)

# Report diagnostics
intmodel.diagnostics()

## Add prop score to interfered data
propadj = rd.PropEst()
pi_hat = propadj.fit_transform(X, Z)
intadj = rd.InterferenceAdj(w=WI)
Zint = intadj.transform(Z)
propmodel = rd.BayesOLS(fit_intercept=False)
propmodel = propmodel.fit(pi_hat, Y, Zint)

# Report diagnostics
propmodel.diagnostics()

## ------- ICAR MODEL  ------- ##
ucar_sd = 1
ucar_str = 0.9
## Generate data
sim = rd.CARSimulator(Nlat, D, sp_confound=W)
X, Y, Z = sim.simulate(treat=tau, y_conf=beta, x_sd=x_sd, y_sd=y_sd, ucar_sd=ucar_sd, vcar_sd=vcar_sd,
                       ucar_str=ucar_str, vcar_str=vcar_str, balance=balance)

## Fit model
model = rd.ICAR(w=W, fit_intercept=False)
model = model.fit(X, Y, Z, nsamples=4000)

## Report diagnostics
model.diagnostics()

## Add nonspatial prop score preprocessing
propadj = rd.PropEst()
pi_hat = propadj.fit_transform(X, Z)

# Fit model
propmodel = rd.ICAR(w=W, fit_intercept=False)
propmodel = propmodel.fit(pi_hat, Y, Z)

# Report diagnostics
propmodel.diagnostics()

## Generate interfered data
intsim = rd.CARSimulator(Nlat, D, interference=WI)
X, Y, Z = intsim.simulate(treat=tau, y_conf=beta, x_sd=x_sd, y_sd=y_sd, ucar_sd=ucar_sd, vcar_sd=vcar_sd,
                          ucar_str=ucar_str, vcar_str=vcar_str, balance=balance, interf=interf_eff)

# Fit model
intadj = rd.InterferenceAdj(w=WI)
Zint = intadj.transform(Z)
intmodel = rd.ICAR(w=W, fit_intercept=False)
intmodel = intmodel.fit(X, Y, Zint)

# Report diagnostics
intmodel.diagnostics()

## Add prop score to interfered data
propadj = rd.PropEst()
pi_hat = propadj.fit_transform(X, Z)
intadj = rd.InterferenceAdj(w=WI)
Zint = intadj.transform(Z)
propmodel = rd.ICAR(w=W, fit_intercept=False)
propmodel = propmodel.fit(pi_hat, Y, Zint)

# Report diagnostics
propmodel.diagnostics()

## ------- JOINT MODEL ------- ##
vcar_sd = 1
vcar_str = 0.9
balance = 0.5
## Generate data
sim = rd.CARSimulator(Nlat, D, sp_confound=W)
X, Y, Z = sim.simulate(treat=tau, y_conf=beta, x_sd=x_sd, y_sd=y_sd, ucar_sd=ucar_sd, vcar_sd=vcar_sd,
                       ucar_str=ucar_str, vcar_str=vcar_str, balance=balance)

## Fit model
model = rd.Joint(w=W, fit_intercept=False)
model = model.fit(X, Y, Z)

## Report diagnostics
model.diagnostics()

## Add nonspatial prop score preprocessing
propadj = rd.PropEst()
pi_hat = propadj.fit_transform(X, Z)

# Fit model
propmodel = rd.Joint(w=W, fit_intercept=False)
propmodel = propmodel.fit(pi_hat, Y, Z)

# Report diagnostics
propmodel.diagnostics()

## Generate interfered data
intsim = rd.CARSimulator(Nlat, D, interference=WI)
X, Y, Z = intsim.simulate(treat=tau, y_conf=beta, x_sd=x_sd, y_sd=y_sd, ucar_sd=ucar_sd, vcar_sd=vcar_sd,
                          ucar_str=ucar_str, vcar_str=vcar_str, balance=balance, interf=interf_eff)

# Fit model
intadj = rd.InterferenceAdj(w=WI)
Zint = intadj.transform(Z)
intmodel = rd.Joint(w=W, fit_intercept=False)
intmodel = intmodel.fit(X, Y, Zint)

# Report diagnostics
intmodel.diagnostics()

## Add prop score to interfered data
propadj = rd.PropEst()
pi_hat = propadj.fit_transform(X, Z)
intadj = rd.InterferenceAdj(w=WI)
Zint = intadj.transform(Z)
propmodel = rd.Joint(w=W, fit_intercept=False)
propmodel = propmodel.fit(pi_hat, Y, Zint)

# Report diagnostics
propmodel.diagnostics()

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
sd_u = 2
rho = 0.9

W = lat2W(Nlat, Nlat)
rowsums = np.fromiter(W.cardinalities.values(), dtype=float)
cov_u = (sd_u**2)*sp.linalg.spsolve(sp.diags(rowsums) - rho*W.sparse, np.eye(N))
car_u = np.linalg.cholesky(cov_u)

## Generate data
# Generate X
# X = np.random.normal(loc=0, scale=x_sd, size=(N, D))

# # Generate Z as a function of X
# prop_scores = np.abs(X.sum(1).reshape(-1, 1)) / np.abs(X.sum(1)).max()
# Z = np.random.binomial(1, p=prop_scores, size=(N, 1))

# # Generate Y, adding UNOBSERVED spatial confounding
# Y = np.dot(X, beta) + np.dot(Z, tau) + np.dot(car_u, np.random.normal(size=(N, 1))) + \
                    # + np.random.normal(loc=0, scale=y_sd, size=(N, 1))

## Generate data from Simulator
sim = rd.Simulator(Nlat, D, sp_confound=W)
X, Y, Z = sim.simulate(treat=tau, yconf=beta, x_sd=x_sd, eps_sd=y_sd, sp_yconf=rho, sp_zconf=rho)

## Fit model
model = rd.ICAR(w=W, fit_intercept=False)
model = model.fit(X, Y, Z)

## Results
print(model.ate_)
print(model.coef_)
print(model.indir_coef_)
print(model.score(X, Y, Z))  # R^2 doesn't work yet

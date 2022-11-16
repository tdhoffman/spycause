import numpy as np
import rundown as rd
from libpysal.weights import lat2W

## Set up parameters
N = 30
D = 2
sd_X = 0.75
sd_Y = 0.1
beta = np.array([[0.5, -1]]).T
tau = 2

## Generate data
# Generate X
X = np.random.normal(loc=0, scale=sd_X, size=(N, D))

# Generate Z as a function of X
prop_scores = np.abs(X.sum(1).reshape(-1, 1)) / np.abs(X.sum(1)).max()
Z = np.random.binomial(1, p=prop_scores, size=(N, 1))

# Generate Y
Y = np.dot(X, beta) + np.dot(Z, tau) + np.random.normal(loc=0, scale=sd_Y, size=(N, 1))

## Fit model
model = rd.BayesOLS(fit_intercept=False)
model = model.fit(X, Y, Z)

## Results
print(model.ate_)
print(model.coef_)
print(model.score(X, Y, Z))

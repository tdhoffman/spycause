import pytensor.tensor as pt
import numpy as np
import pymc as pm


def linear_regression(X, Y, Z):
    N, D = X.shape
    with pm.Model() as model:
        # Priors
        beta = pm.Normal("beta", mu=0, sigma=5, shape=(D,))
        tau = pm.Normal("tau", mu=0, sigma=5)
        sigma = pm.Exponential("sigma", lam=1)

        mu = X @ beta + Z @ tau

        # Likelihood
        likelihood = pm.Normal("Y", mu=mu, sigma=sigma, observed=Y)

    return model


with linear_regression(X, Y, Z) as model:
    idata = pm.sample( chains=1)


def car(X, Y, Z, W):
    # W is unstandardized weights matrix
    N, D = X.shape
    W = W.full()[0]
    Dmat = np.diag(W.sum(axis=1))

    with pm.Model() as model:
        # Priors
        beta = pm.Normal("beta", mu=0, sigma=5, shape=(X.shape[1],))
        tau = pm.Normal("tau", mu=0, sigma=5, shape=(Z.shape[1],))
        alpha = pm.Uniform("alpha", lower=0, upper=1)
        sigma = pm.Exponential("sigma", lam=1)

        # CAR prior
        tau_u = pm.Gamma("tau_u", alpha=2, beta=2)
        u = pm.MvNormal("u", mu=0, tau=tau_u * (Dmat - alpha*W), shape=(N, 1))

        mu = X @ beta + Z @ tau + u
        likelihood = pm.Normal("Y", mu=mu, sigma=sigma, observed=Y)

    return model


def icar(X, Y, Z, W):
    N, D = X.shape

    Dmat = np.diag(W.full()[0].sum(axis=1))

    node1 = W.to_adjlist()['focal'].values
    node2 = W.to_adjlist()['neighbor'].values
    weights = W.to_adjlist()['weight'].values

    Q = Dmat - W.full()[0]

    with pm.Model() as model:
        # Priors
        beta = pm.Normal("beta", mu=0, sigma=5, shape=D)
        tau = pm.Normal("tau", mu=0, sigma=5)
        sigma = pm.Exponential("sigma", lam=1)

        # tau_u = pm.Gamma("tau_u", alpha=2, beta=2)
        # u_diff = u[node1] - u[node2]
        # u = pm.Potential("u", -0.5 * weights * (u_diff @ u_diff))
        # pm.Potential('soft_sum', pm.Normal.dist(0, s*n).logp(eta.sum()))

        tau_u = pm.Gamma("tau_u", alpha=1, beta=1)
        sd_u = pm.Deterministic("sigma_u", 1/pt.sqrt(tau_u))
        u = pm.DensityDist("u", logp=lambda value: -0.5 * pt.dot(pt.transpose(value), pt.dot(Q, value)))
        # u = pm.Flat("u", shape=N)
        # pm.Potential("spatial_diff", -0.5 * weights * pt.dot(u[node1] - u[node2], u[node1] - u[node2]))
        pm.Potential("soft_sum", pm.logp(pm.Normal.dist(0, 0.001*N), u.sum()))
        # u = ICAR("u", tau=tau_u, Q=(Dmat - W))

        mu = pm.Deterministic("mu", pt.dot(X, beta) + pt.dot(Z, tau) + sd_u * u)
        likelihood = pm.Normal("Y", mu=mu, sigma=sigma, observed=Y)

    return model

def bym(x, y):
    with pm.Model() as model:
        # precision priors 
        tau_theta = pm.Gamma("tau_theta", alpha=3.2761, beta=1.81)
        tau_phi = pm.Gamma("tau_phi", alpha=1, beta=1)
        # transform from precision to standard deviation
        sigma_theta = pm.Deterministic("sigma_theta", 1/at.sqrt(tau_theta))
        sigma_phi = pm.Deterministic("sigma_phi", 1/at.sqrt(tau_phi))

        # independent random effect prior, constraining to mean 0
        theta = pm.Normal("theta", mu=0, sigma=1, dims="num_areas")
        theta_cons = pm.Deterministic("theta_cons", theta-at.mean(theta), dims="num_areas")
        # spatial ICAR random effect prior
        phi = pm.Flat("phi", dims="num_areas")
        pm.Potential("spatial_diff", pairwise_diff(phi, node1, node2)) # `pairwise_diff` denotes to equivalent sigma=1 prior
        phi_cons = pm.Deterministic("phi_cons", phi-at.mean(phi), dims="num_areas")

        # regression coefficient priors 
        beta0 = pm.Normal("beta0", mu=0, sigma=5)
        beta1 = pm.Normal("beta1", mu=0, sigma=5)

        # linear predictor
        eta = pm.Deterministic("eta", logE + beta0 + beta1*x + phi_cons*sigma_phi + theta_cons*sigma_theta, dims="num_areas") 

        # likelihood
        obs = pm.Poisson("obs", at.exp(eta), observed=y, dims="num_areas")

class ICAR(pm.Continuous):
    def __init__(self, tau, Q, *args, **kwargs):
        super(ICAR, self).__init__(*args, **kwargs)
        self.tau = tau
        self.Q = Q
        self.mode = 0.

    def logp(self, x):
        tau = self.tau
        Q = self.Q
        return -0.5 * tau * tt.dot(tt.transpose(x), tt.dot(Q, x))


with icar(X, Y, Z, W) as model:
    idata = pm.sample(chains=1)

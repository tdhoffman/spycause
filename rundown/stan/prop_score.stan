data {
  int N;           // number of observation units
  int D;           // number of confounding variables
  vector[N] Y;     // outcome variable
  matrix[N, D] X;  // confounders
  vector[N] Z;     // treatment variable

  // CAR model stuff
  int<lower=0> N_edges;                  // number of edges
  int<lower=1, upper=N> node1[N_edges];  // node1[i] is adjacent to node2[i]
  int<lower=1, upper=N> node2[N_edges];  // and node1[i] < node2[i]
}

parameters {
  vector[D] beta;       // effects of confounders on outcome
  vector[D] alpha;      // effects of confounders on treatment
  real tau;             // effect of treatment on outcome
  vector[N] u;          // spatial effects in outcome model
  vector[N] v;          // spatial effects in treatment model
  real<lower=0> sd_u;   // SD of CAR term for outcome model
  real<lower=0> sd_v;   // SD of CAR term for treatment model
  real<lower=0> sigma;  // SD of outcome
  real psi;             // shared spatial random effect on outcome
}

transformed parameters {
  vector<lower=0, upper=1>[N] pi = inv_logit(X*alpha + sd_v*v);
  real<lower=0> tau_u = 1/sd_u^2;
  real<lower=0> tau_v = 1/sd_v^2;
}

model {
  // Likelihoods
  Z ~ bernoulli(pi);
  Y ~ normal(Z*tau + X*beta + sd_u*u + psi*sd_v*v, sigma);

  // Priors
  alpha ~ normal(0, 5);
  beta ~ normal(0, 5);
  tau_u ~ gamma(3.2761, 1.81);  // Carlin WinBUGS priors on the CAR terms
  tau_v ~ gamma(3.2761, 1.81);
  
  // CAR priors
  target += -0.5 * dot_self(u[node1] - u[node1]);
  target += -0.5 * dot_self(v[node1] - v[node2]);

  // Soft sum-to-zero constraint on spatial effects
  sum(u) ~ normal(0, 0.001*N);
  sum(v) ~ normal(0, 0.001*N);
}


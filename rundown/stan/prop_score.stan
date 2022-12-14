data {
  int N;           // number of observation units
  int D;           // number of confounding variables
  int K;           // 2 if interference-adjusted, 1 otherwise
  vector[N] Y;     // outcome variable
  matrix[N, D] X;  // confounders
  int Z[N];        // treatment variable
  matrix[N, N] W;  // weights matrix

  // CAR model stuff
  int<lower=0> N_edges;                  // number of edges
  int<lower=1, upper=N> node1[N_edges];  // node1[i] is adjacent to node2[i]
  int<lower=1, upper=N> node2[N_edges];  // and node1[i] < node2[i]
}

transformed data {
  vector[N] Zvec = to_vector(Z);
}

parameters {
  vector[D] beta;       // effects of confounders on outcome
  vector[D] alpha;      // effects of confounders on treatment
  vector[K] tau;        // effect of treatment on outcome
  vector[N] u;          // spatial effects in outcome model
  vector[N] v;          // spatial effects in treatment model
  real<lower=0> sd_u;   // SD of ICAR term for outcome model
  real<lower=0> sd_v;   // SD of ICAR term for treatment model
  real<lower=0> sigma2; // var of outcome
  real psi;             // shared spatial random effect on outcome
}

transformed parameters {
  real<lower=0, upper=1> pi[N] = to_array_1d(inv_logit(X*alpha + sd_v*v));
  real sigma = sqrt(sigma2);
  real<lower=0> tau_u = 1/sd_u^2;
  real<lower=0> tau_v = 1/sd_v^2;
}

model {
  // Likelihoods
  Z ~ bernoulli(pi);
  if (K == 2) {
    vector[N] Zlag = W * Zvec;
    Y ~ normal(Zvec*tau[1] + Zlag*tau[2] + X*beta + sd_u*u + psi*sd_v*v, sigma);
  } else {
    Y ~ normal(Zvec*tau[1] + X*beta + sd_u*u + psi*sd_v*v, sigma);
  }

  // Priors
  alpha ~ normal(0, 5);
  beta ~ normal(0, 5);
  psi ~ normal(0, 0.1);
  sigma2 ~ inv_gamma(0.5, 0.005);
  tau_u ~ gamma(3.2761, 1.81);  // Carlin WinBUGS priors on the ICAR terms
  tau_v ~ gamma(3.2761, 1.81);
  
  // CAR priors
  target += -0.5 * dot_self(u[node1] - u[node1]);
  target += -0.5 * dot_self(v[node1] - v[node2]);

  // Soft sum-to-zero constraint on spatial effects
  sum(u) ~ normal(0, 0.001*N);
  sum(v) ~ normal(0, 0.001*N);
}


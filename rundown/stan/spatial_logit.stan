data {
  int<lower=1> N;  // number of observations
  int<lower=1> D;  // number of features
  matrix[N, D] X;  // confounding variables
  int Z[N];  // inputted treatment variable

  // ICAR stuff
  int<lower=1> N_edges;                  // number of edges
  int<lower=1, upper=N> node1[N_edges];  // node1[i] adjacent to node2[i]
  int<lower=1, upper=N> node2[N_edges];  // and node1[i] < node2[i]
}

parameters {
  vector[D] alpha;  // covariate effects

  // ICAR effects
  vector[N] v;   // spatially structured residuals
  real<lower=0> sd_v;  // sd of ICAR effects
}

transformed parameters {
  real<lower=0, upper=1> pi[N] = to_array_1d(inv_logit(X*alpha + sd_v*v));
  real<lower=0> tau_v = inv(sqrt(sd_v));
}

model {
  Z ~ bernoulli(pi);

  alpha ~ normal(0, 5);
  tau_v ~ gamma(0.5, 0.005);

  // ICAR prior
  target += -0.5 * dot_self(v[node1] - v[node2]);
  sum(v) ~ normal(0, 0.001*N);  // equivalent to mean(v) ~ normal(0, 0.001)
}

generated quantities {
  vector[N] log_likelihood;
  vector[N] pi_hat;
  for (i in 1:N) {
    log_likelihood[i] = bernoulli_lpmf(Z[i] | pi[i]);
    pi_hat[i] = bernoulli_rng(pi[i]);
  }
}
  

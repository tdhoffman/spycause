data {
  int<lower=1> N;  // number of observations
  int<lower=1> D;  // number of features
  int          K;  // 2 if interference-adjusted, 1 otherwise
  matrix[N, D] X;  // confounding variables
  vector[N]    y;  // outcome variable
  matrix[N, K] Z;  // inputted treatment variable

  // ICAR stuff
  int<lower=1> N_edges;                  // number of edges
  array[N_edges] int<lower=1, upper=N> node1;  // node1[i] adjacent to node2[i]
  array[N_edges] int<lower=1, upper=N> node2;  // and node1[i] < node2[i]
  vector[N_edges] weights;               // weight for the edge
}

parameters {
  vector[D] beta;  // covariate effects
  vector[K] tau;   // treatment effects (possibly including lag)
  real<lower=0> sigma;  // SD of outcome

  // ICAR effects
  vector[N] u;   // spatially structured residuals
  real<lower=0> sd_u;  // sd of ICAR effects
}

transformed parameters {
  vector[N] mu = X*beta + Z*tau + sd_u*u;
  real<lower=0> tau_u = inv(sqrt(sd_u));
}

model {
  y ~ normal(mu, sigma);

  tau ~ normal(0, 5);
  beta ~ normal(0, 5);
  sigma ~ exponential(1);
  tau_u ~ gamma(0.5, 0.005);

  // ICAR prior
  target += -0.5 * weights * dot_self(u[node1] - u[node2]);
  sum(u) ~ normal(0, 0.001*N);  // equivalent to mean(rho) ~ normal(0, 0.001)
}

generated quantities {
  vector[N] log_likelihood;
  vector[N] y_pred;
  for (i in 1:N) {
    log_likelihood[i] = normal_lpdf(y[i] | mu[i], sigma);
    y_pred[i] = normal_rng(mu[i], sigma);
  }
}

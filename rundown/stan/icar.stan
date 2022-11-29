data {
  int<lower=1> N;  // number of observations
  int<lower=1> D;  // number of features
  int          K;  // 2 if interference-adjusted, 1 otherwise
  matrix[N, D] X;  // confounding variables
  vector[N]    y;  // outcome variable
  matrix[N, K] Z;  // inputted treatment variable

  // ICAR stuff
  int<lower=1> N_edges;                        // number of edges
  int<lower=1, upper=N_edges> node1[N_edges];  // node1[i] adjacent to node2[i]
  int<lower=1, upper=N_edges> node2[N_edges];  // and node1[i] < node2[i]
}

parameters {
  vector[D] beta;  // covariate effects
  vector[K] tau;   // treatment effects (possibly including lag)
  real sigma2;     // variance of outcome

  // ICAR effects
  vector[N] phi;   // spatially structured residuals
  real<lower=0> sd_p;  // sd of ICAR effects
}

transformed parameters {
  real sigma = sqrt(sigma2);  // SD, required for parametrizing the likelihood
}

model {
  y ~ normal(X*beta + Z*tau + sd_phi*phi, sigma);

  tau ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma2 ~ inv_gamma(0.5, 0.005);
  sd_phi ~ gamma(3.2761, 1.81);  // Carlin WinBUGS prior on the ICAR term

  // ICAR prior
  target += -0.5 * dot_self(phi[node1] - phi[node2]);
  sum(phi) ~ normal(0, 0.001*N);  // equivalent to mean(phi) ~ normal(0, 0.001)
}

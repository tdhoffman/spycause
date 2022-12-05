data {
  int<lower=1> N;  // number of observations
  int<lower=1> D;  // number of features
  int          K;  // 2 if interference-adjusted, 1 otherwise
  matrix[N, D] X;  // confounding variables
  vector[N]    y;  // outcome variable
  matrix[N, K] Z;  // inputted treatment variable

  // ICAR stuff
  int<lower=1> N_edges;                  // number of edges
  int<lower=1, upper=N> node1[N_edges];  // node1[i] adjacent to node2[i]
  int<lower=1, upper=N> node2[N_edges];  // and node1[i] < node2[i]
}

parameters {
  vector[D] beta;  // covariate effects
  vector[K] tau;   // treatment effects (possibly including lag)
  real<lower=0> sigma;  // SD of outcome

  // ICAR effects
  vector[N] rho;   // spatially structured residuals
  real<lower=0> sd_r;  // sd of ICAR effects
}

model {
  y ~ normal(X*beta + Z*tau + sd_r*rho, sigma);

  tau ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ exponential(1);
  sd_r ~ gamma(3.2761, 1.81);  // Carlin WinBUGS prior on the ICAR term

  // ICAR prior
  target += -0.5 * dot_self(rho[node1] - rho[node2]);
  sum(rho) ~ normal(0, 0.001*N);  // equivalent to mean(rho) ~ normal(0, 0.001)
}

data {
  int<lower=1> N;  // number of observations
  int<lower=1> D;  // number of features
  matrix[N, D] X;  // confounding variables
  vector[N]    y;  // outcome variable
  vector[N]    Z;  // inputted treatment variable
  vector[N]    A;  // instrument

  // ICAR stuff
  int<lower=1> N_edges;                        // number of edges
  int<lower=1, upper=N_edges> node1[N_edges];  // node1[i] adjacent to node2[i]
  int<lower=1, upper=N_edges> node2[N_edges];  // and node1[i] < node2[i]
}

parameters {
  vector[D] beta;  // covariate effects on outcome
  real tau;        // treatment effects (not including lag)
  vector[D] lamb;  // covariate effects on treatment
  real gamma;      // instrument effect on treatment
  
  real sigma2y;    // variance of outcome
  real sigma2z;    // variance of treatment

  // ICAR effects
  vector[N] u;     // ICAR for outcome
  vector[N] v;     // ICAR for treatment
  real phi;        // balancing factor
}

transformed parameters {
  real sigma = sqrt(sigma2);  // SDf
}

model {
  Z ~ normal(A*gamma + X*lamb + phi*u + v, sigma);
  y ~ normal(gamma*A*tau + X*beta + u, sigma);

  tau ~ normal(0, 10);
  gamma ~ normal(0, 10);
  beta ~ normal(0, 10);
  lamb ~ normal(0, 10);
  sigma2y ~ inv_gamma(0.5, 0.005);
  sigma2d ~ inv_gamma(0.5, 0.005);

  // ICAR priors
  target += -0.5 * dot_self(u[node1] - u[node2]);
  sum(u) ~ normal(0, 0.001*N);

  target += -0.5 * dot_self(v[node1] - v[node2]);
  sum(v) ~ normal(0, 0.001*N);
}

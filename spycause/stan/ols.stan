data {
  int<lower=1> N;  // number of observations
  int<lower=1> D;  // number of features
  int          K;  // 2 if interference-adjusted, 1 otherwise
  matrix[N, D] X;  // confounding variables
  vector[N]    y;  // outcome variable
  matrix[N, K] Z;  // inputted treatment variable
}

parameters {
  vector[D] beta;  // covariate effects
  vector[K] tau;   // treatment effects (possibly including lag)
  real<lower=0> sigma;  // SD of outcome
}

model {
  y ~ normal(X*beta + Z*tau, sigma);

  tau ~ normal(0, 2);
  beta ~ normal(0, 2);
  sigma ~ exponential(1);
}

generated quantities {
  vector[N] log_likelihood;
  for (i in 1:N) log_likelihood[i] = normal_lpdf(y[i] | X[i,]*beta + Z[i]*tau, sigma);

  array[N] real y_pred = normal_rng(X*beta + Z*tau, sigma);
}

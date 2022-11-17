data {
  int<lower=1> N;  // number of observations
  int<lower=1> D;  // number of features
  int          I;  // 2 if interference-adjusted, 1 otherwise
  matrix[N, D] X;  // confounding variables
  vector[N]    y;  // outcome variable
  matrix[N, I] Z;  // inputted treatment variable
}

parameters {
  vector[D] beta;  // covariate effects
  vector[I] tau;   // treatment effects (possibly including lag)
  real<lower=0> sigma2;  // variance of outcome
}

transformed parameters {
  real sigma = sqrt(sigma2);  // SD, required for parametrizing the likelihood
}

model {
  y ~ normal(X*beta + Z*tau, sigma);

  tau ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma2 ~ inv_gamma(0.5, 0.005);
}
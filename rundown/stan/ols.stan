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

// transformed parameters {
// real sigma = sqrt(sigma2);  // SD, required for parametrizing the likelihood
// }

model {
  y ~ normal(X*beta + Z*tau, sigma);

  tau ~ normal(0, 5);
  beta ~ normal(0, 5);
  sigma ~ exponential(1);
  //sigma2 ~ cauchy(0, 5);
}

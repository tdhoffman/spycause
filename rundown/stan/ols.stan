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
  real tau[I];     // treatment effects (possibly including lag)
  real sigma2;     // variance of outcome
}

transformed parameters {
  real sigma = sqrt(sigma2);
}

model {
  y ~ normal(X * beta, sigma);

  tau ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma2 ~ inv_gamma(0.5, 0.005);
}

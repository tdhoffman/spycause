data {
  int<lower=1> N;  // number of observations
  int<lower=1> D;  // number of features
  matrix[N, D] X;  // confounding variables
  int Z[N];  // inputted treatment variable
}

parameters {
  vector[D] alpha;  // covariate effects
}

transformed parameters {
  real<lower=0, upper=1> pi[N] = to_array_1d(inv_logit(X*alpha));
}

model {
  Z ~ bernoulli(pi);
  alpha ~ normal(0, 5);
}

generated quantities {
  vector[N] log_likelihood;
  vector[N] pi_hat;
  for (i in 1:N) {
    log_likelihood[i] = bernoulli_lpmf(Z[i] | pi[i]);
    pi_hat[i] = bernoulli_rng(pi[i]);
  }
}
  

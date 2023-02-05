data {
  int N;           // number of observation units
  int D;           // number of confounding variables
  int K;           // 2 if interference-adjusted, 1 otherwise
  vector[N] Y;     // outcome variable
  matrix[N, D] X;  // confounders
  array[N] int Z;        // treatment variable
  vector[N] Zlag;  // interference-adjusted Z (optional)

  // ICAR stuff
  int<lower=0> N_edges;                  // number of edges
  array[N_edges] int<lower=1, upper=N> node1;  // node1[i] is adjacent to node2[i]
  array[N_edges] int<lower=1, upper=N> node2;  // and node1[i] < node2[i]
  vector[N_edges] weights;               // weights for the edges
}

transformed data {
  vector[N] Zvec = to_vector(Z);
}

parameters {
  vector[D] beta;       // effects of confounders on outcome
  vector[D] alpha;      // effects of confounders on treatment
  vector[K] tau;        // effect of treatment on outcome
  vector[N] u;          // spatial effects in outcome model
  vector[N] v;          // spatial effects in treatment model
  real<lower=0> sd_u;   // SD of ICAR term for outcome model
  real<lower=0> sd_v;   // SD of ICAR term for treatment model
  real<lower=0> sigma2; // var of outcome
  real psi;             // shared spatial random effect on outcome
}

transformed parameters {
  array[N] real<lower=0, upper=1> pi = to_array_1d(inv_logit(X*alpha + sd_v*v));
  real sigma = sqrt(sigma2);
  real<lower=0> tau_u = inv(sqrt(sd_u));
  real<lower=0> tau_v = inv(sqrt(sd_v));

  vector[N] mu;
  if (K == 2) {
    mu = Zvec*tau[1] + Zlag*tau[2] + X*beta + sd_u*u + psi*sd_v*v;
  } else {
    mu = Zvec*tau[1] + X*beta + sd_u*u + psi*sd_v*v;
  }
}

model {
  // Likelihoods
  Z ~ bernoulli(pi);
  Y ~ normal(mu, sigma);

  // Priors
  alpha ~ normal(0, 5);
  beta ~ normal(0, 5);
  tau ~ normal(0, 5);
  psi ~ normal(0, 0.1);
  sigma ~ exponential(1);
  tau_u ~ gamma(0.5, 0.005);
  tau_v ~ gamma(0.5, 0.005);
  
  // CAR priors
  target += -0.5 * weights * dot_self(u[node1] - u[node1]);
  target += -0.5 * weights * dot_self(v[node1] - v[node2]);

  // Soft sum-to-zero constraint on spatial effects
  sum(u) ~ normal(0, 0.001*N);
  sum(v) ~ normal(0, 0.001*N);
}

generated quantities {
  vector[N] log_likelihood;
  vector[N] y_pred;
  for (i in 1:N) {
    log_likelihood[i] = normal_lpdf(Y[i] | mu[i], sigma);
    y_pred[i] = normal_rng(mu[i], sigma);
  }
}

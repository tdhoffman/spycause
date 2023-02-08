functions {
  /**
  * Return the log probability of a proper conditional autoregressive (CAR) prior 
  * with a sparse representation for the adjacency matrix.
  * Manually specifying the sparsity provides significant speedups over expressing
  * the CAR term as coming from a multivariate normal distribution.
  *
  * @param phi Vector containing the parameters with a CAR prior
  * @param prec_phi Precision parameter for the CAR prior (real)
  * @param rho Dependence (usually spatial) parameter for the CAR prior (real)
  * @param W_sparse Sparse representation of adjacency matrix (int array)
  * @param n Length of phi (int)
  * @param W_n Number of adjacent pairs (int)
  * @param D_sparse Number of neighbors for each location (vector)
  * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
  *
  * @return Log probability density of CAR prior up to additive constant
  * 
  * Citation:


  * Joseph, M. (2016). "Exact sparse CAR models in Stan." 
  *     https://mc-stan.org/users/documentation/case-studies/mbjoseph-CARStan.html
  */
  real sparse_car_lpdf(vector phi, real prec_phi, real rho, 
    int[,] W_sparse, vector D_sparse, vector lambda, int n, int W_n) {
      row_vector[n] phit_D; // phi' * D
      row_vector[n] phit_W; // phi' * W
      vector[n] ldet_terms;
    
      phit_D = (phi .* D_sparse)';
      phit_W = rep_row_vector(0, n);
      for (i in 1:W_n) {
        phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
        phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
      }
    
      for (i in 1:n) ldet_terms[i] = log1m(rho * lambda[i]);
      return 0.5 * (n * log(prec_phi)
                    + sum(ldet_terms)
                    - prec_phi * (phit_D * phi - rho * (phit_W * phi)));
  }
}

data {
  int<lower=1> N;  // number of observation units
  int<lower=1> D;  // number of confounding variables
  int<lower=1> K;  // 2 if interference-adjusted, 1 otherwise
  vector[N] Y;     // outcome variable
  matrix[N, D] X;  // confounders
  array[N] int Z;  // treatment variable
  vector[N] Zlag;  // interference-adjusted Z (optional)

  // CAR stuff
  matrix<lower=0, upper=1>[N, N] W;  // adjacency matrix
  int W_n;                           // number of adjacent region pairs
}

transformed data {
  vector[N] Zvec = to_vector(Z);

  int W_sparse[W_n, 2];  // adjacency pairs
  vector[N] D_sparse;    // diagonal of D (number of neighbors for each region)
  vector[N] lambda;      // eigenvalues of D^{-1/2} * W * D^{-1/2}

  // Generate sparse representation for W
  int counter = 1;
  // loop over upper triangular part of W to identify neighbor pairs
  for (i in 1:(N - 1)) {
    for (j in (i + 1):N) {
      if (W[i, j] == 1) {
        W_sparse[counter, 1] = i;
        W_sparse[counter, 2] = j;
        counter = counter + 1;
      }
    }
  }

  // Obtain entries of D
  for (i in 1:N) D_sparse[i] = sum(W[i]);
  vector[N] invsqrtD;
  for (i in 1:N) {
    invsqrtD[i] = 1 / sqrt(D_sparse[i]);
  }
  lambda = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
}

parameters {
  vector[D] beta;       // effects of confounders on outcome
  vector[D] alpha;      // effects of confounders on treatment
  vector[K] tau;        // effect of treatment on outcome
  vector[N] u;          // spatial effects in outcome model
  vector[N] v;          // spatial effects in treatment model
  real<lower=0> prec_u; // prec of CAR term for outcome model
  real<lower=0> prec_v; // prec of CAR term for treatment model
  real<lower=0> sigma2; // var of outcome
  real psi;             // shared spatial random effect on outcome
  real<lower=0, upper=1> rho_u;  // level of spatial autocorrelation for u
  real<lower=0, upper=1> rho_v;  // level of spatial autocorrelation for v
}

transformed parameters {
  array[N] real<lower=0, upper=1> pi = to_array_1d(inv_logit(X*alpha + v));
  real sigma = sqrt(sigma2);

  vector[N] mu;
  if (K == 2) {
    mu = Zvec*tau[1] + Zlag*tau[2] + X*beta + u + psi*v;
  } else {
    mu = Zvec*tau[1] + X*beta + u + psi*v;
  }
}

model {
  // Likelihoods
  Z ~ bernoulli(pi);
  Y ~ normal(mu, sigma);

  // Priors
  alpha ~ normal(0, 2);
  beta ~ normal(0, 2);
  tau ~ normal(0, 2);
  psi ~ normal(0, 0.1);
  sigma ~ exponential(1);
  
  // CAR priors
  u ~ sparse_car(prec_u, rho_u, W_sparse, D_sparse, lambda, N, W_n);
  v ~ sparse_car(prec_v, rho_v, W_sparse, D_sparse, lambda, N, W_n);
  prec_u ~ gamma(0.5, 0.005);
  prec_v ~ gamma(0.5, 0.005);
}

generated quantities {
  real<lower=0> sd_u = inv(sqrt(prec_u));
  real<lower=0> sd_v = inv(sqrt(prec_v));
  
  vector[N] log_likelihood;
  vector[N] y_pred;
  for (i in 1:N) {
    log_likelihood[i] = normal_lpdf(Y[i] | mu[i], sigma);
    y_pred[i] = normal_rng(mu[i], sigma);
  }
}

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
  int<lower=1> N;  // number of observations
  int<lower=1> D;  // number of features
  int          K;  // 2 if interference-adjusted, 1 otherwise
  matrix[N, D] X;  // confounding variables
  vector[N]    y;  // outcome variable
  matrix[N, K] Z;  // inputted treatment variable

  // CAR stuff
  matrix<lower=0, upper=1>[N, N] W;  // adjacency matrix
  int W_n;                           // number of adjacent region pairs
}

transformed data {
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
  vector[D] beta;        // covariate effects
  vector[K] tau;         // treatment effects (possibly including lag)
  real<lower=0> sigma2;  // variance of outcome

  // CAR effects
  vector[N] u;               // spatially structured residuals
  real<lower=0, upper=1> rho;  // level of spatial autocorrelation
  real<lower=0> prec_u;      // precision of u 
}

transformed parameters {
  real sigma = sqrt(sigma2);  // SD, required for parametrizing the likelihood
  vector[N] mu = X*beta + Z*tau + u;
}

model {
  y ~ normal(mu, sigma);

  tau ~ normal(0, 2);
  beta ~ normal(0, 2);
  sigma ~ exponential(1);

  // CAR prior
  u ~ sparse_car(prec_u, rho, W_sparse, D_sparse, lambda, N, W_n);
  prec_u ~ gamma(0.5, 0.005);
}

generated quantities {
  real<lower=0> sd_u = inv(sqrt(prec_u));

  vector[N] log_likelihood;
  vector[N] y_pred;
  for (i in 1:N) {
    log_likelihood[i] = normal_lpdf(y[i] | mu[i], sigma);
    y_pred[i] = normal_rng(mu[i], sigma);
  }
}

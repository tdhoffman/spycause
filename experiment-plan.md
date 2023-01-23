## DGP parameters

Let $N$ be the number of spatial units (one observation per unit) and $D$ be the number of covariates.
Also note that the value $\phi = 0$ corresponds to an ICAR DGP, and any $\phi > 0$ corresponds to the Joint DGP.
The choices of $W_I$ also correspond to various versions of interference: a region-based weights matrix, for example, represents partial interference.

| Math name | Code name | Description | Domain | Plausible values | Testing set |
|:----------|:----------|:------------|:------:|:----------------:|:-----------:|
| $\tau$    | `treat`   | Treatment effect of Z on Y | $(-\infty, \infty)$ | $[-10, 10]$ | $[-2, 2]$ |
| $\alpha$  | `z_conf`  | Effect of nonspatial confounders on Z | $(-\infty, \infty)^D$ | $[-10, 10]^D$ | $[-2, 2]^D$ |
| $\beta$   | `y_conf`  | Effect of nonspatial confounders on Y | $(-\infty, \infty)^D$ | $[-10, 10]^D$ | $[-2, 2]^D$ |
|           | `interf`  | Effect of interference on Y | $(-\infty, \infty)$ | $[-10, 10]$ | $[-2, 2]$ |
|$\sigma^2_X$|`x_sd`| Standard deviation of confounders | $(0, \infty)$ | $(0, 10)$ | $(0, 1]$ |
| $\rho_X$ | `x_sp` | Spatial autocorrelation of confounders | $(0, 1)$ | $[0.5, 0.99]$| $[0.5, 0.99]$ |
|$\sigma^2_Y$|`y_sd`| Standard deviation of outcome | $(0, \infty)$ | $(0, 10)$ | $(0, 1]$ |
| $\rho_U$ | `ucar_str` | Strength of spatial association for spatial confounding on Y | $(0, 1)$ | $[0.5, 0.99]$| $[0.5, 0.99]$ |
|$\sigma^2_U$|`ucar_sd`| Standard deviation of CAR term for confounding on Y | $(0, \infty)$ | $(0, 10)$ | $(0, 1]$ |
| $\rho_V$ | `vcar_str` | Strength of spatial association for spatial confounding on Z | $(0, 1)$ | $[0.5, 0.99]$| $[0.5, 0.99]$ |
|$\sigma^2_V$|`vcar_sd`| Standard deviation of CAR term for confounding on Z | $(0, \infty)$ | $(0, 10)$ | $(0, 1]$ |
|$\phi$| `balance` | Balancing factor that parametrizes the shared spatial confounding between Y and Z | $[0, 1]$ | $[0, 1]$ | $[0, 1]$ |
| $W_C$ | `w` (in `simulator.py`) | Weights matrix for spatial confounding | {binary contiguity, distance-based, region-based}| same as domain | same as domain |
| $W_I$ | `w` (in `preprocessing.py`)| Weights matrix for interference | {binary contiguity, distance-based, region-based} | same as domain | same as domain |


## Model parameters
There are substantially fewer choices on the modeling side of things.
- Choice of model: OLS, ICAR, or Joint.
- Weights choice in model: binary contiguity, distance-based, region-based.
- Kind of interference: binary contiguity, distance-based, region-based, or none.
- Nonspatial conditioning set: propensity score (spatial or nonspatial model and level of smoothing) or confounders.

## Fitting hyperparameters
These are hyperparameters to be tuned for posterior convergence purposes, not to be studied in the manuscript.
They will be controlled in the experiments.
| Code name  | Description                             | Domain        |
|:-----------|:----------------------------------------|:--------------|
| `nsamples` | Number of samples per chain             | $\mathbb{Z}_{\geq 0}$ |
| `nwarmup`  | Number of samples for burn-in per chain | $\mathbb{Z}_{\geq 0}$ |
| `nchains`  | Number of Markov chains to use for sampling | $\mathbb{Z}_{\geq 0}$ |
| `save_warmup` | Flag for saving burn-in samples      | `True`, `False`|
| `delta` | Target Metropolis acceptance rate | $(0, 1)$ |
| `max_depth` | Max tree depth for NUTS | $\mathbb{Z}_{\geq 0}$ |

Below are reasonable settings for the experiments.
Of course, one could run these models with higher powered configurations, but these settings yield acceptable to superb convergence (depending on model) on all the diagnostic metrics.
| Model | Interference | Conditioning set | `nsamples` | `nwarmup` | `nchains` | `save_warmup` | `delta` | `max_depth` |
|:------|:-------------|:-----------------|:----------:|:---------:|:---------:|:-------------:|:-------:|:-----------:|
| OLS   | none         | confounders      | 1000       | 1000      | 1         | False         | 0.8     | 10          |
| OLS   | none         | propensity score | 1000       | 1000      | 1         | False         | 0.8     | 10          |
| OLS   | rook         | confounders      | 1000       | 1000      | 1         | False         | 0.8     | 10          |
| OLS   | rook         | propensity score | 1000       | 1000      | 1         | False         | 0.8     | 10          |
| ICAR  | none         | confounders      | 1000       | 1000      | 1         | False         | 0.8     | 10          |
| ICAR  | none         | propensity score | 1000       | 1000      | 1         | False         | 0.8     | 10          |
| ICAR  | rook         | confounders      | 1000       | 1000      | 1         | False         | 0.8     | 10          |
| ICAR  | rook         | propensity score | 1000       | 1000      | 1         | False         | 0.8     | 10          |
| Joint | none         | confounders      | 1000       | 1000      | 1         | False         | 0.8     | 10          |
| Joint | none         | propensity score | 1000       | 1000      | 1         | False         | 0.8     | 10          |
| Joint | rook         | confounders      | 1000       | 1000      | 1         | False         | 0.8     | 10          |
| Joint | rook         | propensity score | 1000       | 1000      | 1         | False         | 0.8     | 10          |

**TODO** update this table with final `nchains` and `nsamples`

# Experiments
Create a massive hypercube and run everything, recording bias and diagnostic values.
It doesn't look like these simulations will be too intensive, even though we'll be doing a lot of them, so this strategy should just work.
As a final touch, we can krige or otherwise interpolate the results to get an idea of what the rest of the landscape looks like.

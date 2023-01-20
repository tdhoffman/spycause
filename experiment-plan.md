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

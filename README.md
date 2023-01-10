# rundown
The rundown of (a) scenarios provoking issues in spatial causal inference, (b) existing spatial causal models, and (c) the matrix of applications of members of (b) on members of (a).

Throughout, we assume that there is **only one** observation per region (unlike Reich et al. 2021) and that all models permit a causal interpretation.

# Data scenarios

| Number | Linear | Spatial confounding | Interference |    Implemented?    |       Tested?      |
| :----: | :----: | :------------------ | :----------- | :----------------: | :----------------: |
|    1   |    T   | none                | none         | :heavy_check_mark: | :heavy_check_mark: |
|    2   |    F   | none                | none         | :heavy_check_mark: | :heavy_check_mark: |
|    3   |    T   | W                   | none         | :heavy_check_mark: | :heavy_check_mark: |
|    4   |    F   | W                   | none         | :heavy_check_mark: | :heavy_check_mark: |
|    5   |    T   | none                | partial      | :heavy_check_mark: | :heavy_check_mark: |
|    6   |    F   | none                | partial      | :heavy_check_mark: | :heavy_check_mark: |
|    7   |    T   | W                   | partial      | :heavy_check_mark: | :heavy_check_mark: |
|    8   |    F   | W                   | partial      | :heavy_check_mark: | :heavy_check_mark: |
|    9   |    T   | none                | general      | :heavy_check_mark: | :heavy_check_mark: |
|   10   |    F   | none                | general      | :heavy_check_mark: | :heavy_check_mark: |
|   11   |    T   | W                   | general      | :heavy_check_mark: | :heavy_check_mark: |
|   12   |    F   | W                   | general      | :heavy_check_mark: | :heavy_check_mark: |
|   13   |    T   | none                | network      | :heavy_check_mark: | :heavy_check_mark: |
|   14   |    F   | none                | network      | :heavy_check_mark: | :heavy_check_mark: |
|   15   |    T   | W                   | network      | :heavy_check_mark: | :heavy_check_mark: |
|   16   |    F   | W                   | network      | :heavy_check_mark: | :heavy_check_mark: |

**Notes:** 
- It's assumed that there is nonspatial confounding present in all the above. Estimation of these models
  without nonspatial confounding is a special case and is less interesting.
- Partial interference is with respect to a choice of regions (spatial clusters) that will need to be
  selected. A few regionalizations will be chosen.
- Spatial confounding is with respect to a choice of weights, or ways to represent exactly how the 
  neighboring confounding relationships work (notated by "W" here). A few different weights will
  be chosen to represent this.
- Network interference is also with respect to a choice of weights (specification of an adjacency matrix
  for the network). Again, a few different weights will be chosen for this.
- We do everything on lattices for simplicity.
- We assume all interference and confounding are linear for simplicity. That is, the effects of 
  neighbors are always additive and hence can be expressed using a weights matrix.

The `CARSimulator` class contains the code for simulating all these scenarios.
The other simulators are old code, to be dealt with at a later date.

Precisely, the data generating processes for these scenarios are as follows:
- **OLS**: $y \sim N(X\beta + Z\tau, \sigma^2)$, $\pi = \text{expit}(X\alpha + V)$.
  Used for scenarios with no spatial confounding.
- **CAR**: $y \sim N(X\beta + Z\tau + U, \sigma^2)$, $\pi = \text{expit}(X\alpha + V)$. 
  Used for scenarios with spatial confounding
- **Joint**: $y \sim N(X\beta + Z\tau + U, \sigma^2)$, $\pi = \text{expit}(X\alpha + \phi U + V)$.
In all of the DGPs, $Z \sim \text{Bernoulli}(\pi)$, $U \sim \text{CAR}(\rho_u, \sigma_u)$, and $V \sim \text{CAR}(\rho_v, \sigma_v)$.
Note that it is always assumed that there is some spatial pattern $V$ in the treatment allocation that is not accounted for by the covariates $X$ or the missing confounder $U$.

# TODO
- add $V$ to all test simulations
- continue refactoring this doc


# Models
| Name  | Spatial confounding | Interference | Implemented?       | Tested?            | Log likelihood? | Posterior predictive? |
|:-----:|:--------------------|:------------:|:------------------:|:------------------:|:---------------:|:---------------------:|
| OLS+X | none                | none         | :heavy_check_mark: | :heavy_check_mark: |                 |                       |
| OLS+P | none                | none         |                    |                    |                 |                       |
| ICAR+X| unobserved          | none         | :heavy_check_mark: | :heavy_check_mark: |                 |                       |
| ICAR+P| unobserved          | none         |                    |                    |                 |                       |
| Joint | unobserved          | none         | :heavy_check_mark: | :heavy_check_mark: |                 |                       |
| OLS+X | none                | linear       | :heavy_check_mark: | :heavy_check_mark: |                 |                       |
| OLS+P | none                | linear       |                    |                    |                 |                       |
| ICAR+X| unobserved          | linear       | :heavy_check_mark: | :heavy_check_mark: |                 |                       |
| ICAR+P| unobserved          | linear       |                    |                    |                 |                       |
| Joint | unobserved          | linear       | :heavy_check_mark: |                    |                 |                       |

Here, "+X" refers to including all the nonspatial confounders, while "+P" refers to just including treatment and propensity score.
We may not end up using the CAR models for simplicity (ICAR is much better known), but I anticipate the exact sparse CAR models being able to capture spatial structure better.

The DAG for our model setting looks like this:

![image](confounding-setting.png)

where $\pi(x, u) = P(Z = 1 \mid X = x, U = u)$ is the propensity score, $X$ is a set of observed confounders, and $U$ stands for all unobserved spatial confounders.
$\pi$ is placed in a square node to indicate that it is not a random variable, and rather a deterministic function of random variables $X$ and $U$.
We'll vary the way in which $\pi$ is incorporated in the model (directly or using B-spline smoothing).

We omit spatial instrumental variables (IVs) and geographic regression discontinuity design (GRD), as they require more assumptions or structure and therefore are somewhat less comparable to the regression adjustments.
Finally, we omit nonlinear models for simplicity, as I can't find any literature on them in the spatial setting and that means things are going to get complicated.
These are good avenues for future work.

# Data-model matrix
| Data scenario | | |
| :-----------: |-|-|
|       1       | | |
|       2       | | |
|       3       | | |
|       4       | | |
|       5       | | |
|       6       | | |
|       7       | | |
|       8       | | |
|       9       | | |
|      10       | | |
|      11       | | |
|      12       | | |
|      13       | | |
|      14       | | |
|      15       | | |
|      16       | | |

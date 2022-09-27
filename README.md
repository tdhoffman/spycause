# rundown
The rundown of (a) scenarios provoking issues in spatial causal inference, (b) existing spatial causal models, and (c) the matrix of applications of members of (b) on members of (a).

Throughout, we assume that there is **only one** observation per region (unlike Reich et al. 2021).

## TODO
- Make simulations for all the data scenarios
- Sketch out package structure
- Implement all the methods from Reich et al. 2021
- Implement some nonspatial causal models

## Data scenarios

### Confounding
- spatially autocorrelated treatment
- spatially autocorrelated outcome
- spatially autocorrelated confounders

### Interference
First one definitely exists. Do the others exist too?
- treatment at one location influences outcomes at others
- treatment at one location influences treatment at others (?)
- treatment at one location influences confounders at others (?)

### Mixtures
Combinations of spatial autocorrelation and treatment interference.

## Existing spatial causal models

### Reich et al. 2021 methods to mitigate omitted spatially autocorrelated variables
- case-control matching (e.g. Jarner et al. 2002)
- neighborhood adjustments by spatial smoothing (Schnell and Papadogeorgou 2019)
- propensity score methods (Davis et al. 2019)

They review these methods and do a simulation study to compare their precision for estimating a causal effect in the presence of a missing spatial confounder.

### Reich et al. 2021 methods to deal with interference (spillover)
- partial (Zigler et al. 2012)
- network (Tchetgen et al. 2017)
- ways to combine mechanistic and spatial statistical models to anchor the analysis to scientific theory

## Existing nonspatial causal models
(for comparison to the spatial ones)
- DAGs and Pearl's algorithm for eliminating confounders (maybe not?)
- Causal ML (using BART/BCF, neural nets like EconML, representation learning)
- ...


## Data-model Matrix

| Number |                              Scenario                              |    Implemented?    | |
| :----: | :----------------------------------------------------------------- | :----------------: |-|
|    1   | nonspatial, no confounding, linear relationship                    | :heavy_check_mark: | |
|    2   | nonspatial, no confounding, nonlinear relationship                 |                    | |
|    3   | nonspatial, confounding, linear relationship                       | :heavy_check_mark: | |
|    4   | nonspatial, confounding, nonlinear relationship                    |                    | |
|    5   | auto in Y, no confounding, linear relationship                     | :heavy_check_mark: | |
|    6   | auto in Y, no confounding, nonlinear relationship                  |                    | |
|    7   | auto in Y, confounding, linear relationship                        | :heavy_check_mark: | |
|    8   | auto in Y, confounding, nonlinear relationship                     |                    | |
|    9   | auto in X, no confounding, linear relationship                     | :heavy_check_mark: | |
|   10   | auto in X, no confounding, nonlinear relationship                  |                    | |
|   11   | auto in X, confounding, linear relationship                        | :heavy_check_mark: | |
|   12   | auto in X, confounding, nonlinear relationship                     |                    | |
|   13   | auto in Z, no confounding, linear relationship                     | :heavy_check_mark: | |
|   14   | auto in Z, no confounding, nonlinear relationship                  |                    | |
|   15   | auto in Z, confounding, linear relationship                        | :heavy_check_mark: | |
|   16   | auto in Z, confounding, nonlinear relationship                     |                    | |
|   17   | interference, no confounding, linear relationship                  |                    | |
|   18   | interference, no confounding, nonlinear relationship               |                    | |
|   19   | interference, confounding, linear relationship                     |                    | |
|   20   | interference, confounding, nonlinear relationship                  |                    | |
|   21   | interference, auto in Y, no confounding, linear relationship       |                    | |
|   22   | interference, auto in Y, no confounding, nonlinear relationship    |                    | |
|   23   | interference, auto in Y, confounding, linear relationship          |                    | |
|   24   | interference, auto in Y, confounding, nonlinear relationship       |                    | |
|   25   | interference, auto in X, no confounding, linear relationship       |                    | |
|   26   | interference, auto in X, no confounding, nonlinear relationship    |                    | |
|   27   | interference, auto in X, confounding, linear relationship          |                    | |
|   28   | interference, auto in X, confounding, nonlinear relationship       |                    | |
|   29   | interference, auto in Z, no confounding, linear relationship       |                    | |
|   30   | interference, auto in Z, no confounding, nonlinear relationship    |                    | |
|   31   | interference, auto in Z, confounding, linear relationship          |                    | |
|   32   | interference, auto in Z, confounding, nonlinear relationship       |                    | |

**Notes:** 
- "auto in S" is shorthand for indicating that S is spatially autocorrelated
- "confounding" means Z is a function of X
- "interference" indicates that treatment at one location affects outcomes at another

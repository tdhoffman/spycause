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

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


# Data-model matrix

| Number | Linear | Spatial confounding | Interference |    Implemented?    | |
| :----: | :----: | :------------------ | :----------- | :----------------: |-|
|    1   |    T   | none                | none         |                    | |
|    2   |    F   | none                | none         |                    | |
|    3   |    T   | W                   | none         |                    | |
|    4   |    F   | W                   | none         |                    | |
|    5   |    T   | none                | partial      |                    | |
|    6   |    F   | none                | partial      |                    | |
|    7   |    T   | none                | general      |                    | |
|    8   |    F   | none                | general      |                    | |
|    9   |    T   | none                | network      |                    | |
|   10   |    F   | none                | network      |                    | |
|   11   |    T   | W                   | partial      |                    | |
|   12   |    F   | W                   | partial      |                    | |
|   13   |    T   | W                   | general      |                    | |
|   14   |    F   | W                   | general      |                    | |
|   15   |    T   | W                   | network      |                    | |
|   16   |    F   | W                   | network      |                    | |

**Notes:** 
- it's assumed that there is nonspatial confounding present in all the above. Estimation of these models
  without confounding is a special case and is less interesting.
- partial interference is with respect to a choice of regions (spatial clusters) that will need to be
  selected. It may make sense to choose a few regionalizations to use.
- spatial confounding is with respect to a choice of weights, or ways to represent exactly how the 
  neighboring confounding relationships work (notated by "W" here). It may make sense to choose a 
  few different weights to represent this.
- network interference is also with respect to a choice of weights (specification of an adjacency matrix
  for the network). Again, it may make sense to choose a few different weights for this.

**Currently implemented** are autocorrelated X, Y, and Z variable. I need to implement confounding,
which is different.

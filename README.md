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

| Number | Linear | Spatial confounding | Interference |    Implemented?    |       Tested?      |
| :----: | :----: | :------------------ | :----------- | :----------------: | :----------------: |
|    1   |    T   | none                | none         | :heavy_check_mark: | :heavy_check_mark: |
|    2   |    F   | none                | none         | :heavy_check_mark: | :heavy_check_mark: |
|    3   |    T   | W                   | none         | :heavy_check_mark: | :heavy_check_mark: |
|    4   |    F   | W                   | none         | :heavy_check_mark: |                    |
|    5   |    T   | none                | partial      | :heavy_check_mark: | :heavy_check_mark: |
|    6   |    F   | none                | partial      | :heavy_check_mark: |                    |
|    7   |    T   | W                   | partial      | :heavy_check_mark: | :heavy_check_mark: |
|    8   |    F   | W                   | partial      | :heavy_check_mark: |                    |
|    9   |    T   | none                | general      | :heavy_check_mark: | :heavy_check_mark: |
|   10   |    F   | none                | general      | :heavy_check_mark: |                    |
|   11   |    T   | W                   | general      | :heavy_check_mark: | :heavy_check_mark: |
|   12   |    F   | W                   | general      | :heavy_check_mark: |                    |
|   13   |    T   | none                | network      | :heavy_check_mark: | :heavy_check_mark: |
|   14   |    F   | none                | network      | :heavy_check_mark: |                    |
|   15   |    T   | W                   | network      | :heavy_check_mark: | :heavy_check_mark: |
|   16   |    F   | W                   | network      | :heavy_check_mark: |                    |

**Notes:** 
- It's assumed that there is nonspatial confounding present in all the above. Estimation of these models
  without confounding is a special case and is less interesting.
- Partial interference is with respect to a choice of regions (spatial clusters) that will need to be
  selected. It may make sense to choose a few regionalizations to use.
- Spatial confounding is with respect to a choice of weights, or ways to represent exactly how the 
  neighboring confounding relationships work (notated by "W" here). It may make sense to choose a 
  few different weights to represent this.
- Network interference is also with respect to a choice of weights (specification of an adjacency matrix
  for the network). Again, it may make sense to choose a few different weights for this.
- We'll do everything on lattices for simplicity.
- We'll assume all interference and confounding are linear for simplicity. That is, the effects of 
  neighbors are always additive and hence can be expressed using a weights matrix.

**Currently implemented** are autocorrelated X, Y, and Z variable. I need to implement confounding,
which is different, and interference. The current setup also has several classes which inherit from each other.
I think it may be better to create one `Simulator` class and have it contain all the functionality. 
By assuming confounding and interference are linear, we can always express them with a matrix which
makes implementation much simpler. Basically, the `Simulator` class will look like the following:

**`Simulator`**
- `__init__()`:
  - contains required parameters
  - `linear` (default `True`)
  - `sp_confound` (default `False`)
  - `interference` (default `None`)
- method `simulate()`:
  - generate X (spatially autocorrelated?)
  - generate Z as a function of X
  - generate Y as a function of X and Z
  - if `linear` is false: modify functional forms
  - if `sp_confound` is true: have X affect neighboring Y
  - for each non-`None` `interference` option (`"partial"`, `"general"`, `"network"`), change which Z 
    affect Y.
    
Actually, it may be preferable to modify this slightly. Make `Simulator` by default linear, but encode
these functional forms as separate class methods (e.g. `f_x()` generates X). Then, subclasses can override
those functions to offer alternative model forms, as there are a ton of potential nonlinear forms that we
might want. That paradigm might look like:

**`Simulator`**
- `__init__()`:
  - contains required parameters
  - `sp_confound` (default `False`)
  - `interference` (default `None`)
- method `f_x(*args)`:
  - generate X from parameters
  - (spatially autocorrelated?)
- method `f_z(X, *args)`:
  - generate Z linearly from parameters and X
  - if `sp_confound` is true: have X affect neighboring Z
- method `f_y(X, Z, *args)`:
  - generate Y linearly from parameters, X, and Z
  - if `sp_confound` is true: have X affect neighboring Y
  - for each non-`None` `interference` option (`"partial"`, `"general"`, `"network"`), change which Z 
    affect Y.
- method `simulate()`:
  - generate `X = f_x(*args)`
  - generate `Z = f_z(X, *args)`
  - generate `Y = f_y(X, Z, *args)`

This is akin to what I initially wrote, but more concrete and with a few key differences in light of the
scenario list and assumptions. Also -- and this would probably best be worked out during implementation --
the `f_*` functions should be fleshed out a bit more. Do they just specify the relationships between variables?
Do they also specify the nature of dependencies? How much functionality goes into them?


## Actual `Simulator` implementation
Implemented the `Simulator` class successfully and deleted `old_simulator.py`. Currently in the process
of testing the various options in it. I realized that `f_x(*args)` is not necessary as X can simply be
generated as some spatially autocorrelated data by default. Moreover, Z is constrained to be Binomial(1, p)
where p is a vector of probabilities that are a linear function of X and potentially location. This way,
subclasses of this default `Simulator` will be able to override the `f_z` function and easily change only
the linearity of Z rather than other aspects. Finally, to be coherent with functional causal models, 
Y is fully generated in `_create_Y`. The function can be overriden in any number of ways by future
subclasses.

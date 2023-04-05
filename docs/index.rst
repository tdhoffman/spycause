.. spycause documentation master file, created by
   sphinx-quickstart on Fri Mar  3 12:44:18 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to spycause's documentation!
====================================

Spycause (/spī' kôz/ or /e' spē kôz/) is a Python package of spatial causal models, diagnostics, and tools for simulating data.
All models are fitted in a Bayesian framework using `the Stan programming language <https://mc-stan.org>`_.

.. toctree::
   simulator.md
   models.md
   :maxdepth: 2
   :caption: Contents

Next steps for implementations:
* Spatial diff-in-diff
* Spatial instrumental variables
* Geographic regression discontinuity design

References
----------
Reich, B.J., Yang, S., Guan, Y., Giffin, A.B., Miller, M.J., and Rappold, A. (2021). A review of spatial causal inference methods for environmental and epidemiological applications. *International Statistical Review*, 89(4):605-634.

Stan Development Team. (2023). Stan Modeling Language Users Guide and Reference Manual, 2.31. `https://mc-stan.org <https://mc-stan.org>`_.

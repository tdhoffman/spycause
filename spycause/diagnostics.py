__author__ = "Tyler D. Hoffman cause@tdhoffman.com"

"""
Reusable diagnostics function for sampling procedures.
"""

import arviz as az


def diagnostics(model, params=["beta", "tau", "sigma"]):
    """
    Return some diagnostics about the posterior convergence.
    Based on https://rdrr.io/github/tomshafer/tshfr/src/R/stan-utils.R, which is
    in turn based on https://betanalpha.github.io/assets/case_studies/pystan_workflow.html.

    model  : fitted model from this library
             model object containing results for diagnostic computation
    params : list, optional
             list of strings specifying the parameters to compute 
             diagnostics for
    """

    # Divergences
    divergences = model.results_["divergent__"]
    ndivergent = divergences.sum()
    nsamples = len(divergences)
    print(f"{ndivergent} of {nsamples} iterations ended with a divergence ({100*ndivergent/nsamples}%).")
    if ndivergent > 0:
        print("Increasing adapt_delta may remove the divergences.")

    # Tree depth
    treedepths = model.results_["treedepth__"]
    nmaxdepths = (treedepths == model.max_depth).sum()
    print(f"{nmaxdepths} of {nsamples} iterations saturated the max tree depth ({100*nmaxdepths/nsamples}%).")
    if nmaxdepths > 0:
        print("See https://betanalpha.github.io/assets/case_studies/identifiability.html for more information.")

    # ESS
    ess_warning = False
    model.esses = az.ess(model.idata_)
    model.ess_dict = dict.fromkeys(params)
    for param in params:
        model.ess_dict[param] = model.esses[param].values
        ratio = model.ess_dict[param] / nsamples
        if hasattr(ratio, "__len__"):
            if (ratio < 0.001).any():
                ess_warning = True
                print(f"ESS/nsamples for components of parameter {param} is below 0.001.")
        else:
            if ratio < 0.001:
                ess_warning = True
                print(f"ESS/nsamples for parameter {param} is {ratio}, below 0.001.")

    if not ess_warning:
        print("ESS/nsamples looks reasonable for all parameters.")
    else:
        print("ESS/nsamples below 0.001 indicates the ESS has been overestimated.")

    # Rhat
    rhat_warning = False
    model.rhats = az.rhat(model.idata_)
    model.rhat_dict = dict.fromkeys(params)
    for param in params:
        rhats = model.rhats[param].values
        if hasattr(rhats, "__len__"):
            if (rhats > 1.1).any():
                rhat_warning = True
                print(f"Rhats for components of parameter {param} are {rhats}, above 1.1.")
        else:
            if rhats > 1.1:
                rhat_warning = True
                print(f"Rhat for parameter {param} is {rhats}.")
        model.rhat_dict[param] = rhats

    if not rhat_warning:
        print("Rhat looks reasonable for all parameters.")
    else:
        print("Rhat above 1.1: the chains have likely not mixed.")

    # Energy
    bfmi_warning = False
    model.bfmi = az.bfmi(model.idata_)
    if hasattr(model.bfmi, "__len__"):
        if (model.bfmi < 0.2).any():
            bfmi_warning = True
            print(f"BFMIs are {model.bfmi}, under 0.2; may need to reparametrize your model.")
    else:
        if model.bfmi < 0.2:
            print(f"BFMI = {model.bfmi} < 0.2. You may need to reparametrize your model.")

    if not bfmi_warning:
        print("BFMI indicated no pathological behavior.")

"""
Test an installation of PyStan using a simple normal mean estimation problem.

The script does a Stan fit to simulated data from a normal distribution,
estimating the mean with a normal prior/normal sampling dist'n model.  It prints
summaries of the posterior to the terminal, and plots two figures showing
traceplots (of the posterior mean and the log density) and an estimated
posterior PDF (blue curve) along with the analytical PDF (green dashed curve).

Three statistical tests are run, with results printed to the terminal.  They
will occasionally fail even with a sound installation, but this should happen
only rarely.

Windows users:
To enable PyStan to find the MSVC compiler, create a distutils configuration
file with these two lines in it:

[build]
compiler = msvc

For local use with this script, the file should be named 'setup.cfg' and
should be in the directory where you run this script.  To make this
change globally, see the distutils configuration documentation for
the appropriate file name and location:

  https://docs.python.org/2/install/#distutils-configuration-files
  

Created Apr 16, 2015 by Tom Loredo
"""

import numpy as np
import scipy
import matplotlib as mpl
from matplotlib.pyplot import *
from scipy import *
from scipy import stats

from stanfitter import StanFitter

# try:
#     import myplot
#     from myplot import close_all, csavefig
#     #myplot.tex_on()
#     csavefig.save = False
# except ImportError:
#     pass


ion()


#-------------------------------------------------------------------------------
# Setup a "true" data generation model (i.e., sampling distribution) and
# simulate data from it.

# Sampling dist'n for data from a unit-sigma normal @ mu:
mu = 5.
samp_distn = stats.norm(mu, 1.)

# Simulate data:
N = 100
yvals = samp_distn.rvs(N)


#-------------------------------------------------------------------------------
# Compute the analytical posterior for a conjugate normal-normal model, for
# comparison with Stan's MCMC results.

mu0 = 0.  # prior mean
w0 = 10.  # prior width
w = 1./sqrt(N)  # likelihood width
ybar = mean(yvals)  # likelihood location
B = w**2/(w**2 + w0**2)  # shrinkage factor
mu_post = ybar + B*(mu0 - ybar)
sig_post = w*sqrt(1.-B)
post = stats.norm(mu_post, sig_post)

mu_l, mu_u = mu_post - 4.*sig_post, mu_post + 4.*sig_post
mu_vals = linspace(mu_l, mu_u, 300)
pdf_vals = post.pdf(mu_vals)


#-------------------------------------------------------------------------------
# Stan code for a conjugate normal-normal model with unknown mean:

normal_mu_code = """
data {
    int<lower=0> N; // number of samples
    real y[N]; // samples
}

parameters {
    real mu;
}

model {
    mu ~ normal(0, 10.);  // prior is a wide normal
    for (i in 1:N) {
        y[i] ~ normal(mu, 1.);  // sampling dist'n
    }
}
"""


#-------------------------------------------------------------------------------
# Using StanFitter to build and deploy the model.


# Invoke Stan to build the model, caching the built model in the CWD to save
# time if the script is re-run without any Stan code changes.
fitter = StanFitter(normal_mu_code)
# Alternatively, the Stan code can be in a separate .stan file:
#fitter = StanFitter('NormalNormal.stan')  


# Stan requires a dictionary providing the data.
normal_mu_data = {'N': N,  'y': yvals}

# The data could have been provided to StanFitter; the set_data() method
# enables using the same model to fit multiple datasets.
fitter.set_data(normal_mu_data)

# Run 4 chains of 1000 iters; Stan keeps the last half of each -> 2k samples.
# The fitter returns a StanFitResults instance, whose attributes provide
# access to the fit results.
fit = fitter.sample(n_iter=1000, n_chains=4)


#-------------------------------------------------------------------------------
# Examine fit results.

# Print textual summaries for the parameters monitored by Stan (the model
# parameter mu, and the log posterior PDF, log_p).  These include basic
# (minimal) convergence and mixing diagnostics.
print()
print(fit.mu)
print(fit.log_p)

# Also check convergence & mixing by examining trace plots,
# making sure there are no obvious trends or strong, long-range correlations.
# Use each parameter's trace() method to plot the trace on an existing set of
# axes, or to create a new figure with the parameter's traceplot.
f=figure(figsize=(10,8))
ax=f.add_subplot(2,1,1)
fit.mu.trace(axes=ax,alpha=.6)  # without `axes`, this will make its own fig
ax=f.add_subplot(2,1,2)
fit.log_p.trace(axes=ax,alpha=.6)

# Stan's default plot, showing a (marginal) PDF (via KDE) and a merged-chains
# trace plot:
fig = fit.stan_plot()
pdf_ax, trace_ax = fig.get_axes()

# Plot the analytical PDF on top of Stan's PDF estimate.
pdf_ax.plot(mu_vals, pdf_vals, 'g--', lw=3, alpha=.7)


#-------------------------------------------------------------------------------
# Test cases; note they will sometimes (rarely) fail even for correct
# code.  If the 'return' statements are changed to 'assert', these 
# become valid nose test cases, but nose appears to have issues with PyStan
# and/or matplotlib.


def test_post_mean():
    """
    Check that Stan's posterior mean matches the analytical mean to within
    3* the standard error.  This should fail ~< 1% of the time.
    """
    return abs(fit.mu.mean - mu_post)/fit.mu.se_mean < 3.

def test_intvl():
    """
    Check that the true mean is within the 95% interval; this should fail
    5% of the time.
    """
    lo, hi = fit.mu.q025, fit.mu.q975  # quantile attributes
    return (mu > lo) and (mu < hi)

def test_Rhat():
    """
    Test that the chain appears to have converged.  This can fail with
    correct code if the chain was not run long enough.
    """
    return abs(fit.mu.Rhat - 1.) < 0.05  # slightly more strict than 0.1 convention


print('********************************')
print('Test results (should be 3*True):')
print(test_post_mean(), test_intvl(), test_Rhat())
print('********************************\n')

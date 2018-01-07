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

Run with "ipython -i" to keep the plots displayed, and to allow interaction
with the fit.

Windows notes:

PyStan requires access to a C++ compiler compatible with your Python installation;
it uses it to build a Python extension implementing a Stan model.  This script
was tested with Windows 8.1 and Anaconda Python 2.7, using the Stan project's 
binary installation of PyStan, and Microsoft's free compiler for Python 2.7:

  http://www.microsoft.com/en-us/download/details.aspx?id=44266

This is a version of the 2008 Visual C++ compiler, the version used by 
Continuum and Python.org for Windows binary builds of the Python 2.7 series.
Python 3 instead uses the 2010 version.  Unfortunately, Microsoft has not
released a 2010 compiler package for Python 3.  This script has not been
tested for Python 3.  For one approach to enable support for building Python 3
extensions, see:

  http://blog.ionelmc.ro/2014/12/21/compiling-python-extensions-on-windows/

Running PyStan on Windows requires some script elements not needed by other
platforms; this script uses Python's platform module to conditionally
include these elements.

Finally, on Windows, Anaconda Python by default will try to use the
Cygwin gcc compiler, not the Microsoft Visual C++ (MSVC) compiler.  To
change the default behavior, create a distutils configuration file
with these two lines in it:

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

import platform
plat_sys = platform.system()
if plat_sys == 'Windows':
    # setuptools MUST be imported (BEFORE pystan) on Windows; it alters
    # distutils, enabling PyStan to find the correct MVC compiler.  You
    # will also need a distutils config file indicating that the MVC compiler
    # should be used; it should have the following two lines as content
    # (without the Python comment hashes):
    # [build]
    # compiler = msvc
    # For the config file name and location (local and global choices), see:
    #   https://docs.python.org/2/install/#distutils-configuration-files
    import setuptools, pystan
else:
    import pystan

try:
    import myplot
    from myplot import close_all, csavefig
    #myplot.tex_on()
    csavefig.save = False
except ImportError:
    pass


ion()

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

# Sampling dist'n for data from a unit-sigma normal @ mu:
mu = 5.
samp_distn = stats.norm(mu, 1.)

# Simulate data:
N = 100
yvals = samp_distn.rvs(N)

# Analytical posterior for conjugate normal-normal:
mu0 = 0.  # prior mean
w0 = 10.  # prior width
w = 1./sqrt(N)  # likelihood width
ybar = mean(yvals)  # likelihood location
B = w**2/(w**2 + w0**2)  # shrinkage factor
mu_post = ybar + B*(mu0 - ybar)
sig_post = w*sqrt(1.-B)
post_pdf = stats.norm(mu_post, sig_post)

# Stan requires a dictionary providing the data.
normal_mu_data = {'N': N,  'y': yvals}

# Run 4 chains of 1000 iters; Stan keeps the last half of each -> 2k samples.
if plat_sys == 'Windows':
    # By default, Stan will use multiprocessing to accelerate operation.
    # Windows does not support multiprocessing; make Stan run single jobs:
    fit = pystan.stan(model_code=normal_mu_code, data=normal_mu_data,
                      iter=1000, chains=4, n_jobs=1)
else:
    fit = pystan.stan(model_code=normal_mu_code, data=normal_mu_data,
                      iter=1000, chains=4)

# Samples in an array of three dimensions: [iteration, chain, parameter]
samples = fit.extract(permuted=False)
n_iter, n_ch, n_p1 = samples.shape
mu_traces = samples[:,:,0]
logp_traces = samples[:,:,-1]  # log(pdf) of the samples is also available, last

# Summaries from samples:
raw_summary = fit.summary()
mu_summaries = raw_summary['summary'][0]  # 0 entry for mu; 1 for log(p)
mu_mean, mu_mean_se = mu_summaries[:2]  # posterior mean for mu & its std error
mu_std = mu_summaries[:2]  # posterior std devn for mu
mu_cr95 = (mu_summaries[3], mu_summaries[7])  # boundaries of central 95% region
ESS = mu_summaries[8]  # ESS from all post-warmup samples
Rhat = mu_summaries[9]  # pot'l scale reduction convergence diagnostic

print('\n\n***** Stan fit results *****')
print('True mean:         {:.3f}'.format(mu))
print('An. post. mean:    {:.3f}'.format(mu_post))
print('Stan post. mean:   {:.3f} +- {:.3f} (MSE)'.format(mu_mean, mu_mean_se))
print('95% central region: [{:.2f}, {:.2f}]'.format(*mu_cr95))
print('ESS = {}, Rhat = {:.2f}'.format(ESS, Rhat))
print('****************************\n')


# Test cases; note they will sometimes (rarely) fail even for correct
# code.  If the 'return' statements are changed to 'assert', these 
# become valid nose test cases, but nose appears to have issues with PyStan
# and/or matplotlib.


def test_post_mean():
    """
    Check that Stan's posterior mean matches the analytical mean to within
    3* the standard error.  This should fail ~< 1% of the time.
    """
    return abs(mu_mean - mu_post)/mu_mean_se < 3.

def test_intvl():
    """
    Check that the true mean is within the 95% interval; this should fail
    5% of the time.
    """
    lo, hi = mu_cr95
    return (mu > lo) and (mu < hi)

def test_Rhat():
    """
    Test that the chain appears to have converged.  This can fail with
    correct code if the chain was not run long enough.
    """
    return abs(Rhat - 1.) < 0.05  # slightly more strict than 0.1 convention


print('****************************')
print('Test results (should be 3*True):', test_post_mean(), test_intvl(), test_Rhat())
print('****************************\n')


print(fit)  # Stan's textual summary


# Plot the traces for mu and log(p).
figure(figsize=(12,8))
mu_ax = subplot(211)
logp_ax = subplot(212)

for j in range(n_ch):
    mu_ax.plot(mu_traces[:,j], alpha=.5)  # alpha to see overlapping traces
mu_ax.set_ylabel('$\mu$')

for j in range(n_ch):
    logp_ax.plot(mu_traces[:,j], alpha=.5)
logp_ax.set_ylabel('$\log(p)$')
logp_ax.set_xlabel('Iteration')

# Stan's plot, showing a (marginal) PDF and merged trace plot:
fig = fit.traceplot()
pdf_ax, trace_ax = fig.get_axes()

# Plot analytical PDF on top of Stan's PDF estimate.
mu_l, mu_u = pdf_ax.get_xlim()
mus = linspace(mu_l, mu_u, 250)
pdf = post_pdf.pdf(mus)
pdf_ax.plot(mus, pdf, 'g--', lw=3, alpha=.7)

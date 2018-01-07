"""
Utilities for the AAS231 workshop:


* Functions for tracking NumPy's RNG state.  Suggested usage at script start
(customize file name for reused state as needed; use 'reuse_rng = False' to 
continue in the sequence of new pseudo-random numbers):

reuse_rng = True
if reuse_rng:
    restore_rng('numpy_rng_state-reuse.pkl')
else:
    restore_rng()

Then just call save_rng() at script end.


* Function for creating a 2-panel plot illustrating shrinkage of point estimates
in a simple hierarchical Bayesian model.

By Tom Loredo, (c) 2008, 2014
"""

import os, pickle
import os.path as path
from numpy import random, ones_like
from matplotlib.pyplot import *


MT_id = 'MT19937'  # NumPy's RNG as of 1.0.3

def save_rng(fname='numpy_rng_state.pkl'):
    """
    Save the state of NumPy's RNG to a file in the CWD.  Backup the previous
    two saved states if present.

    If the RNG state file exists (from a previous save), rename the previous
    one with a '.1' suffix.  If a '.1' file exists, rename it with a '.2'
    suffix.

    After use, to reproduce the previous run, restore the RNG using the
    state file with a '.1' suffix (the state used for the last run).
    """
    state = random.get_state()
    if state[0] == MT_id:
        id, state = state[0], (state[1], state[2]) # (ID, (key, pos)) for MT
    else:
        raise RuntimeError('numpy.random using unrecognized RNG type!')
    if path.exists(fname):
        fname1 = fname + '.1'
        if path.exists(fname1):
            fname2 = fname + '.2'
            os.rename(fname1, fname2)
        os.rename(fname, fname1)
    ofile = open(fname, 'wb')
    pickle.dump((id, state), ofile)
    ofile.close()
    
def restore_rng(fname='numpy_rng_state.pkl', notify=True):
    """
    Restore the state of NumPy's RNG from the contents of a file in the CWD
    if the file exists; otherwise use (and save) the default initialization.
    """
    if os.access(fname, os.R_OK | os.W_OK):
        rng_file = open(fname, 'rb')
        id, state = pickle.load(rng_file)
        rng_file.close()
        if id == MT_id:
            # Note key is numpy,uint32 -> need repr() to see value.
            if notify:
                print('Recovered RNG state:  %s [%s %s ...] %i' %\
                    (id, repr(state[0][0]), repr(state[0][1]), state[1]))
            random.set_state((id, state[0], state[1]))
        else:
            raise ValueError('Invalid ID for RNG in %s!' % fname)
    else:
        print('No accessible RNG status file; using (and saving) default',
              'initialization.')
        save_rng(fname)



dkred = '#882222'

def shrinkage_plot(x_vals, pdf_vals, x_true, x_ml, x_post, xlabel,
                   log_x=False, log_y=False, legend=True):
    """
    Make a plot showing a population distribution as a PDF in a top pane,
    with line plots beneath showing true subject values and the maximum
    likelihood and marginal posterior point estimates.

    Return the PDF and point estimate axes instances.
    """
    est_fig = figure(figsize=(10,8))

    # Axis rectangles:  left, bottom, width, height
    ax_pdf = est_fig.add_axes([.11, .5, .86, .47])
    ax_pts = est_fig.add_axes([.11, .05, .86, .35], frameon=False)
    ax_pts.autoscale_view(scaley=False) # *** seems to not work (or .plot overrides)

    # Plot the hyperprior.
    # True:
    if log_x and log_y:
        ax_pdf.loglog(x_vals, pdf_vals, 'b-', lw=2, label='True')
    elif log_x:
        ax_pdf.semilogx(x_vals, x_vals*pdf_vals, 'b-', lw=2, label='True')
    else:
        ax_pdf.plot(x_vals, pdf_vals, 'b-', lw=2, label='True')

    ax_pdf.set_xlabel(xlabel)
    ax_pdf.set_ylabel('PDF')
    if legend:
        ax_pdf.legend(frameon=False)

    # Plot true values and estimates:
    # Draw horizontal axes:
    y_true = .96
    y_ml = .5
    y_post = 0.04
    ax_pts.axhline(y_true, color='k')
    ax_pts.axhline(y_ml, color='k')
    ax_pts.axhline(y_post, color='k')
    # Don't plot ticks (major or minor)
    ax_pts.tick_params(bottom=False, top=False, left=False, right=False,
                       which='both')

    if log_x:
        pt_plot = ax_pts.semilogx
    else:
        pt_plot = ax_pts.plot

    # First draw links between estimates for the same subject:
    for xt, ml, post in zip(x_true, x_ml, x_post):
        pt_plot([xt, ml], [y_true, y_ml], 'k-', lw=1)
        pt_plot([ml, post], [y_ml, y_post], 'k-', lw=1)
        # For log_x, negative MLEs won't show up; draw a dashed
        # line from true to post.
        if log_x and ml < 0.:
            pt_plot([xt, post], [y_true, y_post], 'k:', lw=1)

    # Then the points:
    ms = 8
    msb = 10
    mew = 0.5
    u = ones_like(x_true)  # unit y values to scale
    pt_plot(x_true, y_true*u, 'bo', mew=mew, ms=ms)
    pt_plot(x_ml, y_ml*u, 'o', mew=mew, mfc=dkred, ms=ms)
    pt_plot(x_post, y_post*u, 'o', mew=mew, mfc='c', ms=ms)

    # Match the PDF and estimate plot limits.
    ax_pts.set_xlim(*ax_pdf.get_xlim())
    ax_pts.set_ylim(0,1)

    # Label the pt axes:
    tdict = { 'fontsize':20, 'verticalalignment':'bottom', 'horizontalalignment':'left',\
        'transform':ax_pts.transAxes }
    ax_pts.text(.02, y_true+.015, 'True', **tdict)
    ax_pts.text(.02, y_ml+.015, 'ML', **tdict)
    ax_pts.text(.02, y_post+.015, 'Post.', **tdict)

    return ax_pdf, ax_pts


"""
A wrapper around PyStan's compilation and fitting methods, providing a somewhat
more "Pythonic" interface to the fit results.

For PyStan info:

https://pystan.readthedocs.org/en/latest/getting_started.html

Created 2014-11-04 by Tom Loredo
2015-04-17:  Modified for BDA class
2018-01-02:  Modified for PyStan API updates (using v2.17)
2018-01-02:  Modified for Python 2 and 3 compatibility
"""
from __future__ import division
from __future__ import print_function
from six.moves import xrange

import pickle, glob
import datetime, timeit
from hashlib import md5
from collections import Mapping, OrderedDict

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

import platform
plat_is_win = platform.system() == 'Windows'
if plat_is_win:
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


__all__ = ['StanFitter']


# ImmutableAttrDict based on discussion from:
# http://stackoverflow.com/questions/9997176/immutable-dictionary-only-use-as-a-key-for-another-dictionary

class ImmutableAttrDict(Mapping):
    """
    A dict-like container providing item access both through the usual dict
    interface, and through object attributes.  Once set, items are immutable.
    """

    def __init__(self, *args, **kwargs):
        self._odict = OrderedDict(*args, **kwargs)  # will copy an input dict
        # Copy items to __dict__ so they're discoverable by IPython.
        for key, value in self._odict.items():
            if key in self.__dict__:
                raise ValueError('Key collision!')
            self.__dict__[key] = value

    def _asdict(self):
        """
        Return a new OrderedDict holding the (key, value) pairs.
        """
        return OrderedDict(self._odict)

    def __getitem__(self, key):
        return self._odict[key]

    def __len__(self):
        return len(self._odict)

    def __iter__(self):
        return iter(self._odict)

    def __eq__(self, other):
        return self._odict == other._odict

    def __getattr__(self, name):
        try:
            return self._odict[name]
        except KeyError:  # access other mapping methods
            return getattr(self._odict, name)

    def __setattr__(self, name, value):
        if name == '_odict':
            self.__dict__['_odict'] = value
        elif name in self._odict:
            raise TypeError('Existing attributes may not be altered!')
        else:
            if name in self.__dict__:
                raise ValueError('Key collision!')
            self._odict[name] = value
            # Copy to __dict__ so it's discoverable by IPython.
            self.__dict__[name] = value

    # def __delattr__(self, name):
    #     del self._od[name]



# TODO: Rework ParamHandler to avoid self.__dict__ = self; see:
# http://stackoverflow.com/questions/25660358/accessing-ordereddict-keys-like-attributes-in-python
# See ParamValueContainer above.

class ParamHandler(dict):
    """
    A container and handler for posterior sample data for a scalar parameter.

    This is mostly a dict-like object with access to data also possible via
    attributes, based on AttrDict from:

    http://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute-in-python
    """

    def __init__(self, *args, **kwargs):
        if 'fit' not in kwargs:
            raise ValueError('fit argument required!')
        if 'name' not in kwargs:
            raise ValueError('name argument required!')
        super(ParamHandler, self).__init__(*args, **kwargs)
        # NOTE:  The following works only because the dict superclass is
        # implemented in C, with special members in a struct rather than
        # a __dict___, so they remain accessible from self.
        self.__dict__ = self

    def subsample(self, n):
        """
        Return a random subsample of size n from the merged, thinned chains.

        Note that calling this separately for different parameters will not
        produced a set of parameter vectors from the posterior; the parameter
        values will be from different times in the (thinned) chains.
        """
        if n > len(self.thinned):
            raise ValueError('Requested sample size > thinned chain length!')
        return random.choice(self.thinned, n, replace=False)

    def trace(self, chain=None, step=True, axes=None,
                    xlabel=None, ylabel=None, **kwds):
        """
        Make a trace plot for the samples in chain `chain`.  If `chain` is None,
        show traces for all chains, iterating colors accorting to mpl defaults.

        By default, a step plot is used; set `step` to False for a line plot.
        """
        if axes is None:
            fig = plt.figure(figsize=(10,4))
            fig.subplots_adjust(bottom=.2, top=.9)
            axes = plt.subplot(111)
        if chain is None:
            if step:
                times = xrange(self.chains.shape[0])
                for c in range(self.chains.shape[1]):
                    axes.step(times, self.chains[:,c], where='pre',
                              label='%i'%c, **kwds)
            else:
                for c in range(self.chains.shape[1]):
                    axes.plot(self.chains[:,c], **kwds)
        else:
            if step:
                times = xrange(self.chains.shape[0])
                axes.step(times, self.chains[:,chain], where='pre',
                          label='%i'%c, **kwds)
            else:
                axes.plot(self.chains[:,chain], **kwds)
        if xlabel:
            axes.set_xlabel(xlabel)
        else:
            axes.set_xlabel('Sample #')
        if ylabel:
            axes.set_ylabel(ylabel)
        else:
            axes.set_ylabel(self.name)
        if chain is None:
            axes.legend(fontsize='small', labelspacing=.2, borderpad=.3)
        axes.figure.show()  # needed for display update with axes

    def str(self, fmt=None):
        """
        Return a string summarizing fit results.

        If `fmt` is provided it is used as the format for the float values
        in point and interval estimates.  The default format is '.3g' for
        all parameters except log_p, for which it is '.2f'.
        """
        if fmt is None:
            if self.name == 'log_p':
                fmt = '.2f'  # log_p scale is absolute, ~1. per param
            else:
                fmt = '.3g'
        s = 'Parameter <{}> summary:\n'.format(self.name)
        s += 'Convergence and mixing diagnostics:  '
        s += 'Rhat = {:.2f}, ESS = {:d}\n'.format(self.Rhat, int(self.ess))
        s += 'Mean (se), median, sd:  {:{fmt}} ({:{fmt}}),  {:{fmt}},  {:{fmt}}\n'.format(
                self.mean, self.se_mean, self.median, self.sd, fmt=fmt)
        s += 'Central intvls:  50%: [{:{fmt}}, {:{fmt}}];  95%: [{:{fmt}}, {:{fmt}}]\n'.format(
                self.q25, self.q75, self.q025, self.q975, fmt=fmt)
        return s

    def __str__(self):
        return self.str()


def fitparams2attrs(fit, obj):
    """
    Extract parameter space info from a Stan fit object, storing it in
    attributes of the passed object `obj`.

    Extracted info includes (by attribute name):

    `par_names` : list of names of model parameters (unicode strings), not
        including the log_p "parameter" also tracked by Stan
 
    `par_dims` : dict of dimensions of parameters
 
    `par_attr_names` : dict of attribute names used to store parameter values
        in a StanFitResults instance; this is usually just the parameter name
        unless there is a collision with one of the initial attributes of
        the instance, in which case an underscore is appended to the name
    """
    # The _get... methods used here are used in pystan.model.StanModel, but
    # only *after* sampling or optimizing, and their results are not exposed
    # after optimizing.
    obj.par_names = fit._get_param_names()  # unicode param names
    obj.par_dims = {}
    for name, dim in zip(obj.par_names, fit._get_param_dims()):
        obj.par_dims[name] = dim

    # Make an index for accessing chains in fit.extract() results.
    # Note that 'lp__' is included here, and used in _make_param_handler.
    indx = 0
    obj.par_indx = {}
    for name in obj.par_names:
        obj.par_indx[name] = indx
        dims = obj.par_dims[name]
        if dims:
            indx += np.prod(dims)
        else:
            indx += 1  # technically could use prod(dims)=1. for dims=[]
    # obj.log_p_indx = obj.par_indx['lp__']

    # Stan includes log(prob) in the param list; we'll track it separately
    # so remove it from the param info.
    indx_of_lp = obj.par_names.index('lp__')
    del obj.par_names[indx_of_lp]
    del obj.par_dims['lp__']
    # del obj.par_indx['lp__']

    # Collect attribute names for storing param info, protecting from name
    # collision in the namespace of `obj`.
    # *** This doesn't protect against subsequent collision/overwriting of
    # parameter attributes by subsequent values. ***
    # TODO:  Make sure all needed class attributes are defined before this
    # runs, or otherwise protected.
    par_attr_names = {}
    for name in obj.par_names:
        if hasattr(obj, name):
            name_ = name + '_'
            if hasattr(obj, name_):
                raise ValueError('Cannot handle param name collision!')
            print('*** Access param "{0}" via "{0}_". ***'.format(name))
            par_attr_names[name] = name_
        else:
            par_attr_names[name] = name
    obj.par_attr_names = par_attr_names



class StanFitResults:
    """
    Container class storing all results from a Stan fit, i.e., a run of
    a StanModel instance's sample() command.
    """

    # These keys are from the raw summary col names; hope they won't change!
    # Map them to valid Python attribute names.
    col_map = {'mean':'mean',
               'se_mean' : 'se_mean',
               'sd' : 'sd',
               '2.5%' : 'q025',
               '25%' : 'q25',
               '50%' : 'median',
               '75%' : 'q75',
               '97.5%' : 'q975',
               'n_eff' : 'ess',
               'Rhat' : 'Rhat'}

    def __init__(self, fitter, stan_fit):
        """
        Gather results from a StanModel fit (a posterior sampling run),
        providing access via attributes.

        Parameters
        ----------

        fitter : StanFitter instance
            The StanFitter instance that implemented the fit; model properties
            describing the fit are accessed from `fitter`

        fit : PyStan fit instance
            PyStan fit object with results of a posterior sampling run
        """
        self.fitter = fitter
        self.fit = stan_fit
        self.when = datetime.datetime.now()
        fitparams2attrs(stan_fit, self)
        self._get_table_info()
        self._gather_sample_results()

    def _get_table_info(self):
        """
        Get information about the summary table from a fit to the current data.

        This information (largely dimensional/indexing) is in principle
        available once the model and data are both defined, but it is only
        available from Stan post-fit.
        """
        # TODO:  Some of this info might be better to get from fit.fit.sim
        # vs. from the summary table.

        # Collect info from the fit that shouldn't change if the fit is
        # re-run.
        self.raw_summary = self.fit.summary()  # dict of fit statistics (Rhat, ess...)
        # Column names list the various types of statistics.
        self.sum_cols = self.raw_summary['summary_colnames']
        # Get indices into the summary table for the columns.
        self.col_indices = {}
        for i, name in enumerate(self.sum_cols):
            self.col_indices[name] = i
        # Row names list the parameters & lp__; convert from an ndarray to a list.
        self.sum_rows = [name for name in self.raw_summary['summary_rownames']]

        # dict giving row number for each param item:
        self.item_indx = OrderedDict()
        for i, name in enumerate(self.sum_rows):
            self.item_indx[name] = i

    def _make_param_handler(self, name=None, indx=None, log_p=False):
        """
        Create a ParamHandler instance for parameter name `name` and make
        it an attribute, using data from (row,item) in the fit summary table.

        `name` should be the *base* name, i.e., excluding the index if the
        parameter is a vector or array.

        Call with (name) for a scalar parameter.

        Call with (name, indx) for an element of a vector/matrix/array
        parameter.

        Call with (log_p=True) for log(prob).
        """
        # TODO:  This gets data from the raw_summary table; rewrite to get it
        # from the self.fit.sim dict???

        # Set the key to use for Stan table lookups.
        if log_p:
            key = 'lp__'
        else:
            key = name

        # Scalars and vectors handle names differently; vectors use `indx`.
        if indx is None:  # scalar case, including log_p
            fname = name  # full name to store in the handler
            permuted = self.permuted[key]
            row = self.sum_rows.index(key)
            chains = self.chains[:,:,row]
        else:  # array case
            s_indx = [str(i) for i in indx]
            fname = name + '[' + ','.join(s_indx) + ']'
            row = self.sum_rows.index(fname)
            chains = self.chains[:,:,row]
            # We want something like self.permuted[key][:,indx], except the
            # last slice isn't valid syntax.
            # Get permuted samples but with 1st axis last, to make it easy to
            # grab a row of a multidimen array.
            axes = list(range(len(indx)+1))  # +1 for the sample dimen
            axes = axes[1:] + axes[:1]
            swapped = self.permuted[key].transpose(*axes)
            permuted = swapped[indx]

            # TODO: Is there a better way to do that using a full slice made
            # using slice(None), or a multi_index?

        param = ParamHandler(fit=self.fit, name=fname)
        param['permuted'] = permuted  # a 1-D chain, but correlations hidden
        param['chains'] = chains  # stored as chains[t,c] = time t in chain c
        for stat in self.sum_cols:
            col = self.col_indices[stat]
            param[self.col_map[stat]] = self.summaries[row,col]
        # 95% central credible interval:
        param['intvl95'] = (param['q025'], param['q975'])
        return param

    def _gather_sample_results(self):
        """
        Define attributes holding results from the current fit.
        """
        # Get data from the fit results using .extract, from 'stanfit4model.pyx'.

        # Extract chains, kept separate and ordered (permuted=False), with
        # burn-in discarded (inc_warmup=False, default), as an array indexed as
        # [sample #, chain #, param #]; note that log_p is added to the end
        # of the param list.  This is an array.
        self.chains = self.fit.extract(permuted=False)

        # Collect samples from the chains, merged via random permutation
        # (permuted=True), with burn-in discarded (inc_warmup=False), as a
        # param-keyed dict.  This is an OrderedDict.
        self.permuted = self.fit.extract(permuted=True)
        self.summaries = self.raw_summary['summary']

        # Populate namespace with handlers for each param, holding
        # various data from the fit.  Collect ESS & Rhat vals (but
        # not for log_p).
        self.esses = []
        self.Rhats = []
        for name in self.par_names:
            attr_name = self.par_attr_names[name]
            if not self.par_dims[name]:  # scalar param case
                param = self._make_param_handler(name)
                setattr(self, attr_name, param)
                self.esses.append(param.ess)
                self.Rhats.append(param.Rhat)
            else:  # vector/array cases - build an obj array of param instances
                a = np.empty(self.par_dims[name], np.object)
                for indx in np.ndindex(*self.par_dims[name]):
                    param = self._make_param_handler(name, indx)
                    a[indx] = param
                    self.esses.append(param.ess)
                    self.Rhats.append(param.Rhat)
                setattr(self, attr_name, a)
        self.esses = np.array(self.esses)
        self.Rhats = np.array(self.Rhats)

        # Get minimum ESS, to guide thinning.
        self.min_ess = self.esses.min()

        # Make a handler for log_p, the last "parameter" in the Stan table.
        param = self._make_param_handler('log_p', log_p=True)
        setattr(self, 'log_p', param)
        self.min_ess = min(self.min_ess, param.ess)

        # Provide samples merged from thinned chains.  These are views of
        # the chains; the data are not copied.
        clen, nc, npar = self.chains.shape  # chain length, # chains, # params
        tb = self.thinned_by = int(np.ceil(clen / self.min_ess))
        for name in self.par_names:
            attr_name = self.par_attr_names[name]
            if not self.par_dims[name]:  # scalar param
                param = getattr(self, attr_name)
                # Note that a chain is a *column*, not a row.
                thinned = param.chains[::tb,:]
                param.thinned = np.ravel(thinned, order='F')
            elif len(self.par_dims[name]) == 1:  # vector param as list
                params = getattr(self, attr_name)
                for param in params:
                    thinned = param.chains[::tb,:]
                    param.thinned = np.ravel(thinned, order='F')
        param = getattr(self, 'log_p')
        thinned = param.chains[::tb,:]
        param.thinned = np.ravel(thinned, order='F')
        self.n_thinned = param.thinned.shape[0]

    def subsample_indices(self, n):
        """
        Return a set of indices defining a random subsample of size n from the
        merged, thinned chains.
        """
        if n > self.n_thinned:
            raise ValueError('Requested sample size > thinned chain length!')
        return random.choice(self.n_thinned, n)

    def point(self, i):
        """
        Return a point in parameter space corresponding to sample `i` in the
        thinned, merged chain for each parameter.  The point is returned as an
        object with both a dict and an attribute interface to the parameter
        values, accessed by parameter name.
        """
        if i > self.n_thinned:
            raise ValueError('Requested sample is beyond thinned chain length!')
        d = {}
        for name in self.par_names:
            attr_name = self.par_attr_names[name]
            if not self.par_dims[name]:  # scalar param
                param = getattr(self, name)
                d[attr_name] = param.thinned[i]
            elif len(self.par_dims[name]) == 1:  # vector param as list
                params = getattr(self, attr_name)
                l = []
                for param in params:
                    l.append(param.thinned[i])
                d[attr_name] = np.array(l)
        d['log_p'] = getattr(self, 'log_p').thinned[i]
        return ImmutableAttrDict(d)

    def log_prob_upar(self, upar_array, adjust_transform=False):
        """
        Compute the log posterior PDF for the point in *unconstrained*
        parameter space specified by the array `upar_array`.

        Internally, Stan works in a parameter space in which the support
        for each parameter is the entire real line.  If a model parameter
        is constrained (e.g., must be positive), Stan internally transforms
        to an unconstrained version of the parameter.  This method takes
        unconstrained parameter values as its arguments.

        When `adjust_transform` is True, a log Jacobian term is added, as
        used by Stan internally.  It should be false for tasks such as
        finding the mode in the original parameter space.
        """
        return self.fit.log_prob(upar_array, adjust_transform)

    def stan_plot(self, par_names=None):
        """
        Create a new mpl figure with Stan's default summary plot,
        with a marginal PDF estimate and a traceplot produced for model
        parameters.  The traceplot is created by merging
        all chains and randomly permuting the compiled samples.

        If `par_names` is None, the plot will contain results for all
        parameters (in subplots as necessary).  Otherwise, it should be
        a list of names of parameters whose summary plots will be produced.

        Stan's plot is in fact PyMC's traceplot.

        The resulting figure instance is returned.
        """
        return self.fit.plot(par_names)

    def __str__(self):
        return str(self.fit)


class StanFitter:
    """
    Helper class for PyStan model fitting, providing automatic caching of
    a model, and easy access to fit results via attributes.
    """

    def __init__(self, source, data=None, n_chains=None, n_iter=None, 
                 name=None, n_jobs=-1, **kwds):
        """
        Prepare a Stan model; perform a fit (computing posterior samples
        and summary statistics) if `data`, `n_chains` and `n_iter` are
        provided.  If only a subset of these arguments are provided, save
        them for possible use in future fits run with the `sample()` method.

        If the model is new (or revised), it is compiled and the compiled
        code is cached.  If the model has been previously compiled (in the
        runtime directory), the cached code is used, accelerating startup.

        Parameters
        ----------

        source : string
            Path to a file (ending with ".stan") containing the Stan code for
            a model, or a string containing the code itself

        data : dict
            Dict of data corresponding to the model's data block

        n_chains : int
            Number of posterior sampler chains to run

        n_iter : int
            Number of iterations per chain for the initial run

        n_jobs : int, optional
            Sample in parallel if possible, using the multiprocessing module
            to distribute computations among the specified number of jobs.
            (Note that PyStan on Windows does not currently support
            multiprocessing.)  If -1, all CPUs are used.  All Windows runs
            use n_jobs=1.
        """
        self.name = name
        if source.count('\n') == 0 and source[-5:] == '.stan':
            with open(source, 'r') as sfile:
                self.code = sfile.read()
        else:
            self.code = source
        self.code_hash = md5(self.code.encode('ascii')).hexdigest()
        # ID is model name + hash, or just hash if no name:
        if name:
            self.id = '{}-{}'.format(name, self.code_hash)
        else:
            self.id = 'Anon-{}'.format(self.code_hash)
        self._compile()

        self.data = data
        self.n_chains = n_chains
        self.n_iter = n_iter
        self.set_n_jobs(n_jobs)

        if data:
            self.set_data(data)

        # An actual fit, if one is fully specified.
        if data is not None and n_chains is not None and n_iter is not None:
            fit = self.sample(data=data, chains=n_chains, iter=n_iter, n_jobs=n_jobs, **kwds)
            self.fits = [fit]
            return fit
        else:
            self.fits = None
            return None
    
    def _compile(self):
        """
        Compile a Stan model if necessary, loading a previously compiled
        version if available.
        """
        cache_path = 'cached-model-{}.pkl'.format(self.id)
        files = glob.glob(cache_path)
        if files:
            cache_path = files[0]
            self.name, self.id, self.model = pickle.load(open(files[0], 'rb'))
            print('Using cached StanModel from {}...'.format(files[0]))
        else:
            self.model = pystan.StanModel(model_code=self.code)
            with open(cache_path, 'wb') as f:
                pickle.dump((self.name, self.id, self.model), f)

    def set_n_jobs(self, n_jobs):
        """
        Set the number of multiprocessing jobs to use, adjusting the
        number to always be 1 on Windows platforms.

        If `n_jobs` is -1, all CPUs will be used (except on Windows).
        """
        if plat_is_win:
            self.n_jobs = 1
        else:
            self.n_jobs = n_jobs

    def set_data(self, data, seed=None):
        """
        Set the data info dictionary, collect info about parameters for an
        application of the model to the dataset, and initialize Stan's RNG.

        `seed` should be an `int` between 0 and MAX_UINT, inclusive, or NumPy's
        `np.random.RandomState`, whose `randint` method will be used to get
        a seed.  If `seed` is None, a random seed will be used.

        Note that since Stan supports hierarchical models, the
        parameter space may not be completely defined until a dataset is
        specified (the dataset size determines the number of latent
        parameters in hierarchical models).
        """
        self.data = data
        # Note: fit_class API changed to require seed after PyStan-2.14.
        seed = pystan.misc._check_seed(seed)
        self.fit = self.model.fit_class(self.data, seed)
        fitparams2attrs(self.fit, self)

    def sample(self, n_iter=None, n_chains=None, data=None, **kwds):
        """
        Run a posterior sampler using the compiled model, potentially using new
        data.

        The argument order was chosen to make it easiest to refit the same
        data with another (perhaps longer) run of the sampler; sample(n) does
        this.

        This skips the model compilation step, but otherwise runs a fresh
        MCMC chain.
        """
        if n_iter is None:
            n_iter = self.n_iter
        else:
            self.n_iter = n_iter
        if data is not None:
            self.set_data(data)
        if n_chains is None:
            n_chains = self.n_chains
        else:
           self.n_chains = n_chains
        self.n_iter = n_iter
        # The actual fit!
        start_time = timeit.default_timer()
        fit = self.model.sampling(data=self.data, chains=self.n_chains,
                  iter=self.n_iter, n_jobs=self.n_jobs, **kwds)
        elapsed = timeit.default_timer() - start_time
        # fit = pystan.stan(fit=self.fit, data=self.data, chains=self.n_chains,
        #                        iter=self.n_iter, **kwds)

        # *** Consider gathering model info from the 1st fit to a data set
        # here, e.g., as in _get_table_info().

        fit = StanFitResults(self, fit)
        fit.time_samp = elapsed
        return fit

    def mode(self, **kwds):
        """
        Return the mode of the posterior PDF as an object with both a dict
        and an attribute interface to the parameter values.

        Any keyword arguments are passed to PyStan's optimizing() method.
        See the docstring for self.model.optimizing for more info.  Do
        not provide an `as_vector` argument.
        """
        start_time = timeit.default_timer()
        mode_dict = self.model.optimizing(data=self.data, as_vector=False, **kwds)
        elapsed = timeit.default_timer() - start_time
        point = ImmutableAttrDict(mode_dict['par'])
        point.log_p = mode_dict['value']
        point.time_opt = elapsed
        return point


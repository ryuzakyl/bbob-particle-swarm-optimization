#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runs an entire experiment for benchmarking PURE_RANDOM_SEARCH on a testbed.

CAPITALIZATION indicates code adaptations to be made.
This script as well as files bbobbenchmarks.py and fgeneric.py need to be
in the current working directory.

"""

import time
import numpy as np
import fgeneric
import bbobbenchmarks as bn

# ------------------------ setting global parameters -----------------------------

datapath = 'results'

opts = dict(algid='vmml_meta', comments='my metaheuristic for the bbob challenge')

maxfunevals = 100

# minfunevals = 'dim + 2'  # PUT MINIMAL SENSIBLE NUMBER OF EVALUATIONS for a restart
minfunevals = 2

maxrestarts = 10000      # SET to zero if algorithm is entirely deterministic

t0 = time.time()

np.random.seed(int(t0))

# setting my own personal global parameters

# F128: Gallagher with 101 Gaussian peaks with Gauss noise, condition up to 1000, one global rotation
objective_function_id = 128


# --------------------------------------------------------------------------------


def run_optimizer(fun, dim, maxfunevals, ftarget=-np.Inf):
    """
    start the optimizer, allowing for some preparation.
    This implementation is an empty template to be filled 
    
    """
    # prepare
    x_start = 8. * np.random.rand(dim) - 4
    
    # call
    random_search(fun, x_start, maxfunevals, ftarget)


def random_search(fun, x, maxfunevals, ftarget):
    """
    samples new points uniformly randomly in [-5,5]^dim and evaluates
    them on fun until maxfunevals or ftarget is reached, or until
    1e8 * dim function evaluations are conducted.

    """
    dim = len(x)
    # maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = min(maxfunevals, 200)
    fbest = np.inf
    
    for iter in range(0, int(np.ceil(maxfunevals / popsize))):
        xpop = 10. * np.random.rand(popsize, dim) - 5.
        fvalues = fun(xpop)
        idx = np.argsort(fvalues)
        if fbest > fvalues[idx[0]]:
            fbest = fvalues[idx[0]]
            xbest = xpop[idx[0]]
        if fbest < ftarget:  # task achieved 
            break

    return xbest

# --------------------------------------------------------------------------------------

# creating the function object
f = fgeneric.LoggingFunction(datapath, **opts)

# for every direction
for dim in (2, 3, 5, 10, 20, 40):  # small dimensions first, for CPU reasons
    # for every function in bbobbenchmarks.py
    for f_name in bn.nfreeIDs:  # or bn.noisyIDs
        # for every instance in [1, 2, 3, 4, 5, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        for iinstance in range(1, 6) + range(21, 31):
            # setting function parameters
            f.setfun(*bn.instantiate(f_name, iinstance=iinstance))

            # independent restarts until maxfunevals or ftarget is reached
            for restarts in xrange(0, maxrestarts + 1):
                # restarting function in case the user has indicated so
                if restarts > 0:
                    f.restart('independent restart')  # additional info

                # running code to optimize all objective functions
                run_optimizer(f.evalfun, dim,  maxfunevals - f.evaluations, f.ftarget)

                # stop criteria
                if f.fbest < f.ftarget or f.evaluations + minfunevals > maxfunevals:
                    break

            # finalizing the run
            f.finalizerun()

            print('  f%d in %d-D, instance %d: FEs=%d with %d restarts, ''fbest-ftarget=%.4e, elapsed time [h]: %.2f'
                  % (f_name, dim, iinstance, f.evaluations, restarts,
                     f.fbest - f.ftarget, (time.time()-t0)/60./60.))

        print '      date and time: %s' % (time.asctime())

    print '---- dimension %d-D done ----' % dim


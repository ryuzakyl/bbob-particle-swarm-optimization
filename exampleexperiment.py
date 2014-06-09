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

datapath = 'results'

opts = dict(algid='vmml_meta', comments='my metaheuristic for the bbob challenge')

maxfunevals = '100'


def run_optimizer(fun, dim, maxfunevals, ftarget=-np.Inf):
    """start the optimizer, allowing for some preparation. 
    This implementation is an empty template to be filled 
    
    """
    # prepare
    x_start = 8. * np.random.rand(dim) - 4
    
    # call
    PURE_RANDOM_SEARCH(fun, x_start, maxfunevals, ftarget)


def PURE_RANDOM_SEARCH(fun, x, maxfunevals, ftarget):
    """samples new points uniformly randomly in [-5,5]^dim and evaluates
    them on fun until maxfunevals or ftarget is reached, or until
    1e8 * dim function evaluations are conducted.

    """
    dim = len(x)
    maxfunevals = min(1e8 * dim, maxfunevals)
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

minfunevals = 'dim + 2'  # PUT MINIMAL SENSIBLE NUMBER OF EVALUATIONS for a restart
maxrestarts = 10000      # SET to zero if algorithm is entirely deterministic

t0 = time.time()
np.random.seed(int(t0))

f = fgeneric.LoggingFunction(datapath, **opts)
for dim in (2, 3, 5, 10, 20, 40):  # small dimensions first, for CPU reasons
    for f_name in bn.nfreeIDs:  # or bn.noisyIDs
        for iinstance in range(1, 6) + range(21, 31):
            f.setfun(*bn.instantiate(f_name, iinstance=iinstance))

            # independent restarts until maxfunevals or ftarget is reached
            for restarts in xrange(0, maxrestarts + 1):
                if restarts > 0:
                    f.restart('independent restart')  # additional info

                run_optimizer(f.evalfun, dim,  eval(maxfunevals) - f.evaluations, f.ftarget)

                if f.fbest < f.ftarget or f.evaluations + eval(minfunevals) > eval(maxfunevals):
                    break

            f.finalizerun()

            print('  f%d in %d-D, instance %d: FEs=%d with %d restarts, ''fbest-ftarget=%.4e, elapsed time [h]: %.2f'
                  % (f_name, dim, iinstance, f.evaluations, restarts,
                     f.fbest - f.ftarget, (time.time()-t0)/60./60.))

        print '      date and time: %s' % (time.asctime())
    print '---- dimension %d-D done ----' % dim


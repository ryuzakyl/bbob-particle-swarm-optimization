#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runs an entire experiment for benchmarking PURE_RANDOM_SEARCH on a testbed.

CAPITALIZATION indicates code adaptations to be made.
This script as well as files bbobbenchmarks.py and fgeneric.py need to be
in the current working directory.

"""

import time

import fgeneric
import bbobbenchmarks as bn

from heuristics import *

# ------------------------ setting global parameters -----------------------------

# F24: Lunacek bi-Rastrigin Function
objective_function_id = 24

# dimension of the domain. each function is defined for several dimensions.
dimension = 50

# path where the algorithm results will be held
datapath = 'base-200'  # different folder for each experiment

# some details about the algorithm
opts = \
    {
        # id of the algorithm
        'algid': 'vmml_meta',

        # some comments on the algorithm
        'comments': 'my metaheuristic for the bbob challenge'
    }

# maximum number of function evaluations
maxfunevals = 10000

# minfunevals = 'dim + 2'  # PUT MINIMAL SENSIBLE NUMBER OF EVALUATIONS for a restart
minfunevals = 2

# maximum numbers of restarts
maxrestarts = 0      # SET to zero if algorithm is entirely deterministic

# lower and upper bound for each variable (box restriction)
lower_bound = -5
upper_bound = 5
box_restr = (lower_bound, upper_bound)

# setting the seed for the random number
t0 = time.time()
np.random.seed(int(t0))

# -----------------------------------------------------------------------------------------------


def perform_bbob_benchmarks(iinstances, trials):
    # creating the function object
    f = fgeneric.LoggingFunction(datapath, **opts)

    # cumulative sum of the errors of each 'run' of the heuristic
    result = 0.0

    # for each function instance
    for iinstance in iinstances:

        # for each trial
        for trial in trials:
            # initializing the becnhmark in the selected function
            f.setfun(*bn.instantiate(objective_function_id, iinstance=iinstance))

            # running the heuristic
            # _, _, = random_search(f.evalfun, dimension, maxfunevals, box_restr, f.ftarget)
            _, _, = pso(f.evalfun, dimension, maxfunevals, box_restr, f.ftarget)

            # printing information related to each 'run'
            print(
                '  f%d in %d-D, instance %d: FEs=%d, ''fbest-ftarget=%.4e, elapsed time [h]: %.2f\n' %
                (objective_function_id, dimension, iinstance, f.evaluations, f.fbest - f.ftarget, (time.time()-t0)/60./60.))

            # there's no need to store the best solution found in each 'run' because the benchmark
            # saves it ('the fitness') in 'f.fbest'. the value of the 'global optimum' found by our
            # heuristic is stored in 'f.ftarget' and the 'error' can be measured by 'f.fbest - f.ftarget'.

            #  computing cumulative sum for each 'run'
            result += f.fbest - f.ftarget

            # finilizing benchmark for the current 'run'
            f.finalizerun()

    return result

# -----------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # the iinstance parameter is a particular modification to the original 'objective_function_id'
    # bbob function. for example, maybe iinstance=1 means a 90 degree rotation of the 'objective_function_id'
    iinstances = [1, 2, 3, 4, 5]

    # amount of attempts for each 'iinstance' of the objective function
    trials = [1, 2, 3, 4, 5]

    # performing benchmarks
    cumulative_error = perform_bbob_benchmarks(iinstances, trials)

    # total of 'runs'
    total_runs = len(iinstances) * len(trials)

    # computing average error
    avg_error = cumulative_error / total_runs

    # displaying results
    print "----------------------------------------------------------------------------"
    print ""
    print "Error < 1.0e+003 (1000.0)\tGrade 3"
    print "Error < 7.0e+002 (700.0)\tGrade 4"
    print "Error < 4.5e+002 (450.0)\tGrade 5"
    print ""
    print "Average Error:\t%.4e (%.1f)" % (avg_error, avg_error)
    print ""
    print "----------------------------------------------------------------------------"
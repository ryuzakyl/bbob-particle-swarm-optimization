import numpy as np


def random_search(fun, dim, max_fun_evals=10000, box_restr=(-5, 5), f_target=-np.Inf):
    # stores the best value found
    f_best = np.Inf

    # getting the box restriction parameters
    lb, ub = box_restr
    interval_size = ub - lb

    iterations = 10
    solutions_it = max_fun_evals / iterations

    for i in xrange(iterations):
        x_pop = lb + np.random.rand(solutions_it, dim) * interval_size

        f_values = fun(x_pop)

        best_fit_index = np.argsort(f_values)[0]
        best_fit = f_values[best_fit_index]

        if best_fit < f_best:
            f_best = best_fit

        if f_best < f_target:
            break
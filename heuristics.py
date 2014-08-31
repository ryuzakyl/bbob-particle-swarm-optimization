import random as rnd
import numpy as np

# ---------------------------------------------------------------------------------------


def random_search(f, dim, max_fun_evals=10000, box_restr=(-5, 5), f_target=-np.Inf):
    # best element and it's fitness value
    x_best = None
    f_best = np.Inf

    # getting the box restriction parameters
    lb, ub = box_restr
    interval_size = ub - lb

    iterations = 10
    solutions_it = max_fun_evals / iterations

    for i in xrange(iterations):
        x_pop = lb + np.random.rand(solutions_it, dim) * interval_size

        # evaluation samples
        f_values = f(x_pop)

        best_fit_index = np.argsort(f_values)[0]
        best_fit = f_values[best_fit_index]

        if best_fit < f_best:
            x_best = x_pop[best_fit_index]
            f_best = best_fit

        if f_best < f_target:
            break

    # returns best sample and it's fitness value
    return x_best, f_best

# ---------------------------------------------------------------------------------------

# ---------------------- pso parameters ----------------------

# feasibility constraints
TRUNCATE = 0
PENALIZE = 1
REFLECT = 2

# neighbours criteria
RANDOM = 0      # the neighbours list is selected randomly
ADJACENT = 1    # the neighbours list is selected a priori (neigh_size/2 to left and right)
RAND_FIX = 2    # the neighbours list is selected randomly just once
SANDWICH = 3    # the neighbours are the previous and the next element

neigh_dict = dict()


# count of particles in swarm
swarm_size = 100

# maximum of 5 neighbours
# neigh_size = min(5, swarm_size / 10)
neigh_size = 80


# amount of velocity to be retained
alpha = 0.2

# how much of personal best is taken into account
#   if 'beta' is large => the particle will become a separate hill climber
#   if 'beta' is lower => it will be more of a global search rather that local search
beta = 0.2
beta_ini = 0.1
beta_end = 0.2

# how much of the global best is considered
#   if 'sigma' is large => particles tend to move towards the best known region
#   if 'sigma' is lower => more like separate hill climbers
sigma = 0.3  # just for now
sigma_ini = 0.3
sigma_end = 0.4

# how much of the neighbours (informants) is considered (mid-ground between 'beta' and 'sigma')
# more neighbours means that points to a global best instead of a particle's local best
# gamma = (beta + sigma) / 2.0
gamma = 0.3
gamma_ini = 0.4
gamma_end = 0.3

# how fast the particle moves
# if 'epsilon' is large => the particle might miss some important areas
# if 'epsilon' is lower => do fine-grain optimization (kind of like hill climbing)
epsilon = 2.5  # just for now
epsilon_ini = 2.5
epsilon_end = 2.0

# -----------------------------------------------------------

# random particle generator
part_gen = lambda space_dim, lb, ub: lb + np.random.rand(space_dim) * (ub - lb)

# ---------------------- pso algorithm ----------------------


def pso(f, dim, max_fun_evals=10000, box_restr=(-5, 5), f_target=-np.Inf):
    # the particle swarm
    x = init_swarm(swarm_size, dim, box_restr)
    x_prev = init_swarm(swarm_size, dim, box_restr)  # previous position (values) of the particles

    # best solutions found by each particle
    x_aster = x.copy()
    f_aster = np.array([np.Inf] * swarm_size, np.float32)

    # best solutions found by each neighbourhood
    x_plus = x.copy()
    f_plus = np.array([np.Inf] * swarm_size, np.float32)

    # initializing best global solution
    x_best = None
    f_best = np.Inf

    # getting box restrictions
    lb, ub = box_restr

    # amount of iterations
    iter_count = max_fun_evals / swarm_size

    curr_beta = beta_ini
    beta_dx = (beta_end - beta_ini) / iter_count

    curr_gamma = gamma_ini
    gamma_dx = (gamma_end - gamma_ini) / iter_count

    curr_sigma = sigma_ini
    sigma_dx = (sigma_end - sigma_ini) / iter_count

    curr_epsilon = epsilon_ini
    epsilon_dx = (epsilon_end - epsilon_ini) / iter_count

    # iterating over the particle swarm
    for iteration in xrange(iter_count):
        # keeping particles in [lb, ub]
        for k in xrange(swarm_size):
            # if particle out of limits
            if np.any(np.logical_or(x[k] < lb, x[k] > ub)):
                # generating random particle inside limits
                new_part = part_gen(50, lb, ub)
                x[k] = new_part
                # assigning 'zero' to the direction of such particle
                x_prev[k] = new_part

        # fitness of each particle in the current iteration
        f_curr = f(x)

        # for each particle in the swarm
        for p in xrange(swarm_size):
            # updating 'x_aster' fittest solution
            if f_curr[p] < f_aster[p]:
                x_aster[p] = x[p]
                f_aster[p] = f_curr[p]

            # updating 'x_plus' fittest solution
            neighbours = get_neighbours(p, neigh_type=RANDOM)

            best_n_index = np.argmin([f_curr[k] for k in neighbours])
            best_n_index = best_n_index if f_curr[best_n_index] < f_curr[p] else p

            if f_curr[best_n_index] < f_plus[p]:
                x_plus[p] = x[best_n_index]
                f_plus[p] = f_curr[best_n_index]

            # updating 'best' particle and it's fitness value
            if f_curr[p] < f_best:
                x_best = x[p]
                f_best = f_curr[p]

            # updating velocity of particle p
            v = x[p] - x_prev[p]
            for i in xrange(dim):
                # random in [0, beta]
                b = rnd.random() * curr_beta
                # random in [0, gamma]
                c = rnd.random() * gamma
                # random in [0, sigma]
                d = rnd.random() * curr_sigma

                xi = x[p][i]
                s1 = alpha * v[i]
                s2 = b * (x_aster[p][i] - xi)
                s3 = c * (x_plus[p][i] - xi)
                s4 = d * (x_best[i] - xi)
                v[i] = s1 + s2 + s3 + s4

            # updating x_prev
            x_prev[p] = x[p]

            # moving particle p
            x[p] += epsilon * v

        # the optimum has been reached
        if f_best < f_target:
            break

        # updating values for beta, sigma and epsilon
        curr_beta += beta_dx
        curr_gamma += gamma_dx
        curr_sigma += sigma_dx
        curr_epsilon += epsilon_dx

    # returns best sample and it's fitness value
    return x_best, f_best


def init_swarm(n, dim, box_restr):
    # getting the box restriction parameters
    x_min, x_max = box_restr

    # generating a swarm of 'count' particles
    return np.array([part_gen(dim, x_min, x_max) for i in xrange(n)])


# this 'neighbourhood' does not include the particle 'p' because that is used in the caller
def get_neighbours(p_index, neigh_type=RANDOM):
    if neigh_type == RANDOM:
        # getting list of neighbours
        return rnd.sample(xrange(swarm_size), neigh_size)

    elif neigh_type == ADJACENT:
        half = neigh_size / 2

        # left half
        left = [item % swarm_size for item in range(p_index - half, p_index)]

        # right half
        right = [item % swarm_size for item in range(p_index + 1, (p_index + 1) + half)]

        # getting list of neighbours
        return left + right

    elif neigh_type == RAND_FIX:
        if p_index not in neigh_dict.keys():
            # candidates list
            elegible_particles = range(0, p_index) + range(p_index, swarm_size)
            neigh_particles = rnd.sample(elegible_particles, neigh_size)
            neigh_dict[p_index] = neigh_particles

        return neigh_dict[p_index]

    elif neigh_type == SANDWICH:
        return [(p_index - 1) % swarm_size, (p_index + 1) % swarm_size]

    else:
        raise Exception('Unknown particle neghbourhood type.')
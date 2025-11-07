import numpy as np
from .critical_params import compute_critical_mu_u
from .algorithm1 import algorithm1_local_search
from .bisis import simulate_fixed_point


def run_epsilon_sweep(A, tau1, tau2, eps_list, iters=1, costs=None):
    mu_c, u_c, x_star = compute_critical_mu_u(A, tau1, tau2)

    results = []  # list of (eps, avgX, avgY)
    for eps in eps_list:
        u = algorithm1_local_search(A, tau1, tau2, mu_c, u_c, x_star, eps, iters=iters, costs=costs)
        x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u)
        results.append((float(eps), float(x.mean()), float(y.mean())))
    return mu_c, u_c, x_star, results

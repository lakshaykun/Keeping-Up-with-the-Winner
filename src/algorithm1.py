import numpy as np
from numpy.linalg import eig


def pf_eigenvector(M):
    w, v = eig(M)
    idx = np.argmax(w.real)
    return v[:, idx].real


def algorithm1_local_search(A, tau1, tau2, mu_c, u_c, x_star, eps, iters=1, costs=None):
    """Implements the one-step perturbation-based local search (Algorithm 1).
    We respect the weighted budget constraint sum_i w_i * delta_i = 0 by pairing
    +eps and -eps assignments according to descending (nu_i / w_i).
    """
    N = len(u_c)
    Sx = np.diag(1.0 - x_star)

    if costs is None:
        w = np.ones(N)
    else:
        w = np.asarray(costs).astype(float)
        w[w <= 0] = 1.0

    u = u_c.copy()

    for _ in range(iters):
        B = Sx @ (A + mu_c * np.outer(u, u))
        nu = pf_eigenvector(B)  # PF eigenvector around critical point

        score = nu / w
        order = np.argsort(-score)

        delta = np.zeros(N)
        half = N // 2
        pos_idx = order[:half]
        neg_idx = order[half:]

        delta[pos_idx] = +eps
        delta[neg_idx] = -eps

        # Project to [0,1] and re-center to satisfy approximate weighted zero-sum
        u_new = np.clip(u + delta, 0.0, 1.0)

        # Optional fine correction to enforce sum(w*delta)=0 approximately
        gap = np.dot(w, (u_new - u))
        if abs(gap) > 1e-12:
            # shift uniformly over free variables
            adjust = gap / w.sum()
            u_new = np.clip(u_new - adjust, 0.0, 1.0)

        u = u_new

    return u
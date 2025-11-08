import numpy as np
from numpy.linalg import eig

def pf_eigenvector(M):
    w, v = eig(M)
    idx = np.argmax(w.real)
    return v[:, idx].real

def algorithm1_local_search(
    A,
    tau1,
    tau2,
    mu_c,
    u_init,
    x_star,
    eps,
    iters=1,
    costs=None,
    budget_C=None
):
    """
    Local perturbation-based optimization respecting a custom budget constraint:
        sqrt(mu_c) * sum_i w_i u_i = budget_C
    """
    N = len(u_init)
    u = u_init.copy()

    if costs is None:
        w = np.ones(N)
    else:
        w = np.asarray(costs).astype(float)
        w[w <= 0] = 1.0

    for _ in range(iters):
        Sx = np.diag(1.0 - x_star)
        B = Sx @ (A + mu_c * np.outer(u, u))
        nu = pf_eigenvector(B)

        # Perturbation direction
        score = nu / w
        order = np.argsort(-score)

        delta = np.zeros(N)
        half = N // 2
        delta[order[:half]] = +eps
        delta[order[half:]] = -eps

        # Apply and clip
        u_new = np.clip(u + delta, 0, 1)

        # Enforce budget constraint
        if budget_C is not None:
            current = np.sqrt(mu_c) * np.dot(w, u_new)
            if current > 0:
                scale = budget_C / current
                u_new = np.clip(u_new * scale, 0, 1)

        u = u_new

    return u

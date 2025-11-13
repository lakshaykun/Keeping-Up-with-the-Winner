import numpy as np
from numpy.linalg import eig

def pf_eigenvector(M):
    """Compute Perron-Frobenius eigenvector efficiently."""
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
        
    Optimized version with minimal overhead.
    """
    N = len(u_init)
    u = u_init.copy()

    if costs is None:
        w = np.ones(N, dtype=np.float64)
    else:
        w = np.asarray(costs, dtype=np.float64)
        w = np.maximum(w, 1.0)

    # Pre-compute Sx once (it doesn't change)
    Sx = np.diag(1.0 - x_star)
    half = N // 2

    for _ in range(iters):
        # Compute B
        B = Sx @ (A + mu_c * np.outer(u, u))
        
        # Get Perron-Frobenius eigenvector
        nu = pf_eigenvector(B)

        # Compute scores and get top/bottom nodes
        score = nu / w
        order = np.argsort(-score)
        
        # Apply perturbations
        delta = np.zeros(N, dtype=np.float64)
        delta[order[:half]] = eps
        delta[order[half:]] = -eps

        # Apply and clip
        u_new = np.clip(u + delta, 0.0, 1.0)

        # Enforce budget constraint if specified
        if budget_C is not None:
            current = np.sqrt(mu_c) * np.dot(w, u_new)
            if current > 1e-10:
                u_new *= (budget_C / current)
                u_new = np.clip(u_new, 0.0, 1.0)

        u = u_new

    return u

import numpy as np
from numpy.linalg import eig
from scipy.integrate import solve_ivp


def compute_x_star(A, tau1, T=2000.0):
    N = A.shape[0]
    def rhs(t, x):
        return tau1 * (1.0 - x) * (A @ x) - x
    x0 = 0.01 * np.ones(N)
    sol = solve_ivp(rhs, (0.0, T), x0, method="RK45", max_step=1.0, rtol=1e-6, atol=1e-8)
    return sol.y[:, -1]


def pf_eigenpair(M):
    w, v = eig(M)
    idx = np.argmax(w.real)
    return w[idx].real, v[:, idx].real


def compute_critical_mu_u(A, tau1, tau2):
    """Implement Lemma 3.2 to obtain (mu_c, u_c) given A, tau1, tau2.
    Steps:
      1) Compute x* by single-SIS of product 1.
      2) Sx = diag(1 - x*)
      3) PF eigenvector v of Sx A
      4) u_c ∝ Sx^{-1} v
      5) mu_c = (1/tau2 - lambda_max(Sx A)) / (u_c^T Sx u_c)
    """
    x_star = compute_x_star(A, tau1)
    Sx = np.diag(1.0 - x_star)
    lam, v = pf_eigenpair(Sx @ A)

    # u_c as scaled Sx^{-1} v
    # solve Sx u = v  -> u = Sx^{-1} v (Sx is diagonal)
    inv_diag = 1.0 / np.diag(Sx)
    u_c = inv_diag * v
    u_c = np.abs(u_c)
    if u_c.sum() > 0:
        u_c = u_c / u_c.sum()

    denom = u_c.T @ Sx @ u_c
    mu_c = (1.0 / tau2 - lam) / denom
    mu_c = max(mu_c, 0.0)

    return mu_c, u_c, x_star
import numpy as np
from scipy.integrate import solve_ivp


def bisis_rhs(t, z, A, tau1, tau2, mu, u):
    N = A.shape[0]
    x = z[:N]
    y = z[N:]
    S = 1.0 - x - y
    B = A + mu * np.outer(u, u)
    dx = tau1 * S * (A @ x) - x
    dy = tau2 * S * (B @ y) - y
    return np.concatenate([dx, dy])


def simulate_fixed_point(A, tau1, tau2, mu, u, T=2000.0, x0=None, y0=None):
    N = A.shape[0]
    if x0 is None:
        x0 = 0.01 * np.ones(N)
    if y0 is None:
        y0 = 0.01 * np.ones(N)
    z0 = np.concatenate([x0, y0])

    sol = solve_ivp(
        bisis_rhs,
        t_span=(0.0, T),
        y0=z0,
        args=(A, tau1, tau2, mu, u),
        method="RK45",
        max_step=1.0,
        rtol=1e-6,
        atol=1e-8,
    )
    N = A.shape[0]
    x = sol.y[:N, -1]
    y = sol.y[N:, -1]
    return x, y
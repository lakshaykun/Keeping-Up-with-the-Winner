import numpy as np
from scipy.integrate import solve_ivp


def bisis_rhs(t, z, A, tau1, tau2, mu, u, uu_outer):
    """
    Optimized RHS computation for bi-SIS model.
    Pre-computed outer product passed to avoid recomputation.
    """
    N = A.shape[0]
    x = z[:N]
    y = z[N:]
    
    # Vectorized susceptible computation
    S = 1.0 - x - y
    
    # Use pre-computed outer product
    B = A + uu_outer
    
    # Vectorized dynamics
    dx = tau1 * S * (A @ x) - x
    dy = tau2 * S * (B @ y) - y
    
    return np.concatenate([dx, dy])


def simulate_fixed_point(A, tau1, tau2, mu, u, T=2000.0, x0=None, y0=None):
    """
    Optimized fixed-point simulation with pre-computation and better defaults.
    """
    N = A.shape[0]
    
    # Use vectorized initialization
    if x0 is None:
        x0 = np.full(N, 0.01, dtype=np.float64)
    if y0 is None:
        y0 = np.full(N, 0.01, dtype=np.float64)
    
    z0 = np.concatenate([x0, y0])
    
    # Pre-compute outer product (constant throughout integration)
    uu_outer = mu * np.outer(u, u)

    # Use optimized RHS with pre-computed values
    sol = solve_ivp(
        lambda t, z: bisis_rhs(t, z, A, tau1, tau2, mu, u, uu_outer),
        t_span=(0.0, T),
        y0=z0,
        method="RK45",
        max_step=1.0,
        rtol=1e-6,
        atol=1e-8,
        dense_output=False,  # Don't store dense output (saves memory)
        vectorized=False
    )
    
    # Extract final state
    x = sol.y[:N, -1]
    y = sol.y[N:, -1]
    
    return x, y

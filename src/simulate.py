import numpy as np


def simulate_portfolio_losses(mu, sigma, weights, T=10, N=10_000, seed=42):
    """
    Simulates portfolio losses over a T-day horizon using Monte Carlo (Gaussian).

    Uses np.random.default_rng for thread-safe, reproducible draws that do not
    pollute global random state (unlike the legacy np.random.seed / np.random.seed).
    """
    rng = np.random.default_rng(seed)
    mu_scaled    = T * np.asarray(mu,    float)
    sigma_scaled = T * np.asarray(sigma, float)

    simulated_returns = rng.multivariate_normal(mu_scaled, sigma_scaled, size=N)
    portfolio_returns = simulated_returns @ np.asarray(weights, float)
    return -portfolio_returns


def simulate_t_dist_losses(mu, sigma, weights, T=10, N=10_000, df=5, seed=42):
    """
    Monte Carlo portfolio losses using a multivariate Student-t model
    (fat tails via normal / chi-square mixture).

    Uses np.random.default_rng — no global seed side-effects.
    """
    rng = np.random.default_rng(seed)
    d = len(mu)

    mu_scaled    = np.asarray(mu,    float) * T
    sigma_scaled = np.asarray(sigma, float) * T

    Z   = rng.multivariate_normal(np.zeros(d), sigma_scaled, size=N)
    chi = rng.chisquare(df, size=N) / df          # shape (N,)

    t_samples        = mu_scaled + Z / np.sqrt(chi)[:, None]
    portfolio_returns = t_samples @ np.asarray(weights, float)
    return -portfolio_returns


# ------------------------------------------------------------------
# Frozen-scenario helpers (used by the optimizer)
# ------------------------------------------------------------------

def draw_frozen_scenarios(mu, sigma, T, N, seed=0):
    """
    Pre-draw a fixed (N, d) scenario matrix from N(0, Σ·T).

    Shifting by μ·T is deferred to portfolio_losses_frozen so the same
    noise matrix is reused across all optimizer iterations — turning the
    stochastic CVaR objective into a deterministic one for the solver.

    Returns
    -------
    Z         : ndarray (N, d)  — zero-mean Gaussian draws scaled to horizon T
    mu_scaled : ndarray (d,)    — μ * T  (horizon drift)
    """
    rng          = np.random.default_rng(seed)
    sigma_scaled = T * np.asarray(sigma, float)
    mu_scaled    = T * np.asarray(mu,    float)
    Z = rng.multivariate_normal(np.zeros(len(mu)), sigma_scaled, size=N)
    return Z, mu_scaled


def portfolio_losses_frozen(Z, mu_scaled, weights):
    """
    Compute portfolio losses from a pre-drawn scenario matrix.
    Fully deterministic given Z — safe to call inside a scipy optimizer loop.
    """
    w                = np.asarray(weights, float)
    portfolio_returns = (Z + mu_scaled) @ w
    return -portfolio_returns
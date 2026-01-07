import numpy as np
from scipy.optimize import minimize
from src.simulate import simulate_portfolio_losses
from src.risk_metrics import compute_cvar



# Caps per asset invested. Needs a minimum of 5 assets 
def _max_weight_by_count(d: int) -> float:
    if d <= 5:
        return 0.20
    elif d == 6:
        return 0.19
    elif d == 7:
        return 0.18
    elif d == 8:
        return 0.17
    elif d == 9:
        return 0.16
    elif d == 10:
        return 0.15
    elif d == 11:
        return 0.14
    elif d == 12:
        return 0.13
    elif d == 13:
        return 0.12
    elif d == 14:
        return 0.11
    else:  # d >= 15
        return 0.10


# Core static optimizer (your original behavior)
def minimize_cvar(mu, sigma, T, N=100_000, alpha=0.95):
    """
    CVaR-minimizing weights using provided mu, sigma (static).
    Keeps your dynamic single-name caps based on the number of assets.
    """
    d = len(mu)

    def objective(w):
        losses = simulate_portfolio_losses(mu, sigma, w, T=T, N=N)
        return compute_cvar(losses, alpha)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    max_weight = _max_weight_by_count(d)
    bounds = [(0.0, max_weight)] * d
    w0 = np.ones(d) / d

    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return res.x


# EWMA stats (for dynamic optimization)
def _ewma_mean_cov(returns_df, lam=0.97):
    """
    Exponentially weighted mean & covariance (RiskMetrics-style) on a df of
    daily log returns (rows=time, cols=assets). Newer rows get higher weight
    """
    R = returns_df.dropna().values  # shape (T, d)
    Tn, d = R.shape

    # weights: w_t ∝ (1-lam)*lam^{age}, newest row gets largest weight
    w = (1.0 - lam) * lam ** np.arange(Tn - 1, -1, -1)
    w = w / w.sum()

    mu = (R * w[:, None]).sum(axis=0)

    X = R - mu  # broadcast
    cov = np.dot((X.T * w), X)

    # light diagonal shrinkage for stability on short windows
    gamma = 0.10
    cov = (1 - gamma) * cov + gamma * np.diag(np.diag(cov))

    return mu, cov


# Dynamic optimizer from short recent window
def minimize_cvar_from_returns(
    returns_df,
    T,
    N=8_000,
    alpha=0.95,
    method="ewma",          # "ewma" or "sample"
    window_days=30,         # 10–45 recommended for reactive trading
    lam=0.97                # EWMA decay (higher = slower decay)
):
    """
    Convenience wrapper: compute (mu, sigma) on a SHORT recent window,
    then minimize CVaR subject to your caps.
    """
    recent = returns_df.dropna().iloc[-window_days:]
    if recent.shape[0] < 5:
        raise ValueError("Not enough rows in recent window to estimate statistics.")

    if method == "ewma":
        mu, sigma = _ewma_mean_cov(recent, lam=lam)
    elif method == "sample":
        mu = recent.mean().values
        sigma = recent.cov().values
    else:
        raise ValueError("method must be 'ewma' or 'sample'")

    w_opt = minimize_cvar(mu, sigma, T=T, N=N, alpha=alpha)  # reuse static solver
    return w_opt, mu, sigma


# One entry point to choose static vs dynamic
def optimize_cvar(
    mode,
    T,
    N=8_000,
    alpha=0.95,
    mu=None,
    sigma=None,
    returns_df=None,
    window_days=30,
    lam=0.97,
    method="ewma",  # only used if mode="dynamic"
):
    """
    mode: "static"  -> use provided mu, sigma with minimize_cvar(...)
          "dynamic" -> compute mu, sigma from returns_df on a short window, then optimize
    """
    if mode == "static":
        if mu is None or sigma is None:
            raise ValueError("For mode='static', provide mu and sigma.")
        return minimize_cvar(mu, sigma, T=T, N=N, alpha=alpha)

    elif mode == "dynamic":
        if returns_df is None:
            raise ValueError("For mode='dynamic', provide returns_df.")
        w_opt, mu_hat, sigma_hat = minimize_cvar_from_returns(
            returns_df=returns_df,
            T=T,
            N=N,
            alpha=alpha,
            method=method,
            window_days=window_days,
            lam=lam,
        )
        return w_opt

    else:
        raise ValueError("mode must be 'static' or 'dynamic'")


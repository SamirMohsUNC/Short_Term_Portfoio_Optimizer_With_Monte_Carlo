import numpy as np
from scipy.optimize import minimize
from src.simulate import (
    simulate_portfolio_losses,
    draw_frozen_scenarios,
    portfolio_losses_frozen,
)
from src.risk_metrics import compute_cvar


# ---------------------------------------------------------------------------
# Single-name weight caps (unchanged from original)
# ---------------------------------------------------------------------------

def _max_weight_by_count(d: int) -> float:
    if d <= 5:   return 0.20
    elif d == 6: return 0.19
    elif d == 7: return 0.18
    elif d == 8: return 0.17
    elif d == 9: return 0.16
    elif d == 10: return 0.15
    elif d == 11: return 0.14
    elif d == 12: return 0.13
    elif d == 13: return 0.12
    elif d == 14: return 0.11
    else:         return 0.10


# ---------------------------------------------------------------------------
# Core static optimizer — now uses frozen scenarios
# ---------------------------------------------------------------------------

def minimize_cvar(mu, sigma, T, N=100_000, alpha=0.95, prev_weights=None, turnover_penalty=0.0):
    """
    CVaR-minimising weights.

    Improvement over original: scenarios are drawn ONCE before the optimiser
    starts, then reused on every objective evaluation.  This makes the objective
    fully deterministic for the solver (no Monte-Carlo noise between iterations),
    which means SLSQP converges faster and more reliably.

    Parameters
    ----------
    prev_weights : array-like, optional
        Weights from the previous rebalance.  When provided together with a
        positive `turnover_penalty`, the objective becomes:
            CVaR(w)  +  turnover_penalty * ||w - prev_weights||_1
        This discourages unnecessary churn without removing the CVaR focus.
    turnover_penalty : float, default 0.0
        L1 turnover cost coefficient (e.g. 0.001–0.01 is typically sufficient).
    """
    d  = len(mu)
    mu = np.asarray(mu, float)

    # Draw scenarios once — reused on every SLSQP iteration
    Z, mu_scaled = draw_frozen_scenarios(mu, sigma, T=T, N=N, seed=42)

    prev_w = np.asarray(prev_weights, float) if prev_weights is not None else None

    def objective(w):
        losses = portfolio_losses_frozen(Z, mu_scaled, w)
        cvar   = compute_cvar(losses, alpha)
        if prev_w is not None and turnover_penalty > 0.0:
            cvar += turnover_penalty * float(np.abs(w - prev_w).sum())
        return cvar

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    max_weight  = _max_weight_by_count(d)
    bounds      = [(0.0, max_weight)] * d
    w0          = np.ones(d) / d

    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return res.x


# ---------------------------------------------------------------------------
# EWMA statistics (for dynamic optimisation)
# ---------------------------------------------------------------------------

def _ewma_mean_cov(returns_df, lam=0.97):
    """
    Exponentially-weighted mean & covariance (RiskMetrics-style).
    Newer rows receive higher weight.
    """
    R      = returns_df.dropna().values
    Tn, d  = R.shape

    w = (1.0 - lam) * lam ** np.arange(Tn - 1, -1, -1)
    w = w / w.sum()

    mu = (R * w[:, None]).sum(axis=0)
    X  = R - mu
    cov = np.dot((X.T * w), X)

    # Light diagonal shrinkage for numerical stability on short windows
    gamma = 0.10
    cov   = (1 - gamma) * cov + gamma * np.diag(np.diag(cov))
    return mu, cov


# ---------------------------------------------------------------------------
# Dynamic optimiser wrapper
# ---------------------------------------------------------------------------

def minimize_cvar_from_returns(
    returns_df,
    T,
    N=8_000,
    alpha=0.95,
    method="ewma",
    window_days=30,
    lam=0.97,
    prev_weights=None,
    turnover_penalty=0.0,
):
    """
    Compute (mu, sigma) on a short recent window, then minimise CVaR.
    """
    recent = returns_df.dropna().iloc[-window_days:]
    if recent.shape[0] < 5:
        raise ValueError("Not enough rows in recent window to estimate statistics.")

    if method == "ewma":
        mu, sigma = _ewma_mean_cov(recent, lam=lam)
    elif method == "sample":
        mu    = recent.mean().values
        sigma = recent.cov().values
    else:
        raise ValueError("method must be 'ewma' or 'sample'")

    w_opt = minimize_cvar(
        mu, sigma, T=T, N=N, alpha=alpha,
        prev_weights=prev_weights,
        turnover_penalty=turnover_penalty,
    )
    return w_opt, mu, sigma


# ---------------------------------------------------------------------------
# Unified entry-point
# ---------------------------------------------------------------------------

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
    method="ewma",
    prev_weights=None,
    turnover_penalty=0.0,
):
    """
    mode: "static"  -> use provided mu, sigma
          "dynamic" -> compute mu, sigma from returns_df on a short window
    """
    if mode == "static":
        if mu is None or sigma is None:
            raise ValueError("For mode='static', provide mu and sigma.")
        return minimize_cvar(
            mu, sigma, T=T, N=N, alpha=alpha,
            prev_weights=prev_weights,
            turnover_penalty=turnover_penalty,
        )

    elif mode == "dynamic":
        if returns_df is None:
            raise ValueError("For mode='dynamic', provide returns_df.")
        w_opt, _, _ = minimize_cvar_from_returns(
            returns_df=returns_df,
            T=T, N=N, alpha=alpha,
            method=method,
            window_days=window_days,
            lam=lam,
            prev_weights=prev_weights,
            turnover_penalty=turnover_penalty,
        )
        return w_opt

    else:
        raise ValueError("mode must be 'static' or 'dynamic'")
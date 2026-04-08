"""
risk_decomposition.py
---------------------
Component (marginal) CVaR decomposition.

Why this matters
----------------
Total portfolio CVaR tells you *how much* you could lose in the tail.
Component CVaR tells you *which assets are causing it* — so you know
where reducing weight would actually lower risk.

Method
------
For a portfolio with weights w and P&L scenarios r_i (N x d matrix):

  1. Identify the tail scenarios: those where portfolio loss >= VaR_alpha
  2. For each asset j, its component CVaR contribution is:
         C_j = w_j * E[ r_j | portfolio loss >= VaR ]
     negated so that positive = contributing to loss.

  By construction:  sum_j C_j  ==  CVaR  (exact, no approximation).

Reference: Tasche (1999), "Risk contributions and performance measurement".
"""

import numpy as np
import pandas as pd
from src.simulate import simulate_portfolio_losses, simulate_t_dist_losses
from src.risk_metrics import compute_var


def component_cvar(
    mu,
    sigma,
    weights,
    tickers,
    T=1,
    N=100_000,
    alpha=0.95,
    dist="t-dist",
    df=5,
    seed=42,
):
    """
    Compute per-asset CVaR contributions for a portfolio.

    Parameters
    ----------
    mu, sigma   : mean vector and covariance matrix (aligned to tickers)
    weights     : portfolio weight vector
    tickers     : list of asset names (for labelling)
    T, N, alpha : horizon, MC paths, confidence level
    dist        : "normal" or "t-dist"
    df          : degrees of freedom (t-dist only)
    seed        : RNG seed

    Returns
    -------
    dict with keys:
        "total_cvar"          : float   — portfolio CVaR (matches compute_cvar)
        "component_cvar"      : ndarray — per-asset CVaR contribution (sums to total)
        "pct_contribution"    : ndarray — each asset's share of total CVaR (%)
        "marginal_cvar"       : ndarray — ∂CVaR/∂w_j (unscaled by weight)
        "dataframe"           : pd.DataFrame — tidy summary table
    """
    mu      = np.asarray(mu,      float)
    sigma   = np.asarray(sigma,   float)
    weights = np.asarray(weights, float)
    d       = len(mu)

    # --- Simulate full asset-level scenario matrix ---
    rng = np.random.default_rng(seed)
    mu_s    = mu    * T
    sigma_s = sigma * T

    if dist == "normal":
        R = rng.multivariate_normal(mu_s, sigma_s, size=N)          # (N, d)
    elif dist == "t-dist":
        Z   = rng.multivariate_normal(np.zeros(d), sigma_s, size=N)
        chi = rng.chisquare(df, size=N) / df
        R   = mu_s + Z / np.sqrt(chi)[:, None]
    else:
        raise ValueError("dist must be 'normal' or 't-dist'.")

    port_returns = R @ weights          # (N,)
    port_losses  = -port_returns        # sign convention: loss > 0 is bad

    var = compute_var(port_losses, alpha)
    tail_mask = port_losses >= var      # bool (N,)

    if tail_mask.sum() == 0:
        raise RuntimeError("No tail scenarios found — try increasing N or lowering alpha.")

    # Asset returns in tail scenarios, shape (|tail|, d)
    R_tail = R[tail_mask]               # returns (not losses) in tail

    # Marginal CVaR: E[-r_j | tail] for each asset j
    marginal = -R_tail.mean(axis=0)     # (d,)

    # Component CVaR: w_j * marginal_j  — sums exactly to total CVaR
    component = weights * marginal      # (d,)
    total     = float(component.sum())  # should match compute_cvar(port_losses, alpha)

    pct = (component / total * 100.0) if total != 0 else np.zeros(d)

    df_out = pd.DataFrame({
        "ticker":           tickers,
        "weight":           weights,
        "marginal_cvar":    marginal,
        "component_cvar":   component,
        "pct_contribution": pct,
    }).sort_values("pct_contribution", ascending=False).reset_index(drop=True)

    return {
        "total_cvar":       total,
        "component_cvar":   component,
        "pct_contribution": pct,
        "marginal_cvar":    marginal,
        "dataframe":        df_out,
    }


def print_risk_decomposition(result, total_value=1.0, label="Portfolio"):
    """Pretty-print a component_cvar result dict."""
    df   = result["dataframe"]
    cvar = result["total_cvar"]

    print(f"\n{'='*58}")
    print(f"  CVaR Risk Decomposition — {label}")
    print(f"  Total CVaR: {cvar:.4%}   (${cvar * total_value:,.2f})")
    print(f"{'='*58}")
    print(f"  {'Ticker':<8} {'Weight':>7} {'Comp.CVaR':>11} {'$ At Risk':>11} {'Share':>7}")
    print(f"  {'-'*54}")
    for _, row in df.iterrows():
        print(
            f"  {row['ticker']:<8}"
            f"  {row['weight']:>6.2%}"
            f"  {row['component_cvar']:>10.4%}"
            f"  ${row['component_cvar'] * total_value:>9,.2f}"
            f"  {row['pct_contribution']:>6.1f}%"
        )
    print(f"{'='*58}\n")
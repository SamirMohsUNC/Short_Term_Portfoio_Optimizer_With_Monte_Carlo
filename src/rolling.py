import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.optimizer import minimize_cvar
from src.simulate import simulate_portfolio_losses, simulate_t_dist_losses
from src.risk_metrics import compute_var, compute_cvar


def realized_horizon_loss(returns_block, weights):
    """Sum daily log returns over the horizon for each asset; loss = -(weighted sum)."""
    asset_logret = returns_block.sum(axis=0).values
    return float(-(np.dot(asset_logret, weights)))


def _performance_metrics(log_returns_arr, horizon=1, trading_days=252):
    """
    Compute annualised Sharpe ratio, max drawdown, and Calmar ratio from a
    series of per-step log returns.

    Parameters
    ----------
    log_returns_arr : ndarray  — per-step log returns (one entry per rebalance)
    horizon         : int      — horizon days per step (used for annualisation)
    """
    r = np.asarray(log_returns_arr, float)
    if r.size == 0:
        return {"sharpe": np.nan, "max_drawdown": np.nan, "calmar": np.nan}

    steps_per_year = trading_days / horizon

    mean_r = r.mean()
    std_r  = r.std(ddof=1) if r.size > 1 else np.nan

    sharpe = (mean_r / std_r * np.sqrt(steps_per_year)) if std_r and std_r > 0 else np.nan

    # Max drawdown from cumulative log-return path
    cum = np.exp(np.cumsum(r))
    running_max = np.maximum.accumulate(cum)
    drawdowns   = (cum - running_max) / running_max
    max_dd      = float(drawdowns.min())               # most negative

    ann_return = float(np.exp(mean_r * steps_per_year) - 1.0)
    calmar     = (ann_return / abs(max_dd)) if max_dd != 0 else np.nan

    return {
        "sharpe":       float(sharpe),
        "max_drawdown": max_dd,
        "calmar":       float(calmar),
        "ann_return":   ann_return,
    }


def rolling_optimize_and_dashboard(
    returns_df,
    baseline_weights,
    alpha=0.95,
    window=252,
    horizon=1,
    N=50_000,
    dist="t-dist",      # "normal" | "t-dist"
    df=5,
    save_dir="reports",
    seed=123,
):
    """
    Rolling CVaR-minimising optimisation with risk evaluation vs a baseline.

    returns_df       : DataFrame of DAILY log returns (index=dates, cols=tickers)
    baseline_weights : np.array aligned to returns_df.columns

    Bugs fixed vs original:
    - dist='t-dist' branch now correctly calls simulate_t_dist_losses.
    - Baseline realized loss now uses `future` (not `train`).

    New:
    - Sharpe, max drawdown, Calmar ratio added to summary for both strategies.
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    dates               = []
    w_rolling           = []
    var_fore            = []
    cvar_fore           = []
    rlz_losses_rolling  = []
    rlz_losses_baseline = []
    turnover            = []
    roll_log_returns    = []
    base_log_returns    = []

    baseline_weights = np.asarray(baseline_weights, dtype=float)
    baseline_weights = baseline_weights / baseline_weights.sum()
    prev_w = None

    if len(returns_df) < window + horizon:
        raise ValueError("Not enough data: increase history or reduce window/horizon.")

    for t in range(window, len(returns_df) - horizon + 1):
        train  = returns_df.iloc[t - window: t]
        future = returns_df.iloc[t: t + horizon]

        mu    = train.mean().values
        sigma = train.cov().values

        # Optimise weights for this step
        w_opt = minimize_cvar(mu, sigma, T=horizon, N=N)

        # Forecast VaR/CVaR with the correct distribution
        local_seed = int(rng.integers(0, 1_000_000_000))
        if dist == "normal":
            losses = simulate_portfolio_losses(mu, sigma, w_opt, T=horizon, N=N, seed=local_seed)
        else:
            # BUG FIX: was calling simulate_portfolio_losses (Gaussian) for t-dist branch
            losses = simulate_t_dist_losses(mu, sigma, w_opt, T=horizon, N=N, df=df, seed=local_seed)

        v  = compute_var(losses, alpha)
        es = compute_cvar(losses, alpha)

        # Realized losses over the NEXT horizon
        rlz_roll = realized_horizon_loss(future, w_opt)
        # BUG FIX: was using `train` (historical window) instead of `future`
        rlz_base = realized_horizon_loss(future, baseline_weights)

        dates.append(returns_df.index[t + horizon - 1])
        w_rolling.append(w_opt)
        var_fore.append(v)
        cvar_fore.append(es)
        rlz_losses_rolling.append(rlz_roll)
        rlz_losses_baseline.append(rlz_base)
        roll_log_returns.append(-rlz_roll)
        base_log_returns.append(-rlz_base)

        # Turnover from previous step
        turnover.append(0.0 if prev_w is None else float(np.abs(w_opt - prev_w).sum()))
        prev_w = w_opt

    # Arrays
    dates     = pd.to_datetime(pd.Index(dates))
    var_fore  = np.array(var_fore)
    cvar_fore = np.array(cvar_fore)
    rlz_roll  = np.array(rlz_losses_rolling)
    rlz_base  = np.array(rlz_losses_baseline)
    exceed    = (rlz_roll >= var_fore).astype(int)

    roll_lr = np.array(roll_log_returns)
    base_lr = np.array(base_log_returns)
    cum_roll = np.exp(np.cumsum(roll_lr))
    cum_base = np.exp(np.cumsum(base_lr))

    # Performance metrics
    roll_perf = _performance_metrics(roll_lr, horizon=horizon)
    base_perf = _performance_metrics(base_lr, horizon=horizon)

    summary = {
        "alpha":    alpha,
        "window":   window,
        "horizon":  horizon,
        "N":        N,
        "dist":     dist,
        "df":       df,
        # Risk
        "var_hit_rate":     float(exceed.mean()),
        "expected_hit_rate": float(1 - alpha),
        "avg_VaR":          float(np.mean(var_fore)),
        "avg_CVaR":         float(np.mean(cvar_fore)),
        "avg_turnover_L1":  float(np.mean(turnover)),
        # Growth
        "rolling_final_growth":              float(cum_roll[-1]),
        "baseline_final_growth":             float(cum_base[-1]),
        "rolling_vs_baseline_outperformance": float(cum_roll[-1] / cum_base[-1] - 1.0),
        # Rolling strategy metrics
        "rolling_sharpe":       roll_perf["sharpe"],
        "rolling_max_drawdown": roll_perf["max_drawdown"],
        "rolling_calmar":       roll_perf["calmar"],
        "rolling_ann_return":   roll_perf["ann_return"],
        # Baseline metrics
        "baseline_sharpe":       base_perf["sharpe"],
        "baseline_max_drawdown": base_perf["max_drawdown"],
        "baseline_calmar":       base_perf["calmar"],
        "baseline_ann_return":   base_perf["ann_return"],
    }

    # Dashboard
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_dir, f"rolling_dashboard_{ts}.png")

    fig, axes = plt.subplots(4, 1, figsize=(11, 16))

    # 1) Cumulative growth
    axes[0].plot(dates, cum_roll, label="Rolling-Optimised")
    axes[0].plot(dates, cum_base, label="Baseline (Static)", linestyle="--")
    axes[0].set_title("Cumulative Growth (log-return compounding)")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend()

    # 2) VaR forecast vs realized loss
    axes[1].plot(dates, var_fore, label=f"Forecast VaR (α={alpha})", color="steelblue")
    axes[1].plot(dates, rlz_roll, label="Realized Loss", color="tomato", alpha=0.7)
    exceedance_dates = dates[exceed.astype(bool)]
    exceedance_vals  = rlz_roll[exceed.astype(bool)]
    axes[1].scatter(exceedance_dates, exceedance_vals, color="red", s=15, zorder=5, label="Exceedance")
    axes[1].set_title(f"Forecast VaR vs Realized Loss  |  Hit rate: {exceed.mean():.3f} (expected {1-alpha:.3f})")
    axes[1].set_ylabel("Loss (log-return)")
    axes[1].legend()

    # 3) Drawdown
    dd_roll = (cum_roll - np.maximum.accumulate(cum_roll)) / np.maximum.accumulate(cum_roll)
    dd_base = (cum_base - np.maximum.accumulate(cum_base)) / np.maximum.accumulate(cum_base)
    axes[2].fill_between(dates, dd_roll, 0, alpha=0.4, label="Rolling Drawdown", color="steelblue")
    axes[2].fill_between(dates, dd_base, 0, alpha=0.4, label="Baseline Drawdown", color="orange")
    axes[2].set_title("Drawdown")
    axes[2].set_ylabel("Drawdown (%)")
    axes[2].legend()

    # 4) Turnover
    axes[3].plot(dates, turnover, label="Turnover (L1)", color="purple")
    axes[3].set_title("Portfolio Turnover (L1 distance step-to-step)")
    axes[3].set_ylabel("L1 distance")
    axes[3].legend()

    for ax in axes:
        ax.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return summary, out_path
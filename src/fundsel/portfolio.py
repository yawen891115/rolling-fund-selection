"""Portfolio construction: which funds (top-K with a holding buffer) and how much (risk parity).

Risk parity (equal risk contribution, ERC)
------------------------------------------
Portfolio volatility sigma_p = sqrt(w' S w). Fund i contributes RC_i = w_i (S w)_i / sigma_p, and the RC_i
sum to sigma_p. Risk parity asks for equal RC_i.

Solver trick (Spinu 2013): solve the strictly convex problem
        min_y  0.5 y' S y - (1/n) sum_i log(y_i),   y > 0
and normalise w = y / sum(y). The first-order condition (S y)_i = 1 / (n y_i) gives y_i (S y)_i = 1/n
for every i, which is exactly "equal risk contributions".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


def select_funds(scores: pd.Series, holdings: list[str], top_k: int, buffer_k: int) -> list[str]:
    """Take the top_k funds by score, but keep a current holding as long as it still ranks inside
    the top buffer_k (avoids churning funds that hover around rank K and paying redemption fees)."""
    ranked = scores.sort_values(ascending=False).index.tolist()
    keep = [c for c in holdings if c in set(ranked[:buffer_k])]
    new = [c for c in ranked if c not in keep][: max(top_k - len(keep), 0)]
    return keep + new


def estimate_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance (annualised): sample covariance shrunk towards a scaled identity."""
    X = returns.fillna(0.0).to_numpy()
    return LedoitWolf().fit(X).covariance_ * 252


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    y0 = 1.0 / np.sqrt(np.diag(cov))                  # start from inverse volatility

    def f(y):
        return 0.5 * y @ cov @ y - np.log(y).sum() / n

    def grad(y):
        return cov @ y - 1.0 / (n * y)

    sol = minimize(f, y0, jac=grad, method="L-BFGS-B", bounds=[(1e-10, None)] * n)
    return sol.x / sol.x.sum()


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Share of portfolio risk contributed by each fund (sums to 1)."""
    rc = w * (cov @ w)
    return rc / rc.sum()


def target_weights(scores: pd.Series, holdings: list[str], hist_returns: pd.DataFrame,
                   top_k: int, buffer_k: int, scheme: str = "rp") -> pd.Series:
    chosen = select_funds(scores, holdings, top_k, buffer_k)
    if scheme == "ew":
        return pd.Series(1.0 / len(chosen), index=chosen)
    cov = estimate_cov(hist_returns[chosen])
    return pd.Series(risk_parity_weights(cov), index=chosen)

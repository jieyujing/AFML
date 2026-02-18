"""
Performance Metrics Module (Polars Optimized).

This module implements statistical performance metrics for strategy evaluation,
specifically focusing on the Deflated Sharpe Ratio (DSR) as described in
AFML Chapter 14.
"""

import numpy as np
import polars as pl
from typing import Union
from scipy.stats import norm


def estimated_sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """
    Calculate the estimated Sharpe Ratio (non-annualized).
    """
    if len(returns) < 2:
        return 0.0
    return np.mean(returns - risk_free) / np.std(returns, ddof=1)


def annualized_sharpe_ratio(
    returns: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252
) -> float:
    """
    Calculate the annualized Sharpe Ratio.
    """
    sr = estimated_sharpe_ratio(returns, risk_free / periods_per_year)
    return sr * np.sqrt(periods_per_year)


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    skewness: float,
    kurtosis: float,
    n_obs: int,
) -> float:
    """
    Calculate the Probabilistic Sharpe Ratio (PSR).
    """
    if n_obs <= 1:
        return 0.5

    numerator = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1)
    denominator = np.sqrt(
        1 - skewness * observed_sr + (kurtosis - 1) / 4 * observed_sr**2
    )

    if np.isnan(denominator) or denominator == 0:
        return 0.0

    test_statistic = numerator / denominator
    return norm.cdf(test_statistic)


def deflated_sharpe_ratio(
    observed_sr: float,
    skewness: float,
    kurtosis: float,
    n_obs: int,
    n_trials: int,
    variance_trials: float = 1.0,
) -> float:
    """
    Calculate the Deflated Sharpe Ratio (DSR).
    """
    if n_trials < 1:
        n_trials = 1

    sr_benchmark = np.sqrt(2 * np.log(n_trials))

    return probabilistic_sharpe_ratio(
        observed_sr, sr_benchmark, skewness, kurtosis, n_obs
    )


def get_strategy_metrics(
    returns: Union[pl.Series, np.ndarray], n_trials: int = 1
) -> dict:
    """
    Compute comprehensive strategy metrics including DSR.
    """
    if isinstance(returns, pl.Series):
        x = returns.to_numpy()
    else:
        x = returns

    x = x[~np.isnan(x)]

    if len(x) < 2:
        return {}

    mean_ret = np.mean(x)
    std_ret = np.std(x, ddof=1)

    try:
        from scipy.stats import skew, kurtosis

        sk = skew(x)
        kt = kurtosis(x, fisher=False)
    except ImportError:
        sk = 0
        kt = 3

    sr = mean_ret / std_ret if std_ret > 0 else 0
    ann_sr = sr * np.sqrt(252)

    psr = probabilistic_sharpe_ratio(sr, 0.0, sk, kt, len(x))
    dsr = deflated_sharpe_ratio(sr, sk, kt, len(x), n_trials)

    return {
        "mean_return": mean_ret,
        "std_return": std_ret,
        "skewness": sk,
        "kurtosis": kt,
        "sharpe_ratio": sr,
        "annualized_sharpe_ratio": ann_sr,
        "psr": psr,
        "dsr": dsr,
        "n_obs": len(x),
    }

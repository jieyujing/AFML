"""CUSUM filter calibrator for target event rates."""

import numpy as np
import pandas as pd
from afmlkit.sampling import cusum_filter


def calibrate_cusum_rates(
    bars: pd.DataFrame,
    target_rates: list[float],
    k_min: float = 0.001,
    k_max: float = 10.0,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> pd.DataFrame:
    """
    Binary-search k value for each target CUSUM event rate.

    :param bars: Dollar bars DataFrame with 'close' column.
    :param target_rates: List of target event rates, e.g. [0.05, 0.10, 0.15].
    :param k_min: Lower bound of k search.
    :param k_max: Upper bound of k search.
    :param tol: Convergence tolerance on |actual_rate - target_rate|.
    :param max_iter: Max binary search iterations.
    :returns: DataFrame with columns [rate, k, actual_rate, n_events].
    """
    # Compute log returns from close prices
    close = bars['close'].values
    log_returns = np.diff(np.log(close))

    n = len(log_returns)
    results = []

    for target_rate in target_rates:
        # Binary search for k
        k_low, k_high = k_min, k_max
        k_guess = (k_low + k_high) / 2.0

        for _ in range(max_iter):
            # Compute actual event rate for current k
            event_indices = cusum_filter(log_returns, np.array([k_guess]))
            n_events = len(event_indices)
            actual_rate = n_events / n

            # Check convergence
            if abs(actual_rate - target_rate) < tol:
                break

            # Update binary search bounds
            if actual_rate > target_rate:
                # Too many events, need higher threshold (higher k)
                k_low = k_guess
            else:
                # Too few events, need lower threshold (lower k)
                k_high = k_guess

            k_guess = (k_low + k_high) / 2.0

        # Final computation with converged k
        event_indices = cusum_filter(log_returns, np.array([k_guess]))
        n_events = len(event_indices)
        actual_rate = n_events / n

        results.append({
            'rate': target_rate,
            'k': k_guess,
            'actual_rate': actual_rate,
            'n_events': n_events,
        })

    return pd.DataFrame(results)
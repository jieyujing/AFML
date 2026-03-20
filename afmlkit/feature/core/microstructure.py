"""
Microstructure Features for financial time series.

Implements liquidity and price impact measures from market microstructure theory.
These features capture the hidden costs of trading and market quality.

For Bar-level data (OHLCV):
- Amihud's Lambda: Illiquidity measure
- Roll Model: Implied bid-ask spread from price serial correlation
- Corwin-Schultz: Spread estimator from high-low prices

For Tick-level data (requires individual trades):
- Kyle's Lambda: Price impact of order flow
- PIN: Probability of Informed Trading
- Tick Rule: Infer trade direction from price changes

References
----------
    Amihud, Y. (2002). Illiquidity and stock returns.
    Roll, R. (1984). A simple implicit measure of the effective bid-ask spread.
    Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads.
    Kyle, A. S. (1985). Continuous auctions and insider trading.
"""
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from typing import Optional, Tuple
import pandas as pd

from afmlkit.feature.base import SISOTransform, MISOTransform


# ---------------------------------------------------------------------------
# Numba Backend - Core Functions
# ---------------------------------------------------------------------------

@njit(nogil=True, parallel=True)
def amihud_illiquidity(
    returns: NDArray[np.float64],
    dollar_volume: NDArray[np.float64],
    window: int
) -> NDArray[np.float64]:
    """
    Compute Amihud's illiquidity measure (Amihud's Lambda).

    ILLIQ_t = (1/D) × Σ |r_d| / DV_d

    where:
    - r_d = daily return
    - DV_d = dollar volume
    - D = number of days in window

    :param returns: Array of returns (can be log returns or simple returns)
    :param dollar_volume: Array of dollar volume (price × volume)
    :param window: Rolling window size
    :returns: Amihud illiquidity series

    Higher values indicate higher illiquidity (more price impact per dollar traded).

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> returns = np.random.randn(100) * 0.02
    >>> volume = np.abs(np.random.randn(100)) * 1e6
    >>> illiq = amihud_illiquidity(returns, volume, window=20)
    >>> len(illiq)
    100

    References
    ----------
        Amihud, Y. (2002). Illiquidity and stock returns: cross-section and
        time-series effects. Journal of Financial Markets, 5(1), 31-56.
    """
    n = len(returns)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < window:
        return result

    # Use absolute returns
    abs_returns = np.abs(returns)

    for t in prange(window - 1, n):
        sum_ratio = 0.0
        count = 0

        for i in range(t - window + 1, t + 1):
            if dollar_volume[i] > 1e-12:  # Avoid division by zero
                sum_ratio += abs_returns[i] / dollar_volume[i]
                count += 1

        if count > 0:
            result[t] = sum_ratio / count

    return result


@njit(nogil=True)
def roll_spread(
    prices: NDArray[np.float64],
    window: int
) -> NDArray[np.float64]:
    """
    Compute Roll's implicit bid-ask spread estimator.

    The Roll model assumes the observed price alternates between bid and ask.
    The spread is estimated from negative serial correlation in price changes.

    S = 2 × sqrt(-Cov(ΔP_t, ΔP_{t-1}))

    :param prices: Array of prices (typically close prices)
    :param window: Rolling window for covariance estimation
    :returns: Roll spread estimate series (as a fraction of price)

    Notes
    -----
    - Returns NaN when covariance is positive (which would give imaginary spread)
    - The spread is expressed as a decimal (e.g., 0.001 = 10 bps)

    References
    ----------
        Roll, R. (1984). A simple implicit measure of the effective bid-ask spread
        in an efficient market. Journal of Finance, 39(4), 1127-1139.
    """
    n = len(prices)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < window + 1:
        return result

    # Compute price changes
    delta_p = np.diff(prices)

    for t in range(window, n):
        # Get price changes in window
        window_deltas = delta_p[t - window: t]

        # Compute covariance of ΔP_t and ΔP_{t-1}
        # Cov(X_t, X_{t-1}) = E[(X_t - μ)(X_{t-1} - μ)]
        mean_delta = 0.0
        for i in range(window):
            mean_delta += window_deltas[i]
        mean_delta /= window

        cov = 0.0
        for i in range(1, window):
            cov += (window_deltas[i] - mean_delta) * (window_deltas[i - 1] - mean_delta)
        cov /= (window - 1)

        # Spread = 2 * sqrt(-Cov) if Cov < 0
        if cov < 0:
            spread = 2.0 * np.sqrt(-cov)
            # Normalize by price
            result[t] = spread / prices[t]
        else:
            result[t] = np.nan

    return result


@njit(nogil=True, parallel=True)
def corwin_schultz_spread(
    high: NDArray[np.float64],
    low: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute Corwin-Schultz bid-ask spread estimator.

    Uses the relationship between high and low prices to estimate the spread.
    The key insight is that spread causes high prices to be higher and low
    prices to be lower than the true value.

    S = (2 × (e^α - 1)) / (1 + e^α)

    where:
    α = (√(2 × β) - √β) / (3 - 2 × √2) - √(γ / (3 - 2 × √2))
    β = Σ (ln(H_t/L_t))² over 2 days
    γ = (ln(H_{t,t+1}/L_{t,t+1}))² over 2-day high-low

    :param high: Array of high prices
    :param low: Array of low prices
    :returns: Corwin-Schultz spread estimate (as a fraction)

    Notes
    -----
    - Uses 2-day estimation window
    - Returns NaN for negative spread estimates (can happen with noisy data)
    - Spread is expressed as decimal (e.g., 0.001 = 10 bps)

    References
    ----------
        Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask
        spreads from daily high and low prices. Journal of Finance, 67(2), 719-760.
    """
    n = len(high)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < 2:
        return result

    # Pre-compute log high/low ratios
    log_hl = np.empty(n, dtype=np.float64)
    for i in range(n):
        if high[i] > 0 and low[i] > 0 and high[i] > low[i]:
            log_hl[i] = np.log(high[i] / low[i])
        else:
            log_hl[i] = 0.0

    # Constants
    sqrt2 = np.sqrt(2.0)
    k = 1.0 / (3.0 - 2.0 * sqrt2)  # ≈ 2.82

    for t in prange(1, n):
        # Beta: sum of squared log(H/L) over 2 days
        beta = log_hl[t] ** 2 + log_hl[t - 1] ** 2

        # Gamma: squared log of 2-day high / 2-day low
        two_day_high = max(high[t], high[t - 1])
        two_day_low = min(low[t], low[t - 1])

        if two_day_high <= two_day_low or two_day_low <= 0:
            result[t] = np.nan
            continue

        gamma = np.log(two_day_high / two_day_low) ** 2

        # Alpha
        term1 = sqrt2 * np.sqrt(beta) if beta > 0 else 0.0
        term2 = np.sqrt(gamma) if gamma > 0 else 0.0

        alpha = (term1 - np.sqrt(beta)) * k - term2 * np.sqrt(k)

        # Spread
        exp_alpha = np.exp(alpha)
        spread = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)

        # Only keep positive spreads
        if spread > 0:
            result[t] = spread
        else:
            result[t] = np.nan

    return result


@njit(nogil=True, parallel=True)
def rolling_corwin_schultz_spread(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    window: int
) -> NDArray[np.float64]:
    """
    Rolling average of Corwin-Schultz spread estimates.

    :param high: Array of high prices
    :param low: Array of low prices
    :param window: Rolling window for averaging
    :returns: Rolling average spread series
    """
    # First compute daily spreads
    daily_spread = corwin_schultz_spread(high, low)

    n = len(daily_spread)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < window:
        return result

    for t in prange(window - 1, n):
        sum_spread = 0.0
        count = 0
        for i in range(t - window + 1, t + 1):
            if not np.isnan(daily_spread[i]) and daily_spread[i] > 0:
                sum_spread += daily_spread[i]
                count += 1
        if count > 0:
            result[t] = sum_spread / count

    return result


@njit(nogil=True)
def high_low_volatility(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    window: int
) -> NDArray[np.float64]:
    """
    Parkinson high-low volatility estimator.

    σ_p = √(1/(4×n×ln(2)) × Σ ln(H_i/L_i)²)

    :param high: Array of high prices
    :param low: Array of low prices
    :param window: Rolling window
    :returns: Annualized volatility (assuming daily data)

    References
    ----------
        Parkinson, M. (1980). The extreme value method for estimating the
        variance of the rate of return. Journal of Business, 53(1), 61-65.
    """
    n = len(high)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < window:
        return result

    # Constant: 1 / (4 * ln(2))
    k = 1.0 / (4.0 * np.log(2.0))

    for t in range(window - 1, n):
        sum_sq = 0.0
        count = 0
        for i in range(t - window + 1, t + 1):
            if high[i] > low[i] and low[i] > 0:
                log_hl = np.log(high[i] / low[i])
                sum_sq += log_hl ** 2
                count += 1

        if count > 0:
            result[t] = np.sqrt(k * sum_sq / count)

    return result


@njit(nogil=True, parallel=True)
def abelson_spread(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    window: int
) -> NDArray[np.float64]:
    """
    Abelson spread estimator (simplified version).

    Combines Roll model with volume information.

    :param close: Close prices
    :param volume: Trading volume
    :param window: Rolling window
    :returns: Spread estimate
    """
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < window + 1:
        return result

    # Compute price changes
    delta_p = np.diff(close)

    for t in prange(window, n):
        window_deltas = delta_p[t - window: t]

        # Variance of price changes
        mean_delta = 0.0
        for i in range(window):
            mean_delta += window_deltas[i]
        mean_delta /= window

        var = 0.0
        for i in range(window):
            var += (window_deltas[i] - mean_delta) ** 2
        var /= window

        # Covariance
        cov = 0.0
        for i in range(1, window):
            cov += (window_deltas[i] - mean_delta) * (window_deltas[i - 1] - mean_delta)
        cov /= (window - 1)

        # Adjusted spread
        if var > 0 and cov < 0:
            # Roll spread scaled by volume
            roll_spread_val = 2.0 * np.sqrt(-cov)
            vol_factor = np.sqrt(np.mean(volume[t - window: t]) / volume[t]) if volume[t] > 0 else 1.0
            result[t] = (roll_spread_val / close[t]) * vol_factor

    return result


# ---------------------------------------------------------------------------
# SISO/MISO Transforms
# ---------------------------------------------------------------------------

class AmihudTransform(MISOTransform):
    """
    MISO Transform for Amihud's illiquidity measure.

    Computes the rolling Amihud illiquidity ratio, which measures
    the price impact per unit of dollar volume.

    Parameters
    ----------
    return_col : str
        Column name for returns
    volume_col : str
        Column name for volume (will be multiplied by price)
    price_col : str
        Column name for price (to compute dollar volume)
    window : int
        Rolling window size

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from afmlkit.feature.core.microstructure import AmihudTransform
    >>>
    >>> dates = pd.date_range('2023-01-01', periods=200, freq='D')
    >>> df = pd.DataFrame({
    ...     'close': 100 + np.random.randn(200).cumsum(),
    ...     'volume': np.abs(np.random.randn(200)) * 1e6
    ... }, index=dates)
    >>> df['returns'] = df['close'].pct_change()
    >>>
    >>> transform = AmihudTransform('returns', 'volume', 'close', window=20)
    >>> illiq = transform(df, backend='nb')
    """

    def __init__(
        self,
        return_col: str,
        volume_col: str,
        price_col: str,
        window: int
    ):
        self.window = window
        output_col = f'amihud_{window}'

        super().__init__([return_col, volume_col, price_col], output_col)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        missing = [col for col in self.requires if col not in x.columns]
        if missing:
            raise ValueError(f"Input columns {missing} not found in DataFrame")
        return True

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        returns = x[self.requires[0]].values.astype(np.float64)
        volume = x[self.requires[1]].values.astype(np.float64)
        price = x[self.requires[2]].values.astype(np.float64)

        dollar_volume = price * volume
        result = amihud_illiquidity(returns, dollar_volume, self.window)

        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)


class RollSpreadTransform(SISOTransform):
    """
    SISO Transform for Roll's implicit bid-ask spread.

    Estimates the effective bid-ask spread from negative serial
    correlation in price changes.

    Parameters
    ----------
    input_col : str
        Column name for prices (typically 'close')
    window : int
        Rolling window for covariance estimation

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from afmlkit.feature.core.microstructure import RollSpreadTransform
    >>>
    >>> dates = pd.date_range('2023-01-01', periods=200, freq='D')
    >>> prices = 100 + np.random.randn(200).cumsum()
    >>> df = pd.DataFrame({'close': prices}, index=dates)
    >>>
    >>> transform = RollSpreadTransform('close', window=20)
    >>> spread = transform(df, backend='nb')
    """

    def __init__(
        self,
        input_col: str,
        window: int
    ):
        self.window = window
        output_col = f'roll_spread_{window}'

        super().__init__(input_col, output_col)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if self.requires[0] not in x.columns:
            raise ValueError(f"Input column {self.requires[0]} not found in DataFrame")
        return True

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        prices = x[self.requires[0]].values.astype(np.float64)
        result = roll_spread(prices, self.window)
        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)


class CorwinSchultzTransform(MISOTransform):
    """
    MISO Transform for Corwin-Schultz spread estimator.

    Uses high and low prices to estimate the bid-ask spread.

    Parameters
    ----------
    high_col : str
        Column name for high prices
    low_col : str
        Column name for low prices
    window : int, optional
        Rolling window for averaging (default 1 = no averaging)

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from afmlkit.feature.core.microstructure import CorwinSchultzTransform
    >>>
    >>> dates = pd.date_range('2023-01-01', periods=200, freq='D')
    >>> high = 100 + np.abs(np.random.randn(200))
    >>> low = 100 - np.abs(np.random.randn(200))
    >>> df = pd.DataFrame({'high': high, 'low': low}, index=dates)
    >>>
    >>> transform = CorwinSchultzTransform('high', 'low', window=5)
    >>> spread = transform(df, backend='nb')
    """

    def __init__(
        self,
        high_col: str,
        low_col: str,
        window: int = 1
    ):
        self.window = window
        output_col = f'cs_spread_{window}' if window > 1 else 'cs_spread'

        super().__init__([high_col, low_col], output_col)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        missing = [col for col in self.requires if col not in x.columns]
        if missing:
            raise ValueError(f"Input columns {missing} not found in DataFrame")
        return True

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        high = x[self.requires[0]].values.astype(np.float64)
        low = x[self.requires[1]].values.astype(np.float64)

        if self.window > 1:
            result = rolling_corwin_schultz_spread(high, low, self.window)
        else:
            result = corwin_schultz_spread(high, low)

        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)


class ParkinsonVolatilityTransform(MISOTransform):
    """
    MISO Transform for Parkinson high-low volatility.

    Uses high and low prices to estimate volatility, which is
    more efficient than close-to-close volatility when prices
    follow a continuous diffusion process.

    Parameters
    ----------
    high_col : str
        Column name for high prices
    low_col : str
        Column name for low prices
    window : int
        Rolling window size

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from afmlkit.feature.core.microstructure import ParkinsonVolatilityTransform
    >>>
    >>> dates = pd.date_range('2023-01-01', periods=200, freq='D')
    >>> high = 100 + np.abs(np.random.randn(200))
    >>> low = 100 - np.abs(np.random.randn(200))
    >>> df = pd.DataFrame({'high': high, 'low': low}, index=dates)
    >>>
    >>> transform = ParkinsonVolatilityTransform('high', 'low', window=20)
    >>> vol = transform(df, backend='nb')
    """

    def __init__(
        self,
        high_col: str,
        low_col: str,
        window: int
    ):
        self.window = window
        output_col = f'parkinson_vol_{window}'

        super().__init__([high_col, low_col], output_col)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        missing = [col for col in self.requires if col not in x.columns]
        if missing:
            raise ValueError(f"Input columns {missing} not found in DataFrame")
        return True

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        high = x[self.requires[0]].values.astype(np.float64)
        low = x[self.requires[1]].values.astype(np.float64)

        result = high_low_volatility(high, low, self.window)

        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def amihud(
    df: pd.DataFrame,
    return_col: str = 'returns',
    volume_col: str = 'volume',
    price_col: str = 'close',
    window: int = 20
) -> pd.Series:
    """
    Convenience function to compute Amihud illiquidity.

    :param df: DataFrame with returns, volume, and price
    :param return_col: Column name for returns
    :param volume_col: Column name for volume
    :param price_col: Column name for price
    :param window: Rolling window
    :returns: Amihud illiquidity Series
    """
    transform = AmihudTransform(return_col, volume_col, price_col, window)
    return transform(df, backend='nb')


def roll_spread_estimate(
    prices: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Convenience function to compute Roll spread.

    :param prices: Price Series
    :param window: Rolling window
    :returns: Roll spread Series
    """
    transform = RollSpreadTransform(prices.name or 'price', window)
    df = prices.to_frame(name=prices.name or 'price')
    return transform(df, backend='nb')


def corwin_schultz(
    df: pd.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    window: int = 1
) -> pd.Series:
    """
    Convenience function to compute Corwin-Schultz spread.

    :param df: DataFrame with high and low prices
    :param high_col: Column name for high prices
    :param low_col: Column name for low prices
    :param window: Rolling window for averaging
    :returns: Corwin-Schultz spread Series
    """
    transform = CorwinSchultzTransform(high_col, low_col, window)
    return transform(df, backend='nb')
"""
Entropy Features for financial time series.

Implements information-theoretic measures of market efficiency and predictability.
Low entropy indicates repetitive patterns (predictable), while high entropy
indicates random white noise (unpredictable).

Features:
- Shannon's Entropy Rate: Theoretical limit of information content
- Lempel-Ziv (LZ) Entropy: Compression-based entropy estimate
- Kontoyiannis' LZ Entropy: Improved LZ algorithm
- Entropy-Implied Volatility: Volatility derived from entropy

References
----------
    López de Prado, M. (2018). *Advances in Financial Machine Learning*.
    Chapter 18 — Entropy Features.

    Shannon, C. E. (1948). A mathematical theory of communication.
    Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences.
    Kontoyiannis, I. (1998). Asymptotically optimal lossy Lempel-Ziv coding.
"""
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from typing import Tuple, Optional
import pandas as pd

from afmlkit.feature.base import SISOTransform


# ---------------------------------------------------------------------------
# Numba Backend - Core Functions
# ---------------------------------------------------------------------------

@njit(nogil=True)
def discretize_returns(
    returns: NDArray[np.float64],
    n_bins: int = 3
) -> NDArray[np.int64]:
    """
    Discretize continuous returns into integer symbols.

    :param returns: Array of returns
    :param n_bins: Number of bins (default 3: down, flat, up)
    :returns: Array of integer symbols (0 to n_bins-1)

    For n_bins=3:
    - 0: negative returns (down)
    - 1: near-zero returns (flat)
    - 2: positive returns (up)
    """
    n = len(returns)
    symbols = np.empty(n, dtype=np.int64)

    if n_bins == 3:
        # Special case: ternary encoding based on sign
        # Define "flat" as returns within one standard deviation
        std = np.std(returns)
        threshold = std * 0.1  # 10% of std as "flat" threshold

        for i in range(n):
            if returns[i] < -threshold:
                symbols[i] = 0  # Down
            elif returns[i] > threshold:
                symbols[i] = 2  # Up
            else:
                symbols[i] = 1  # Flat
    else:
        # General quantile-based binning
        sorted_returns = np.sort(returns)
        bin_edges = np.empty(n_bins - 1, dtype=np.float64)

        for i in range(n_bins - 1):
            idx = int((i + 1) * n / n_bins)
            bin_edges[i] = sorted_returns[idx]

        for i in range(n):
            symbol = 0
            for j in range(n_bins - 1):
                if returns[i] > bin_edges[j]:
                    symbol = j + 1
                else:
                    break
            symbols[i] = symbol

    return symbols


@njit(nogil=True)
def compute_symbol_frequencies(
    symbols: NDArray[np.int64],
    n_symbols: int
) -> NDArray[np.float64]:
    """
    Compute frequency distribution of symbols.

    :param symbols: Array of integer symbols
    :param n_symbols: Number of possible symbols
    :returns: Frequency array (sums to 1)
    """
    n = len(symbols)
    counts = np.zeros(n_symbols, dtype=np.float64)

    for i in range(n):
        if 0 <= symbols[i] < n_symbols:
            counts[symbols[i]] += 1.0

    # Normalize to frequencies
    total = np.sum(counts)
    if total > 0:
        counts /= total

    return counts


@njit(nogil=True)
def shannon_entropy(
    frequencies: NDArray[np.float64]
) -> float:
    """
    Compute Shannon entropy from frequency distribution.

    H = -Σ p_i × log2(p_i)

    :param frequencies: Frequency distribution (must sum to 1)
    :returns: Shannon entropy in bits

    Interpretation:
    - H = 0: Only one symbol (completely predictable)
    - H = log2(n_symbols): Uniform distribution (maximally random)
    """
    entropy = 0.0

    for p in frequencies:
        if p > 1e-10:  # Avoid log(0)
            entropy -= p * np.log2(p)

    return entropy


@njit(nogil=True)
def shannon_entropy_rate(
    symbols: NDArray[np.int64],
    n_symbols: int,
    order: int = 1
) -> float:
    """
    Compute Shannon entropy rate for n-gram sequences.

    H_rate = H(X_{t+1} | X_t, X_{t-1}, ..., X_{t-order+1})

    :param symbols: Array of integer symbols
    :param n_symbols: Number of possible symbols
    :param order: Markov order (1 = pairwise, 2 = triplet, etc.)
    :returns: Conditional entropy in bits

    Higher order captures more complex dependencies.
    """
    n = len(symbols)

    if n <= order:
        return 0.0

    # Count n-grams and (n-1)-grams
    # For order k, we count (k+1)-grams and k-grams
    n_context = n_symbols ** order
    n_joint = n_context * n_symbols

    context_counts = np.zeros(n_context, dtype=np.float64)
    joint_counts = np.zeros(n_joint, dtype=np.float64)

    for i in range(order, n):
        # Compute context index
        context_idx = 0
        for j in range(order):
            context_idx = context_idx * n_symbols + symbols[i - order + j]

        # Joint index
        joint_idx = context_idx * n_symbols + symbols[i]

        context_counts[context_idx] += 1.0
        joint_counts[joint_idx] += 1.0

    # Compute conditional entropy H(X_{t+1} | context)
    entropy = 0.0

    for ctx in range(n_context):
        if context_counts[ctx] < 1:
            continue

        p_context = context_counts[ctx] / (n - order)

        # Conditional distribution
        for sym in range(n_symbols):
            joint_idx = ctx * n_symbols + sym
            if joint_counts[joint_idx] > 0:
                p_cond = joint_counts[joint_idx] / context_counts[ctx]
                entropy -= p_context * p_cond * np.log2(p_cond)

    return entropy


@njit(nogil=True)
def lempel_ziv_complexity(
    symbols: NDArray[np.int64],
    n_symbols: int
) -> int:
    """
    Compute Lempel-Ziv complexity (number of distinct substrings).

    The LZ complexity measures the number of new substrings encountered
    when parsing the sequence from left to right.

    :param symbols: Array of integer symbols
    :param n_symbols: Number of possible symbols
    :returns: Number of distinct LZ factors (c)

    This is the parsing complexity, not yet the entropy.
    """
    n = len(symbols)

    if n == 0:
        return 0

    c = 1  # Start with first symbol as first factor
    i = 0  # Current position

    while i < n - 1:
        # Find longest match in already parsed part
        max_match = 1

        for j in range(i + 1, n):
            # Try to find substring [i+1:j+1] in [0:i+1]
            found = False
            length = j - i

            for k in range(i + 1 - length + 1):
                match = True
                for m in range(length):
                    if k + m >= i + 1:
                        match = False
                        break
                    if symbols[k + m] != symbols[i + 1 + m]:
                        match = False
                        break

                if match:
                    found = True
                    break

            if found:
                max_match = length
            else:
                break

        i += max_match
        if i < n:
            c += 1

    return c


@njit(nogil=True)
def lempel_ziv_entropy(
    symbols: NDArray[np.int64],
    n_symbols: int
) -> float:
    """
    Compute Lempel-Ziv entropy estimate.

    H_LZ = (c × log2(n)) / n

    where c is the LZ complexity and n is the sequence length.

    :param symbols: Array of integer symbols
    :param n_symbols: Number of possible symbols
    :returns: LZ entropy estimate in bits

    References
    ----------
        Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences.
        IEEE Transactions on Information Theory, 22(1), 75-81.
    """
    n = len(symbols)

    if n <= 1:
        return 0.0

    c = lempel_ziv_complexity(symbols, n_symbols)

    # LZ entropy formula
    entropy = (c * np.log2(n)) / n

    return entropy


@njit(nogil=True)
def kontoyiannis_lz_entropy(
    symbols: NDArray[np.int64],
    n_symbols: int
) -> float:
    """
    Compute Kontoyiannis' LZ entropy estimate.

    This is an improved version that computes the shortest non-redundant
    substring length at each position, then averages.

    H_K = (1/n) × Σ log2(L_i)

    where L_i is the length of the shortest substring starting at i
    that hasn't appeared before.

    :param symbols: Array of integer symbols
    :param n_symbols: Number of possible symbols
    :returns: Kontoyiannis LZ entropy in bits

    References
    ----------
        Kontoyiannis, I. (1998). Asymptotically optimal lossy Lempel-Ziv coding.
        IEEE International Symposium on Information Theory.
    """
    n = len(symbols)

    if n <= 1:
        return 0.0

    log_sum = 0.0
    count = 0

    i = 0
    while i < n:
        # Find shortest substring starting at i that doesn't appear in [0:i]
        max_len = 1

        for length in range(1, n - i + 1):
            found = False

            # Check if symbols[i:i+length] appears in symbols[0:i]
            for k in range(i):
                if k + length > i:
                    break

                match = True
                for m in range(length):
                    if symbols[k + m] != symbols[i + m]:
                        match = False
                        break

                if match:
                    found = True
                    break

            if not found:
                max_len = length
                break

        # Add log(max_len) to sum
        if max_len > 0:
            log_sum += np.log2(max_len)
            count += 1

        i += max_len

    if count > 0:
        return log_sum / count
    else:
        return 0.0


@njit(nogil=True)
def entropy_implied_volatility(
    entropy: float,
    annualize: bool = True
) -> float:
    """
    Convert entropy to implied volatility.

    Under normal distribution assumption:
    σ = √(2πe) × 2^(-H)

    This is a theoretical mapping between entropy and volatility.

    :param entropy: Shannon entropy in bits
    :param annualize: If True, multiply by √252 for annualized vol
    :returns: Implied volatility

    References
    ----------
        For a Gaussian process, entropy H = 0.5 × log2(2πeσ²)
        Therefore σ = √(2πe) × 2^(-H)
    """
    # σ = sqrt(2πe) × 2^(-H)
    sigma = np.sqrt(2 * np.pi * np.e) * (2.0 ** (-entropy))

    if annualize:
        sigma *= np.sqrt(252)

    return sigma


@njit(nogil=True, parallel=True)
def rolling_entropy(
    returns: NDArray[np.float64],
    window: int,
    n_bins: int = 3,
    method: int = 0  # 0=shannon, 1=lz, 2=kontoyiannis
) -> NDArray[np.float64]:
    """
    Compute rolling entropy measure.

    :param returns: Array of returns
    :param window: Rolling window size
    :param n_bins: Number of bins for discretization
    :param method: Entropy method (0=Shannon, 1=LZ, 2=Kontoyiannis)
    :returns: Rolling entropy series
    """
    n = len(returns)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < window:
        return result

    for t in prange(window - 1, n):
        window_returns = returns[t - window + 1: t + 1]
        symbols = discretize_returns(window_returns, n_bins)

        if method == 0:
            freqs = compute_symbol_frequencies(symbols, n_bins)
            result[t] = shannon_entropy(freqs)
        elif method == 1:
            result[t] = lempel_ziv_entropy(symbols, n_bins)
        else:  # method == 2
            result[t] = kontoyiannis_lz_entropy(symbols, n_bins)

    return result


@njit(nogil=True, parallel=True)
def rolling_entropy_rate(
    returns: NDArray[np.float64],
    window: int,
    n_bins: int = 3,
    order: int = 1
) -> NDArray[np.float64]:
    """
    Compute rolling Shannon entropy rate with Markov order.

    :param returns: Array of returns
    :param window: Rolling window size
    :param n_bins: Number of bins for discretization
    :param order: Markov order (1 for pairwise, 2 for triplet, etc.)
    :returns: Rolling entropy rate series
    """
    n = len(returns)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < window:
        return result

    for t in prange(window - 1, n):
        window_returns = returns[t - window + 1: t + 1]
        symbols = discretize_returns(window_returns, n_bins)
        result[t] = shannon_entropy_rate(symbols, n_bins, order)

    return result


# ---------------------------------------------------------------------------
# SISO Transform
# ---------------------------------------------------------------------------

class ShannonEntropyTransform(SISOTransform):
    """
    SISO Transform for Shannon entropy.

    Computes rolling Shannon entropy of discretized returns.

    Parameters
    ----------
    input_col : str
        Column name for returns
    window : int
        Rolling window size
    n_bins : int
        Number of bins for discretization (default 3: up/flat/down)

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from afmlkit.feature.core.entropy import ShannonEntropyTransform
    >>>
    >>> dates = pd.date_range('2023-01-01', periods=200, freq='D')
    >>> returns = np.random.randn(200) * 0.02
    >>> df = pd.DataFrame({'returns': returns}, index=dates)
    >>>
    >>> transform = ShannonEntropyTransform('returns', window=50, n_bins=3)
    >>> entropy = transform(df, backend='nb')

    Interpretation:
    - Low entropy (< 1 bit): More predictable patterns
    - High entropy (> 1.5 bits): Near random behavior
    """

    def __init__(
        self,
        input_col: str,
        window: int,
        n_bins: int = 3
    ):
        self.window = window
        self.n_bins = n_bins
        output_col = f'shannon_entropy_{window}'

        super().__init__(input_col, output_col)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if self.requires[0] not in x.columns:
            raise ValueError(f"Input column {self.requires[0]} not found in DataFrame")
        return True

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        returns = x[self.requires[0]].values.astype(np.float64)
        result = rolling_entropy(returns, self.window, self.n_bins, method=0)
        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)


class LZEntropyTransform(SISOTransform):
    """
    SISO Transform for Lempel-Ziv entropy.

    Computes rolling LZ entropy using compression-based methods.

    Parameters
    ----------
    input_col : str
        Column name for returns
    window : int
        Rolling window size
    n_bins : int
        Number of bins for discretization

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from afmlkit.feature.core.entropy import LZEntropyTransform
    >>>
    >>> dates = pd.date_range('2023-01-01', periods=200, freq='D')
    >>> returns = np.random.randn(200) * 0.02
    >>> df = pd.DataFrame({'returns': returns}, index=dates)
    >>>
    >>> transform = LZEntropyTransform('returns', window=50)
    >>> entropy = transform(df, backend='nb')
    """

    def __init__(
        self,
        input_col: str,
        window: int,
        n_bins: int = 3
    ):
        self.window = window
        self.n_bins = n_bins
        output_col = f'lz_entropy_{window}'

        super().__init__(input_col, output_col)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if self.requires[0] not in x.columns:
            raise ValueError(f"Input column {self.requires[0]} not found in DataFrame")
        return True

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        returns = x[self.requires[0]].values.astype(np.float64)
        result = rolling_entropy(returns, self.window, self.n_bins, method=1)
        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)


class KontoyiannisEntropyTransform(SISOTransform):
    """
    SISO Transform for Kontoyiannis LZ entropy.

    Improved LZ entropy estimate using shortest non-redundant substrings.

    Parameters
    ----------
    input_col : str
        Column name for returns
    window : int
        Rolling window size
    n_bins : int
        Number of bins for discretization

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from afmlkit.feature.core.entropy import KontoyiannisEntropyTransform
    >>>
    >>> dates = pd.date_range('2023-01-01', periods=200, freq='D')
    >>> returns = np.random.randn(200) * 0.02
    >>> df = pd.DataFrame({'returns': returns}, index=dates)
    >>>
    >>> transform = KontoyiannisEntropyTransform('returns', window=50)
    >>> entropy = transform(df, backend='nb')
    """

    def __init__(
        self,
        input_col: str,
        window: int,
        n_bins: int = 3
    ):
        self.window = window
        self.n_bins = n_bins
        output_col = f'kontoyiannis_entropy_{window}'

        super().__init__(input_col, output_col)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if self.requires[0] not in x.columns:
            raise ValueError(f"Input column {self.requires[0]} not found in DataFrame")
        return True

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        returns = x[self.requires[0]].values.astype(np.float64)
        result = rolling_entropy(returns, self.window, self.n_bins, method=2)
        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)


class EntropyImpliedVolTransform(SISOTransform):
    """
    SISO Transform for entropy-implied volatility.

    Converts Shannon entropy to implied volatility under
    Gaussian assumptions.

    Parameters
    ----------
    input_col : str
        Column name for returns
    window : int
        Rolling window size
    n_bins : int
        Number of bins for discretization
    annualize : bool
        Whether to annualize the volatility

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from afmlkit.feature.core.entropy import EntropyImpliedVolTransform
    >>>
    >>> dates = pd.date_range('2023-01-01', periods=200, freq='D')
    >>> returns = np.random.randn(200) * 0.02
    >>> df = pd.DataFrame({'returns': returns}, index=dates)
    >>>
    >>> transform = EntropyImpliedVolTransform('returns', window=50)
    >>> vol = transform(df, backend='nb')
    """

    def __init__(
        self,
        input_col: str,
        window: int,
        n_bins: int = 3,
        annualize: bool = True
    ):
        self.window = window
        self.n_bins = n_bins
        self.annualize = annualize
        output_col = f'entropy_vol_{window}'

        super().__init__(input_col, output_col)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if self.requires[0] not in x.columns:
            raise ValueError(f"Input column {self.requires[0]} not found in DataFrame")
        return True

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        returns = x[self.requires[0]].values.astype(np.float64)
        entropy = rolling_entropy(returns, self.window, self.n_bins, method=0)

        # Convert entropy to volatility
        result = np.empty(len(entropy), dtype=np.float64)
        for i in range(len(entropy)):
            if np.isnan(entropy[i]):
                result[i] = np.nan
            else:
                result[i] = entropy_implied_volatility(entropy[i], self.annualize)

        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def compute_entropy(
    returns: pd.Series,
    window: int = 50,
    n_bins: int = 3,
    method: str = 'shannon'
) -> pd.Series:
    """
    Convenience function to compute rolling entropy.

    :param returns: Return Series
    :param window: Rolling window
    :param n_bins: Number of bins
    :param method: 'shannon', 'lz', or 'kontoyiannis'
    :returns: Rolling entropy Series
    """
    method_map = {'shannon': 0, 'lz': 1, 'kontoyiannis': 2}

    if method not in method_map:
        raise ValueError(f"Unknown method: {method}")

    result = rolling_entropy(
        returns.values,
        window,
        n_bins,
        method_map[method]
    )

    return pd.Series(result, index=returns.index, name=f'{method}_entropy_{window}')
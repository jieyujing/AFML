"""
PCA (Principal Component Analysis) Features for financial time series.

Implements orthogonal feature extraction using PCA, with automatic
component selection to achieve a target variance explanation threshold.

This module uses numpy/sklearn instead of Numba because:
1. SVD operations are already highly optimized in BLAS/LAPACK
2. PCA requires full matrix operations, not element-wise loops
3. The overhead of Python calls is negligible compared to SVD cost

References
----------
    López de Prado, M. (2018). *Advances in Financial Machine Learning*.
    Chapter 4 — Feature Importance Analysis.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Sequence
import pandas as pd

from afmlkit.feature.base import MIMOTransform


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def compute_pca(
    X: NDArray[np.float64],
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    """
    Compute PCA using SVD with automatic component selection.

    :param X: Data matrix of shape (n_samples, n_features), already centered
    :param n_components: Number of components (if None, auto-select based on variance)
    :param variance_threshold: Minimum cumulative variance to explain (default 0.95)
    :returns: Tuple of (components, explained_variance_ratio, transformed_data, n_components_selected)

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 10)
    >>> components, var_ratio, transformed, n = compute_pca(X)
    >>> print(f"Selected {n} components explaining {var_ratio[:n].sum():.2%} variance")
    """
    n_samples, n_features = X.shape

    # Compute SVD
    # X = U @ S @ Vt
    # Components are rows of Vt
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Explained variance ratio
    # variance_explained = S^2 / (n_samples - 1)
    total_var = np.sum(S ** 2)
    explained_variance_ratio = (S ** 2) / total_var

    # Cumulative variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Determine number of components
    if n_components is None:
        # Auto-select based on variance threshold
        n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1
        n_components = min(n_components, n_features)
    else:
        n_components = min(n_components, n_features)

    # Select top components
    components = Vt[:n_components]  # Shape: (n_components, n_features)
    explained_variance_ratio = explained_variance_ratio[:n_components]

    # Transform data
    # transformed = X @ Vt[:n_components].T
    # Or equivalently: transformed = U[:, :n_components] @ np.diag(S[:n_components])
    transformed = U[:, :n_components] * S[:n_components]

    return components, explained_variance_ratio, transformed, n_components


def compute_pca_with_standardization(
    X: NDArray[np.float64],
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95,
    standardize: bool = True
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict]:
    """
    Compute PCA with optional standardization (z-score normalization).

    :param X: Data matrix of shape (n_samples, n_features)
    :param n_components: Number of components (if None, auto-select)
    :param variance_threshold: Minimum cumulative variance (default 0.95)
    :param standardize: If True, standardize features to unit variance
    :returns: Tuple of (components, explained_variance_ratio, transformed_data, stats_dict)

    The stats_dict contains mean and std for applying to new data.
    """
    n_samples, n_features = X.shape

    # Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Optionally standardize
    if standardize:
        std = np.std(X_centered, axis=0, ddof=1)
        std[std < 1e-10] = 1.0  # Avoid division by zero
        X_scaled = X_centered / std
    else:
        std = np.ones(n_features)
        X_scaled = X_centered

    # Compute PCA
    components, var_ratio, transformed, n_comp = compute_pca(
        X_scaled, n_components, variance_threshold
    )

    stats = {
        'mean': mean,
        'std': std,
        'standardized': standardize
    }

    return components, var_ratio, transformed, stats


def transform_with_pca(
    X: NDArray[np.float64],
    components: NDArray[np.float64],
    mean: NDArray[np.float64],
    std: Optional[NDArray[np.float64]] = None
) -> NDArray[np.float64]:
    """
    Apply learned PCA transformation to new data.

    :param X: New data matrix of shape (n_samples, n_features)
    :param components: PCA components from compute_pca
    :param mean: Feature means from training
    :param std: Feature stds from training (if standardized)
    :returns: Transformed data of shape (n_samples, n_components)
    """
    X_centered = X - mean

    if std is not None:
        X_scaled = X_centered / std
    else:
        X_scaled = X_centered

    return X_scaled @ components.T


def rolling_pca(
    X: NDArray[np.float64],
    window: int,
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95,
    min_periods: Optional[int] = None
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Rolling PCA transformation.

    Computes PCA using a rolling window, returning the transformed
    data and the number of components used at each time point.

    :param X: Data matrix of shape (n_samples, n_features)
    :param window: Rolling window size
    :param n_components: Fixed number of components (if None, auto-select)
    :param variance_threshold: Minimum cumulative variance for auto-selection
    :param min_periods: Minimum periods for first calculation (default: window)
    :returns: Tuple of (transformed_data, n_components_per_time)

    Note: The number of output features may vary across time points.
    This function returns the maximum possible output size and pads with NaN.
    """
    n_samples, n_features = X.shape

    if min_periods is None:
        min_periods = window

    # Maximum possible components
    max_components = min(n_features, window)

    # Output arrays
    # We'll use max_components as the output dimension
    transformed = np.full((n_samples, max_components), np.nan, dtype=np.float64)
    n_components_used = np.zeros(n_samples, dtype=np.int64)

    for t in range(min_periods - 1, n_samples):
        # Extract window
        start_idx = max(0, t - window + 1)
        window_data = X[start_idx: t + 1]

        # Compute PCA on window
        try:
            _, _, window_transformed, n_comp = compute_pca(
                window_data, n_components, variance_threshold
            )

            # Store the last transformed point (current time)
            # Pad with NaN if fewer components than max
            transformed[t, :n_comp] = window_transformed[-1]
            n_components_used[t] = n_comp

        except Exception:
            # If PCA fails (e.g., constant features), leave as NaN
            pass

    return transformed, n_components_used


def compute_feature_correlation_distance(
    X: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute correlation-based distance matrix for hierarchical clustering.

    Distance = sqrt(0.5 * (1 - correlation))

    This satisfies metric space properties and is used in AFML
    for feature clustering.

    :param X: Data matrix of shape (n_samples, n_features)
    :returns: Distance matrix of shape (n_features, n_features)
    """
    # Compute correlation matrix
    corr = np.corrcoef(X.T)

    # Convert to distance
    # Ensure correlation is in [-1, 1]
    corr = np.clip(corr, -1, 1)

    # Distance formula from AFML
    dist = np.sqrt(0.5 * (1 - corr))

    return dist


# ---------------------------------------------------------------------------
# MIMO Transform
# ---------------------------------------------------------------------------

class PCATransform(MIMOTransform):
    """
    MIMO Transform for PCA feature extraction.

    Transforms multiple input features into orthogonal principal components,
    automatically selecting the number of components to explain the target
    variance threshold.

    Parameters
    ----------
    input_cols : Sequence[str]
        Names of input feature columns
    n_components : Optional[int]
        Fixed number of components (if None, auto-select)
    variance_threshold : float
        Minimum cumulative variance to explain (default 0.95)
    standardize : bool
        Whether to standardize features before PCA (default True)
    output_prefix : str
        Prefix for output column names (default 'pca')

    Examples
    --------
    Basic usage:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from afmlkit.feature.core.pca_features import PCATransform
    >>>
    >>> # Create sample features
    >>> n = 500
    >>> dates = pd.date_range('2023-01-01', periods=n, freq='D')
    >>> features = pd.DataFrame(
    ...     np.random.randn(n, 10),
    ...     columns=[f'feat_{i}' for i in range(10)],
    ...     index=dates
    ... )
    >>>
    >>> # Create and apply transform
    >>> transform = PCATransform(
    ...     input_cols=[f'feat_{i}' for i in range(10)],
    ...     variance_threshold=0.95
    ... )
    >>> result = transform(features, backend='pd')
    >>> print(f"Number of components: {len(result)}")

    Notes
    -----
    This transform is NOT Numba-accelerated because SVD operations
    are already highly optimized in BLAS/LAPACK libraries.
    """

    def __init__(
        self,
        input_cols: Sequence[str],
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
        standardize: bool = True,
        output_prefix: str = 'pca'
    ):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.standardize = standardize
        self.output_prefix = output_prefix

        # Output column names will be set after fitting
        # For now, use placeholder
        output_cols = [f'{output_prefix}_temp']

        super().__init__(input_cols, output_cols)

        # Store fitted parameters
        self._components: Optional[NDArray] = None
        self._mean: Optional[NDArray] = None
        self._std: Optional[NDArray] = None
        self._n_components_fitted: Optional[int] = None
        self._explained_variance_ratio: Optional[NDArray] = None

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        missing = [col for col in self.requires if col not in x.columns]
        if missing:
            raise ValueError(f"Input columns {missing} not found in DataFrame")

        return True

    def _fit(self, X: NDArray[np.float64]) -> None:
        """Fit PCA on the data."""
        components, var_ratio, _, stats = compute_pca_with_standardization(
            X,
            n_components=self.n_components,
            variance_threshold=self.variance_threshold,
            standardize=self.standardize
        )

        self._components = components
        self._mean = stats['mean']
        self._std = stats['std'] if self.standardize else None
        self._n_components_fitted = components.shape[0]
        self._explained_variance_ratio = var_ratio

        # Update output column names
        self.produces = [f'{self.output_prefix}_{i+1}' for i in range(self._n_components_fitted)]

    def _pd(self, x: pd.DataFrame) -> tuple[pd.Series, ...]:
        """Pandas implementation (also the only implementation)."""
        X = x[self.requires].values.astype(np.float64)

        # Fit if not already fitted
        if self._components is None:
            self._fit(X)

        # Transform
        transformed = transform_with_pca(
            X, self._components, self._mean, self._std
        )

        # Return as tuple of Series
        return tuple(
            pd.Series(transformed[:, i], index=x.index, name=self.output_name[i])
            for i in range(self._n_components_fitted)
        )

    def _nb(self, x: pd.DataFrame) -> tuple[pd.Series, ...]:
        """Same as pandas - PCA uses optimized BLAS/LAPACK."""
        return self._pd(x)

    def get_explained_variance_ratio(self) -> NDArray[np.float64]:
        """
        Get the explained variance ratio for each component.

        :returns: Array of variance ratios
        """
        if self._explained_variance_ratio is None:
            raise ValueError("Transform has not been fitted yet")
        return self._explained_variance_ratio

    def get_cumulative_variance(self) -> NDArray[np.float64]:
        """
        Get cumulative explained variance.

        :returns: Array of cumulative variance ratios
        """
        return np.cumsum(self.get_explained_variance_ratio())

    @property
    def output_name(self) -> list[str]:
        """Get output column names."""
        return self.produces


class RollingPCATransform(MIMOTransform):
    """
    MIMO Transform for rolling PCA feature extraction.

    Computes PCA on a rolling window, allowing the transformation
    to adapt to changing market conditions.

    Parameters
    ----------
    input_cols : Sequence[str]
        Names of input feature columns
    window : int
        Rolling window size
    n_components : Optional[int]
        Fixed number of components (if None, auto-select per window)
    variance_threshold : float
        Minimum cumulative variance for auto-selection
    output_prefix : str
        Prefix for output column names

    Notes
    -----
    Rolling PCA is computationally expensive: O(n * window * features^2)
    For large datasets, consider using non-rolling PCA or reducing window size.
    """

    def __init__(
        self,
        input_cols: Sequence[str],
        window: int,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
        output_prefix: str = 'rolling_pca'
    ):
        self.window = window
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.output_prefix = output_prefix

        # Determine max output components
        n_features = len(input_cols)
        max_components = min(n_features, window)

        output_cols = [f'{output_prefix}_{i+1}' for i in range(max_components)]

        super().__init__(input_cols, output_cols)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        if not isinstance(x, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        missing = [col for col in self.requires if col not in x.columns]
        if missing:
            raise ValueError(f"Input columns {missing} not found in DataFrame")

        return True

    def _pd(self, x: pd.DataFrame) -> tuple[pd.Series, ...]:
        """Compute rolling PCA."""
        X = x[self.requires].values.astype(np.float64)

        transformed, _ = rolling_pca(
            X,
            window=self.window,
            n_components=self.n_components,
            variance_threshold=self.variance_threshold
        )

        n_components = transformed.shape[1]

        return tuple(
            pd.Series(transformed[:, i], index=x.index, name=self.output_name[i])
            for i in range(n_components)
        )

    def _nb(self, x: pd.DataFrame) -> tuple[pd.Series, ...]:
        """Same as pandas."""
        return self._pd(x)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def pca_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    variance_threshold: float = 0.95,
    standardize: bool = True
) -> pd.DataFrame:
    """
    Convenience function to compute PCA features.

    :param df: DataFrame with feature columns
    :param feature_cols: Names of feature columns to use
    :param variance_threshold: Minimum cumulative variance
    :param standardize: Whether to standardize features
    :returns: DataFrame with PCA columns

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> dates = pd.date_range('2023-01-01', periods=500, freq='D')
    >>> df = pd.DataFrame(
    ...     np.random.randn(500, 5),
    ...     columns=['a', 'b', 'c', 'd', 'e'],
    ...     index=dates
    ... )
    >>> result = pca_features(df, ['a', 'b', 'c', 'd', 'e'])
    >>> result.columns.tolist()
    ['pca_1', 'pca_2', ...]
    """
    transform = PCATransform(
        input_cols=feature_cols,
        variance_threshold=variance_threshold,
        standardize=standardize
    )

    results = transform(df, backend='pd')

    return pd.DataFrame({s.name: s for s in results})
"""
Volume Distribution Factor Transforms.

Implements volume distribution statistics transforms including Shannon entropy,
skewness, kurtosis, peak ratio, and differential standard deviation.

References
----------
    Marcos López de Prado. *Advances in Financial Machine Learning*.
    Marcos López de Prado. *Machine Learning for Asset Managers*.
"""

import numpy as np
import pandas as pd

from afmlkit.feature.base import SISOTransform


def _entropy_func(x: pd.DataFrame) -> float:
    """
    Compute Shannon entropy of volume distribution for a single resample window.

    :param x: DataFrame with 'amount' column for the resample window.
    :returns: Shannon entropy value.
    """
    amounts = x['amount'].values
    if len(amounts) == 0 or amounts.sum() == 0:
        return np.nan

    binned = pd.cut(amounts, bins=5)
    counts = pd.Series(binned).value_counts(normalize=True).values
    counts = counts[counts > 0]
    return - (counts * np.log(counts)).sum()


class VolEntropyTransform(SISOTransform):
    """
    QIML0514: Shannon entropy of volume distribution.

    Measures the uniformity of volume distribution across bins. Higher entropy
    indicates more uniform (unstable) distribution; lower entropy indicates
    concentrated (stable) distribution.

    The transform uses ``groupby('code').resample(frequency).apply(_entropy_func)``
    to compute entropy per instrument per time bucket.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param n_bins: Number of bins for volume distribution binning.
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_entropy').

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 10,
        ...     'amount': [100, 200, 150, 300, 250, 180, 220, 270, 190, 210],
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=10, freq='h')
        >>> transform = VolEntropyTransform(frequency='3h')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        n_bins: int = 5,
        input_col: str = 'amount',
        output_col: str = 'vol_entropy',
    ):
        super().__init__(input_col, output_col)
        self.frequency = frequency
        self.n_bins = n_bins

    def get_params(self) -> dict:
        """
        Get the parameters of the transform.

        :returns: Dictionary of current parameter values.
        """
        return {
            'frequency': self.frequency,
            'n_bins': self.n_bins,
            'input_col': self.requires[0],
            'output_col': self.produces[0],
        }

    def set_params(self, **params):
        """
        Set the parameters of the transform.

        :param params: Keyword parameters to set.
        :returns: self
        """
        if 'frequency' in params:
            self.frequency = params['frequency']
        if 'n_bins' in params:
            self.n_bins = params['n_bins']
        if 'input_col' in params:
            self.requires = [params['input_col']]
        if 'output_col' in params:
            self.produces = [params['output_col']]
        return self

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        """
        Compute Shannon entropy of volume distribution using pandas.

        :param x: DataFrame with 'code' and 'amount' columns.
        :returns: pd.Series with entropy values, index aligned to input DataFrame.
        """
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")

        def _entropy_func_inner(x_inner: pd.Series) -> float:
            """Compute entropy from a Series of amounts."""
            amounts = x_inner.values
            if len(amounts) == 0 or amounts.sum() == 0:
                return np.nan
            binned = pd.cut(amounts, bins=self.n_bins)
            counts = pd.Series(binned).value_counts(normalize=True).values
            counts = counts[counts > 0]
            return - (counts * np.log(counts)).sum()

        # Reset index to get timestamp as column, preserving for later
        original_index = x.index
        df_reset = x.reset_index()
        timestamp_col = df_reset.columns[0]  # Typically 'timestamp' or 'index' for DatetimeIndex

        # Compute entropy per (code, time_bucket) using explicit loop
        # This works consistently for single or multiple codes
        records = []
        for code, group in df_reset.groupby('code'):
            group_indexed = group.set_index(timestamp_col)
            for period_start, period_group in group_indexed.resample(self.frequency):
                ent = _entropy_func_inner(period_group['amount'])
                records.append({
                    'code': code,
                    'bucket': period_start,
                    self.produces[0]: ent,
                })
        entropy_df = pd.DataFrame(records)

        # Create time bucket column for merging
        df_reset['bucket'] = df_reset[timestamp_col].dt.floor(self.frequency)

        # Merge entropy values back
        merged = df_reset.merge(entropy_df, on=['code', 'bucket'], how='left')

        result = merged[self.produces[0]].values
        return pd.Series(result, index=original_index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.

        Volume distribution entropy involves groupby-resample semantics
        which are not efficiently expressible in pure NumPy/Numba.
        """
        return self._pd(x)

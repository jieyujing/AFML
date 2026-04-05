"""
Volume distribution factor transforms — QIML series (0514/0607/0618/0124/0116).

These transforms operate on OHLCV DataFrames and use pandas groupby+resample
semantics. They are intentionally NOT Numba-accelerated because their value
lies in the cross-sectional aggregation pattern, not in element-wise loops.
"""

import numpy as np
import pandas as pd
from typing import Union

from afmlkit.feature.base import SISOTransform


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

    def _pd(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Compute Shannon entropy of volume distribution using pandas.

        :param x: DataFrame with 'code' and 'amount' columns.
        :returns: pd.Series with entropy values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("VolEntropyTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")

        def _entropy(x_inner: pd.DataFrame) -> float:
            """Shannon entropy of volume distribution for one resample window."""
            amounts_arr = np.asarray(x_inner['amount'].values, dtype=np.float64)
            if len(amounts_arr) < 2 or amounts_arr.sum() == 0:
                return np.nan
            binned = pd.cut(amounts_arr, bins=self.n_bins)
            vc = pd.Series(binned).value_counts(normalize=True)
            counts = np.asarray(vc.values, dtype=np.float64)
            counts = counts[counts > 0]
            return - (counts * np.log(counts)).sum()

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            ent = _entropy(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'entropy': ent})

        result_df = pd.DataFrame(records).set_index('idx')
        entropy_val: np.ndarray = result_df['entropy'].values  # type: ignore[assignment]
        out = pd.Series(entropy_val, index=original_index, name=self.output_name)
        return out

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.

        Volume distribution entropy involves groupby-resample semantics
        which are not efficiently expressible in pure NumPy/Numba.
        """
        return self._pd(x)

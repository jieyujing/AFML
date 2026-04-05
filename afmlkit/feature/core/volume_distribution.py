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
        # Override output name to include n_bins for uniqueness
        self.produces = [f'{output_col}_{n_bins}bins']

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


class VolSkewTransform(SISOTransform):
    """
    QIML0607: Skewness of volume distribution across 5-min sub-buckets.

    Measures the asymmetry of volume distribution within each resample window.
    Positive skew indicates volume concentrated at the start; negative skew
    indicates volume concentrated at the end.

    The transform uses ``groupby('code').resample(frequency).apply(_skew_func)``
    to compute skewness per instrument per time bucket.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_skew').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 20,
        ...     'amount': np.random.rand(20) * 1000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=20, freq='h')
        >>> transform = VolSkewTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'vol_skew',
    ):
        super().__init__(input_col, output_col)
        self.frequency = frequency

    def get_params(self) -> dict:
        """
        Get the parameters of the transform.

        :returns: Dictionary of current parameter values.
        """
        return {
            'frequency': self.frequency,
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
        if 'input_col' in params:
            self.requires = [params['input_col']]
        if 'output_col' in params:
            self.produces = [params['output_col']]
        return self

    def _pd(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Compute skewness of volume distribution using pandas.

        :param x: DataFrame with 'code' and 'amount' columns.
        :returns: pd.Series with skewness values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("VolSkewTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")

        def _skew(x_inner: pd.DataFrame) -> float:
            """Skewness of volume distribution for one resample window."""
            sub = x_inner.resample('5min')['amount'].sum().dropna()
            if sub.sum() == 0 or len(sub) < 3:
                return np.nan
            share = sub / sub.sum()
            return share.skew()

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            skew_val = _skew(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'vol_skew': skew_val})

        result_df = pd.DataFrame(records).set_index('idx')
        skew_arr: np.ndarray = result_df['vol_skew'].values  # type: ignore[assignment]
        out = pd.Series(skew_arr, index=original_index, name=self.output_name)
        return out

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.

        Volume distribution skewness involves groupby-resample semantics
        which are not efficiently expressible in pure NumPy/Numba.
        """
        return self._pd(x)


class VolKurtTransform(SISOTransform):
    """
    QIML0618: Kurtosis of volume distribution across 5-min sub-buckets.

    Measures the peakedness (heavy tails) of volume distribution within each
    resample window. Higher kurtosis indicates more extreme volume outliers.

    The transform uses ``groupby('code').resample(frequency).apply(_kurt_func)``
    to compute kurtosis per instrument per time bucket.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_kurt').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 20,
        ...     'amount': np.random.rand(20) * 1000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=20, freq='h')
        >>> transform = VolKurtTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'vol_kurt',
    ):
        super().__init__(input_col, output_col)
        self.frequency = frequency

    def get_params(self) -> dict:
        """
        Get the parameters of the transform.

        :returns: Dictionary of current parameter values.
        """
        return {
            'frequency': self.frequency,
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
        if 'input_col' in params:
            self.requires = [params['input_col']]
        if 'output_col' in params:
            self.produces = [params['output_col']]
        return self

    def _pd(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Compute kurtosis of volume distribution using pandas.

        :param x: DataFrame with 'code' and 'amount' columns.
        :returns: pd.Series with kurtosis values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("VolKurtTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")

        def _kurt(x_inner: pd.DataFrame) -> float:
            """Kurtosis of volume distribution for one resample window."""
            sub = x_inner.resample('5min')['amount'].sum().dropna()
            if sub.sum() == 0 or len(sub) < 4:
                return np.nan
            share = sub / sub.sum()
            return share.kurt()

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            kurt_val = _kurt(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'vol_kurt': kurt_val})

        result_df = pd.DataFrame(records).set_index('idx')
        kurt_arr: np.ndarray = result_df['vol_kurt'].values  # type: ignore[assignment]
        out = pd.Series(kurt_arr, index=original_index, name=self.output_name)
        return out

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.

        Volume distribution kurtosis involves groupby-resample semantics
        which are not efficiently expressible in pure NumPy/Numba.
        """
        return self._pd(x)


class VolPeakTransform(SISOTransform):
    """
    QIML0124: Volume peak ratio — fraction of bars whose volume exceeds
    mean + 1 standard deviation within each resample window.

    A higher peak ratio indicates more volatile (unstable) volume distribution
    with concentrated spikes; a lower ratio indicates more stable distribution.

    The transform uses ``groupby('code').resample(frequency).apply(_peak_func)``
    to compute the peak ratio per instrument per time bucket.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_peak').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 20,
        ...     'amount': np.random.rand(20) * 1000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=20, freq='h')
        >>> transform = VolPeakTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'vol_peak',
    ):
        super().__init__(input_col, output_col)
        self.frequency = frequency

    def get_params(self) -> dict:
        """
        Get the parameters of the transform.

        :returns: Dictionary of current parameter values.
        """
        return {
            'frequency': self.frequency,
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
        if 'input_col' in params:
            self.requires = [params['input_col']]
        if 'output_col' in params:
            self.produces = [params['output_col']]
        return self

    def _pd(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Compute volume peak ratio using pandas.

        :param x: DataFrame with 'code' and 'amount' columns.
        :returns: pd.Series with peak ratio values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("VolPeakTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")

        def _peak(x_inner: pd.DataFrame) -> float:
            """Peak ratio of volume distribution for one resample window."""
            n = len(x_inner)
            if n < 2:
                return np.nan
            amounts_arr = np.asarray(x_inner['amount'].values, dtype=np.float64)
            mean = amounts_arr.mean()
            std = amounts_arr.std()
            if std == 0:
                return np.nan
            threshold = mean + std
            count = (amounts_arr > threshold).sum()
            return count / n

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            peak_val = _peak(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'vol_peak': peak_val})

        result_df = pd.DataFrame(records).set_index('idx')
        peak_arr: np.ndarray = result_df['vol_peak'].values  # type: ignore[assignment]
        out = pd.Series(peak_arr, index=original_index, name=self.output_name)
        return out

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.

        Volume peak ratio involves groupby-resample semantics
        which are not efficiently expressible in pure NumPy/Numba.
        """
        return self._pd(x)


class VolDiffStdTransform(SISOTransform):
    """
    QIML0116: Standard deviation of normalized volume diff (average trade size).

    Measures the volatility of average trade size (amount/count) across bars
    within each resample window. Higher values indicate more unstable
    trade-size patterns.

    The transform uses ``groupby('code').resample(frequency).apply(_diff_std_func)``
    to compute the diff std per instrument per time bucket.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_diff_std').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 20,
        ...     'amount': np.random.rand(20) * 1000 + 100,
        ...     'count': np.random.randint(1, 50, size=20),
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=20, freq='h')
        >>> transform = VolDiffStdTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'vol_diff_std',
    ):
        super().__init__(input_col, output_col)
        self.frequency = frequency

    def get_params(self) -> dict:
        """
        Get the parameters of the transform.

        :returns: Dictionary of current parameter values.
        """
        return {
            'frequency': self.frequency,
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
        if 'input_col' in params:
            self.requires = [params['input_col']]
        if 'output_col' in params:
            self.produces = [params['output_col']]
        return self

    def _pd(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Compute volume diff std using pandas.

        :param x: DataFrame with 'code', 'amount', and 'count' columns.
        :returns: pd.Series with diff std values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("VolDiffStdTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'count' not in x.columns:
            raise ValueError("VolDiffStdTransform requires a DataFrame with 'count' column.")

        def _diff_std(x_inner: pd.DataFrame) -> float:
            """Std of normalized volume diff for one resample window."""
            xx = x_inner['amount'] / x_inner['count']
            xx = xx.replace([np.inf, -np.inf], np.nan).dropna()
            if len(xx) < 2 or xx.mean() == 0:
                return np.nan
            return (xx.diff().abs() / xx.mean()).std()

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            diff_std_val = _diff_std(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'vol_diff_std': diff_std_val})

        result_df = pd.DataFrame(records).set_index('idx')
        diff_std_arr: np.ndarray = result_df['vol_diff_std'].values  # type: ignore[assignment]
        out = pd.Series(diff_std_arr, index=original_index, name=self.output_name)
        return out

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.

        Volume diff std involves groupby-resample semantics
        which are not efficiently expressible in pure NumPy/Numba.
        """
        return self._pd(x)

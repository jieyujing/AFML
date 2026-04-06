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


class AmbiguityTransform(SISOTransform):
    """
    QIML0212: Ambiguity aversion factor — difference of volume and amount ratios.

    Measures the流动性 cost when investors rush to close positions due to
    ambiguous volatility. The factor is the difference between volume-based
    and amount-based ambiguity ratios.

    The transform uses ``groupby('code').resample(frequency).apply(_ambiguity_func)``
    to compute the factor per instrument per time bucket.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_ambiguity').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 50,
        ...     'close': 100 + np.cumsum(np.random.randn(50) * 0.5),
        ...     'volume': np.random.rand(50) * 1000,
        ...     'amount': np.random.rand(50) * 100000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=50, freq='min')
        >>> transform = AmbiguityTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'vol_ambiguity',
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
        Compute ambiguity aversion factor using pandas.

        :param x: DataFrame with 'code', 'close', 'volume', and 'amount' columns.
        :returns: pd.Series with ambiguity values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("AmbiguityTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'close' not in x.columns or 'volume' not in x.columns:
            raise ValueError("AmbiguityTransform requires 'close' and 'volume' columns.")

        def _ambiguity(x_inner: pd.DataFrame) -> float:
            """Ambiguity ratio difference for one resample window."""
            x_ret = x_inner['close'].pct_change()
            x_amb = x_ret.rolling(5).std().rolling(5).std()
            x_fogging = x_inner[x_amb > x_amb.mean()]
            if len(x_fogging) == 0 or x_inner['volume'].mean() == 0:
                return np.nan
            x_amb_ratio = x_fogging['volume'].mean() / x_inner['volume'].mean()
            x_amb_ratio_1 = x_fogging['amount'].mean() / x_inner['amount'].mean() if x_inner['amount'].mean() != 0 else np.nan
            if np.isnan(x_amb_ratio_1):
                return np.nan
            return x_amb_ratio - x_amb_ratio_1

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _ambiguity(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.

        Ambiguity involves rolling window and conditional filtering
        which are not efficiently expressible in pure NumPy/Numba.
        """
        return self._pd(x)


class AmbiguityVolTransform(SISOTransform):
    """
    QIML0301: Ambiguity aversion factor — volume-based ratio.

    Measures the degree of trading when ambiguity is high, indicating
    investor ambiguity aversion via volume ratio in foggy periods.

    The transform uses ``groupby('code').resample(frequency).apply(_ambiguity_func)``
    to compute the factor per instrument per time bucket.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_ambiguity_vol').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 50,
        ...     'close': 100 + np.cumsum(np.random.randn(50) * 0.5),
        ...     'volume': np.random.rand(50) * 1000,
        ...     'amount': np.random.rand(50) * 100000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=50, freq='min')
        >>> transform = AmbiguityVolTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'vol_ambiguity_vol',
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
        Compute volume-based ambiguity aversion using pandas.

        :param x: DataFrame with 'code', 'close', and 'volume' columns.
        :returns: pd.Series with ambiguity values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("AmbiguityVolTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'close' not in x.columns or 'volume' not in x.columns:
            raise ValueError("AmbiguityVolTransform requires 'close' and 'volume' columns.")

        def _ambiguity(x_inner: pd.DataFrame) -> float:
            """Ambiguity volume ratio for one resample window."""
            x_ret = x_inner['close'].pct_change()
            x_amb = x_ret.rolling(5).std().rolling(5).std()
            x_fogging = x_inner[x_amb > x_amb.mean()]
            if len(x_fogging) == 0 or x_inner['volume'].mean() == 0:
                return np.nan
            return x_fogging['volume'].mean() / x_inner['volume'].mean()

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _ambiguity(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.
        """
        return self._pd(x)


class AmbiguityCountTransform(SISOTransform):
    """
    QIML0331: Ambiguity aversion factor — count-based ratio.

    Measures the degree of trading when ambiguity is high, indicating
    investor ambiguity aversion via trade count ratio in foggy periods.

    The transform uses ``groupby('code').resample(frequency).apply(_ambiguity_func)``
    to compute the factor per instrument per time bucket.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_ambiguity_count').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 50,
        ...     'close': 100 + np.cumsum(np.random.randn(50) * 0.5),
        ...     'volume': np.random.rand(50) * 1000,
        ...     'count': np.random.randint(1, 100, size=50),
        ...     'amount': np.random.rand(50) * 100000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=50, freq='min')
        >>> transform = AmbiguityCountTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'vol_ambiguity_count',
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
        Compute count-based ambiguity aversion using pandas.

        :param x: DataFrame with 'code', 'close', and 'count' columns.
        :returns: pd.Series with ambiguity values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("AmbiguityCountTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'close' not in x.columns or 'count' not in x.columns:
            raise ValueError("AmbiguityCountTransform requires 'close' and 'count' columns.")

        def _ambiguity(x_inner: pd.DataFrame) -> float:
            """Ambiguity count ratio for one resample window."""
            x_ret = x_inner['close'].pct_change()
            x_amb = x_ret.rolling(5).std().rolling(5).std()
            x_fogging = x_inner[x_amb > x_amb.mean()]
            if len(x_fogging) == 0 or x_inner['count'].mean() == 0:
                return np.nan
            return x_fogging['count'].mean() / x_inner['count'].mean()

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _ambiguity(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.
        """
        return self._pd(x)


class PShapeTransform(SISOTransform):
    """
    QIML0401: Volume profile P-shape — low price level relative to max.

    Computes the volume-weighted price level that accumulates 50% of total
    volume (starting from the peak volume price), then returns the normalized
    distance from that low price level to the maximum close price.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_p_shape').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 50,
        ...     'close': np.random.rand(50) * 10 + 95,
        ...     'amount': np.random.rand(50) * 100000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=50, freq='min')
        >>> transform = PShapeTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'vol_p_shape',
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
        Compute P-shape factor using pandas.

        :param x: DataFrame with 'code', 'close', and 'amount' columns.
        :returns: pd.Series with P-shape values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("PShapeTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'close' not in x.columns:
            raise ValueError("PShapeTransform requires 'close' column.")

        def _p_shape(x_inner: pd.DataFrame) -> float:
            """P-shape value for one resample window."""
            vol_sum = x_inner.groupby('close')['amount'].sum()
            if len(vol_sum) == 0 or vol_sum.sum() == 0:
                return np.nan
            vol_acc_sum = vol_sum.sum()
            idx = np.argmax(vol_sum)
            ratio = vol_sum.iloc[idx] / vol_acc_sum
            num = 0
            while ratio < 0.5 and num <= idx:
                num += 1
                if idx - num < 0:
                    break
                window = vol_sum.iloc[max(0, idx - num): idx + num + 1]
                ratio = window.sum() / vol_acc_sum
            try:
                if idx - num >= 0:
                    vsa_low = vol_sum.index[idx - num]
                else:
                    vsa_low = np.min(vol_sum.index)
            except Exception:
                vsa_low = np.min(vol_sum.index)
            if vsa_low == 0:
                return np.nan
            return (x_inner['close'].max() - vsa_low) / vsa_low

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _p_shape(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.
        """
        return self._pd(x)


class BShapeTransform(SISOTransform):
    """
    QIML0413: Volume profile B-shape — high price level relative to min.

    Computes the volume-weighted price level that accumulates 50% of total
    volume (starting from the peak volume price), then returns the normalized
    distance from the minimum close price to that high price level.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_b_shape').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 50,
        ...     'close': np.random.rand(50) * 10 + 95,
        ...     'amount': np.random.rand(50) * 100000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=50, freq='min')
        >>> transform = BShapeTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'vol_b_shape',
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
        Compute B-shape factor using pandas.

        :param x: DataFrame with 'code', 'close', and 'amount' columns.
        :returns: pd.Series with B-shape values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("BShapeTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'close' not in x.columns:
            raise ValueError("BShapeTransform requires 'close' column.")

        def _b_shape(x_inner: pd.DataFrame) -> float:
            """B-shape value for one resample window."""
            vol_sum = x_inner.groupby('close')['amount'].sum()
            if len(vol_sum) == 0 or vol_sum.sum() == 0:
                return np.nan
            vol_acc_sum = vol_sum.sum()
            idx = np.argmax(vol_sum)
            ratio = vol_sum.iloc[idx] / vol_acc_sum
            num = 0
            while ratio < 0.5 and idx + num < len(vol_sum):
                num += 1
                if idx + num >= len(vol_sum):
                    break
                window = vol_sum.iloc[idx: min(idx + num + 1, len(vol_sum))]
                ratio = window.sum() / vol_acc_sum
            try:
                if idx + num < len(vol_sum):
                    vsa_high = vol_sum.index[idx + num]
                else:
                    vsa_high = np.max(vol_sum.index)
            except Exception:
                vsa_high = np.max(vol_sum.index)
            if vsa_high == 0:
                return np.nan
            return (vsa_high - x_inner['close'].min()) / vsa_high

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _b_shape(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.
        """
        return self._pd(x)


class UnanimousBuyingTransform(SISOTransform):
    """
    QIML0629: Unanimous buying pressure — up/down volume ratio.

    Measures the ratio of up-volume to down-volume for bars where price
    movement is unambiguous (alpha > 0.5). Returns (vol_up + vol_down) /
    (vol_up - vol_down) when vol_up != vol_down, else 0.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_unan_buy').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 50,
        ...     'open': 100 + np.random.randn(50).cumsum(),
        ...     'high': 100 + np.random.randn(50).cumsum() + 1,
        ...     'low': 100 + np.random.randn(50).cumsum() - 1,
        ...     'close': 100 + np.random.randn(50).cumsum(),
        ...     'amount': np.random.rand(50) * 100000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=50, freq='min')
        >>> transform = UnanimousBuyingTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'vol_unan_buy',
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
        Compute unanimous buying pressure using pandas.

        :param x: DataFrame with 'code', 'open', 'high', 'low', 'close', 'amount' columns.
        :returns: pd.Series with unanimous buying values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("UnanimousBuyingTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        for col in ('open', 'high', 'low', 'close', 'amount'):
            if col not in x.columns:
                raise ValueError(f"UnanimousBuyingTransform requires '{col}' column.")

        def _unan_buy(x_inner: pd.DataFrame) -> float:
            """Unanimous buying ratio for one resample window."""
            denom = (x_inner['high'] - x_inner['low']).replace(0, np.nan)
            alpha = (x_inner['close'] - x_inner['open']).abs() / denom.fillna(0)
            total = x_inner['amount'].sum()
            if total == 0:
                return np.nan
            vol_up_mask = (alpha > 0.5) & (x_inner['close'].pct_change() > 0)
            vol_down_mask = (alpha > 0.5) & (x_inner['close'].pct_change() < 0)
            vol_up = x_inner['amount'][vol_up_mask].sum() / total
            vol_down = x_inner['amount'][vol_down_mask].sum() / total
            diff = vol_up - vol_down
            if diff == 0:
                return 0.0
            return (vol_up + vol_down) / diff

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _unan_buy(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.
        """
        return self._pd(x)


class VolStdTransform(SISOTransform):
    """
    QIML0722: Volume standard deviation across resampled buckets.

    Computes the standard deviation of the fraction of volume in each
    5-minute sub-bucket relative to the total window volume.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_vol_std').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 50,
        ...     'amount': np.random.rand(50) * 100000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=50, freq='min')
        >>> transform = VolStdTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'vol_vol_std',
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
        Compute volume std using pandas.

        :param x: DataFrame with 'code' and 'amount' columns.
        :returns: pd.Series with volume std values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("VolStdTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'amount' not in x.columns:
            raise ValueError("VolStdTransform requires 'amount' column.")

        def _vol_std(x_inner: pd.DataFrame) -> float:
            """Volume std for one resample window."""
            total = x_inner['amount'].sum()
            if total == 0:
                return np.nan
            sub = x_inner.resample('5min')['amount'].sum()
            sub_ratio = sub / total
            if len(sub_ratio) < 2:
                return np.nan
            return sub_ratio.std()

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _vol_std(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.
        """
        return self._pd(x)


class VolumeVarTransform(SISOTransform):
    """
    QIML0806: Volume variance within the resample window.

    Computes the variance of the 'volume' column within each resampled
    time bucket.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'volume').
    :param output_col: Output column suffix (default 'vol_vol_var').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 50,
        ...     'volume': np.random.rand(50) * 1000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=50, freq='min')
        >>> transform = VolumeVarTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'volume',
        output_col: str = 'vol_vol_var',
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
        Compute volume variance using pandas.

        :param x: DataFrame with 'code' and 'volume' columns.
        :returns: pd.Series with volume variance values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("VolumeVarTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'volume' not in x.columns:
            raise ValueError("VolumeVarTransform requires 'volume' column.")

        def _vol_var(x_inner: pd.DataFrame) -> float:
            """Volume variance for one resample window."""
            return float(x_inner['volume'].var())

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _vol_var(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.
        """
        return self._pd(x)


class UnanimousTradingTransform(SISOTransform):
    """
    QIML0827: Unanimous trading volume — proportion of volume in unambiguous bars.

    Measures the proportion of total volume that occurs in bars where price
    movement is unambiguous (alpha > 0.5), including both up and down moves.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_unan_trade').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 50,
        ...     'open': 100 + np.random.randn(50).cumsum(),
        ...     'high': 100 + np.random.randn(50).cumsum() + 1,
        ...     'low': 100 + np.random.randn(50).cumsum() - 1,
        ...     'close': 100 + np.random.randn(50).cumsum(),
        ...     'amount': np.random.rand(50) * 100000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=50, freq='min')
        >>> transform = UnanimousTradingTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'vol_unan_trade',
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
        Compute unanimous trading volume using pandas.

        :param x: DataFrame with 'code', 'open', 'high', 'low', 'close', 'amount' columns.
        :returns: pd.Series with unanimous trading values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("UnanimousTradingTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        for col in ('open', 'high', 'low', 'close', 'amount'):
            if col not in x.columns:
                raise ValueError(f"UnanimousTradingTransform requires '{col}' column.")

        def _unan_trade(x_inner: pd.DataFrame) -> float:
            """Unanimous trading volume ratio for one resample window."""
            denom = (x_inner['high'] - x_inner['low']).replace(0, np.nan)
            alpha = (x_inner['close'] - x_inner['open']).abs() / denom.fillna(0)
            total = x_inner['amount'].sum()
            if total == 0:
                return np.nan
            vol_unan = x_inner['amount'][alpha > 0.5].sum() / total
            return vol_unan

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _unan_trade(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.
        """
        return self._pd(x)


class TailVolumeRatioTransform(SISOTransform):
    """
    QIML0914: Tail volume ratio — volume share in last 5 minutes of the hour.

    Measures the proportion of total volume that occurs in the last 5 minutes
    (minute >= 55) within each resample window. Higher values indicate
    concentration of volume near the close.

    The transform uses ``groupby('code').resample(frequency)`` semantics
    to compute the ratio per instrument per time bucket.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'volume').
    :param output_col: Output column suffix (default 'tail_vol_ratio').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 120,
        ...     'volume': np.random.rand(120) * 1000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=120, freq='min')
        >>> transform = TailVolumeRatioTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'volume',
        output_col: str = 'tail_vol_ratio',
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
        Compute tail volume ratio using pandas.

        :param x: DataFrame with 'code' and 'volume' columns.
        :returns: pd.Series with tail volume ratio values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("TailVolumeRatioTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'volume' not in x.columns:
            raise ValueError("TailVolumeRatioTransform requires 'volume' column.")

        def _tail_ratio(x_inner: pd.DataFrame) -> float:
            """Tail volume ratio for one resample window."""
            minutes = pd.DatetimeIndex(x_inner.index).minute  # type: ignore[union-abstract]
            total_vol = x_inner['volume'].sum()
            if total_vol == 0:
                return 0.0
            late_mask = minutes >= 55
            ratio = x_inner.loc[late_mask, 'volume'].sum() / total_vol
            return ratio if not pd.isna(ratio) else 0.0

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _tail_ratio(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.
        """
        return self._pd(x)


class VolumeRatioTransform(SISOTransform):
    """
    QIML1014: Volume ratio — open-period amount over close-period amount.

    Measures the ratio of amount traded in the first 5 minutes (minute < 5)
    to the last 5 minutes (minute >= 55) within each resample window.
    Values > 1 indicate more volume at the open; values < 1 indicate
    more volume at the close.

    The transform uses ``groupby('code').resample(frequency)`` semantics
    to compute the ratio per instrument per time bucket.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_ratio').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 120,
        ...     'amount': np.random.rand(120) * 100000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=120, freq='min')
        >>> transform = VolumeRatioTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'vol_ratio',
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
        Compute volume ratio using pandas.

        :param x: DataFrame with 'code' and 'amount' columns.
        :returns: pd.Series with volume ratio values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("VolumeRatioTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'amount' not in x.columns:
            raise ValueError("VolumeRatioTransform requires 'amount' column.")

        def _vol_ratio(x_inner: pd.DataFrame) -> float:
            """Volume ratio (open / close) for one resample window."""
            minutes = pd.DatetimeIndex(x_inner.index).minute  # type: ignore[union-abstract]
            total_amt = x_inner['amount'].sum()
            if total_amt == 0:
                return 0.0
            open_mask = minutes < 5
            close_mask = minutes >= 55
            open_amt = x_inner.loc[open_mask, 'amount'].sum()
            close_amt = x_inner.loc[close_mask, 'amount'].sum()
            if close_amt == 0:
                return 0.0
            return open_amt / close_amt

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _vol_ratio(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.
        """
        return self._pd(x)


class VolumeShareTransform(SISOTransform):
    """
    QIML1021: Volume share — sum of open and close period amount shares.

    Measures the combined proportion of total amount traded in the first
    5 minutes (minute < 5) and the last 5 minutes (minute >= 55) within
    each resample window. Higher values indicate concentration of volume
    near market open and close.

    The transform uses ``groupby('code').resample(frequency)`` semantics
    to compute the share per instrument per time bucket.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_share').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 120,
        ...     'amount': np.random.rand(120) * 100000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=120, freq='min')
        >>> transform = VolumeShareTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'vol_share',
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
        Compute volume share using pandas.

        :param x: DataFrame with 'code' and 'amount' columns.
        :returns: pd.Series with volume share values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("VolumeShareTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'amount' not in x.columns:
            raise ValueError("VolumeShareTransform requires 'amount' column.")

        def _vol_share(x_inner: pd.DataFrame) -> float:
            """Volume share (open + close) for one resample window."""
            minutes = pd.DatetimeIndex(x_inner.index).minute  # type: ignore[union-abstract]
            total_amt = x_inner['amount'].sum()
            if total_amt == 0:
                return 0.0
            open_mask = minutes < 5
            close_mask = minutes >= 55
            open_amt = x_inner.loc[open_mask, 'amount'].sum()
            close_amt = x_inner.loc[close_mask, 'amount'].sum()
            return (open_amt + close_amt) / total_amt

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _vol_share(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.
        """
        return self._pd(x)


class AmountQuantileTransform(SISOTransform):
    """
    QIML1105: Amount quantile — normalized price-per-unit quantile.

    Computes the normalized quantile of price-per-unit (volume/count) sorted
    values, dropping the last 2 entries. The 10th percentile of the sorted
    price-per-unit series is normalized by its range.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'volume').
    :param output_col: Output column suffix (default 'amt_quantile').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 50,
        ...     'volume': np.random.rand(50) * 1000,
        ...     'count': np.random.randint(1, 100, size=50),
        ...     'amount': np.random.rand(50) * 100000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=50, freq='min')
        >>> transform = AmountQuantileTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'volume',
        output_col: str = 'amt_quantile',
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
        Compute amount quantile using pandas.

        :param x: DataFrame with 'code', 'volume', and 'count' columns.
        :returns: pd.Series with amount quantile values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("AmountQuantileTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'volume' not in x.columns:
            raise ValueError("AmountQuantileTransform requires 'volume' column.")
        if 'count' not in x.columns:
            raise ValueError("AmountQuantileTransform requires 'count' column.")

        def _amt_quantile(x_inner: pd.DataFrame) -> float:
            """Amount quantile for one resample window."""
            price_per_unit = x_inner['volume'] / x_inner['count']
            price_per_unit = price_per_unit.replace([np.inf, -np.inf], np.nan).dropna()
            price_per_unit = price_per_unit.sort_values()
            # Drop last 2
            if len(price_per_unit) > 2:
                price_per_unit = price_per_unit[:-2]
            if len(price_per_unit) < 2:
                return 0.0
            pmin = price_per_unit.min()
            pmax = price_per_unit.max()
            if pmax == pmin:
                return 0.0
            qua = (price_per_unit.quantile(0.1) - pmin) / (pmax - pmin)
            return float(qua) if not pd.isna(qua) else 0.0

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _amt_quantile(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.
        """
        return self._pd(x)


class TradingIntensityTransform(SISOTransform):
    """
    QIML1113: Trading intensity — Spearman correlation of price-per-unit vs volume.

    Computes price-per-unit (volume / amount) and measures its Spearman rank
    correlation with volume. Higher values indicate that larger trades tend
    to have different price-per-unit characteristics.

    The transform uses ``groupby('code').resample(frequency)`` semantics
    to compute the correlation per instrument per time bucket.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'volume').
    :param output_col: Output column suffix (default 'trading_intensity').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 120,
        ...     'amount': np.random.rand(120) * 100000,
        ...     'volume': np.random.rand(120) * 1000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=120, freq='min')
        >>> transform = TradingIntensityTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'volume',
        output_col: str = 'trading_intensity',
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
        Compute trading intensity using pandas.

        :param x: DataFrame with 'code', 'volume', and 'amount' columns.
        :returns: pd.Series with trading intensity values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("TradingIntensityTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'volume' not in x.columns:
            raise ValueError("TradingIntensityTransform requires 'volume' column.")
        if 'amount' not in x.columns:
            raise ValueError("TradingIntensityTransform requires 'amount' column.")

        def _trading_intensity(x_inner: pd.DataFrame) -> float:
            """Trading intensity for one resample window."""
            price_per_unit = x_inner['volume'] / x_inner['amount'].replace(0, np.nan)
            corr = price_per_unit.corr(x_inner['volume'], method='spearman')
            return float(corr) if not pd.isna(corr) else 0.0

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _trading_intensity(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.
        """
        return self._pd(x)


class TailVolumeRatioVTransform(SISOTransform):
    """
    QIML1215: Tail volume ratio (value) — amount share in last 5 minutes of the hour.

    Measures the proportion of total amount that occurs in the last 5 minutes
    (minute >= 55) within each resample window. Higher values indicate
    concentration of value near the close.

    The transform uses ``groupby('code').resample(frequency)`` semantics
    to compute the ratio per instrument per time bucket.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'tail_vol_ratio_v').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 120,
        ...     'amount': np.random.rand(120) * 100000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=120, freq='min')
        >>> transform = TailVolumeRatioVTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'tail_vol_ratio_v',
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
        Compute tail volume ratio (value) using pandas.

        :param x: DataFrame with 'code' and 'amount' columns.
        :returns: pd.Series with tail volume ratio values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("TailVolumeRatioVTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'amount' not in x.columns:
            raise ValueError("TailVolumeRatioVTransform requires 'amount' column.")

        def _tail_ratio_v(x_inner: pd.DataFrame) -> float:
            """Tail volume ratio (value) for one resample window."""
            minutes = pd.DatetimeIndex(x_inner.index).minute  # type: ignore[union-abstract]
            total_amt = x_inner['amount'].sum()
            if total_amt == 0:
                return 0.0
            late_mask = minutes >= 55
            ratio = x_inner.loc[late_mask, 'amount'].sum() / total_amt
            return ratio if not pd.isna(ratio) else 0.0

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _tail_ratio_v(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.
        """
        return self._pd(x)


class TVDTransform(SISOTransform):
    """
    QIML1222: Total volume distribution — std of normalized amounts.

    Normalizes amounts by their sum within the window, then takes the
    standard deviation. Measures the dispersion of trading value across
    the resample window. Higher values indicate more heterogeneous
    distribution; lower values indicate more uniform distribution.

    The transform uses ``groupby('code').resample(frequency)`` semantics
    to compute the std per instrument per time bucket.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'tvd').

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'code': ['A'] * 120,
        ...     'amount': np.random.rand(120) * 100000,
        ... })
        >>> df.index = pd.date_range('2024-01-01', periods=120, freq='min')
        >>> transform = TVDTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'tvd',
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
        Compute TVD using pandas.

        :param x: DataFrame with 'code' and 'amount' columns.
        :returns: pd.Series with TVD values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("TVDTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'amount' not in x.columns:
            raise ValueError("TVDTransform requires 'amount' column.")

        def _tvd(x_inner: pd.DataFrame) -> float:
            """TVD for one resample window."""
            total = x_inner['amount'].sum()
            if total == 0:
                return 0.0
            normalized = x_inner['amount'] / total
            std_val = normalized.std()
            return float(std_val) if not pd.isna(std_val) else 0.0

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _tvd(group)
            for idx in group.index:
                records.append({'idx': idx, 'code': code, 'bucket': ts, 'val': val})

        result_df = pd.DataFrame(records).set_index('idx')
        arr: np.ndarray = result_df['val'].values  # type: ignore[assignment]
        return pd.Series(arr, index=original_index, name=self.output_name)

    def _nb(self, x: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        Numba fallback: delegate to pandas implementation.
        """
        return self._pd(x)

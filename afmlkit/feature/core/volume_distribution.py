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


class PShapeDiffTransform(SISOTransform):
    """
    QIML0503: Volume profile P-shape — close relative to low price level.

    Computes the volume-weighted price level that accumulates 50% of total
    volume (starting from the peak volume price), then returns the normalized
    distance from that low price level to the current close price.

    :param frequency: Resampling frequency (e.g. '5min', '15min', 'H', 'D').
    :param input_col: Input column name (default 'amount').
    :param output_col: Output column suffix (default 'vol_p_shape_diff').

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
        >>> transform = PShapeDiffTransform(frequency='H')
        >>> result = transform(df)
    """

    def __init__(
        self,
        frequency: str = 'H',
        input_col: str = 'amount',
        output_col: str = 'vol_p_shape_diff',
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
        Compute P-shape diff factor using pandas.

        :param x: DataFrame with 'code', 'close', and 'amount' columns.
        :returns: pd.Series with P-shape diff values, index aligned to input DataFrame.
        """
        if isinstance(x, pd.Series):
            raise ValueError("PShapeDiffTransform requires a DataFrame with 'code' column.")
        if 'code' not in x.columns:
            raise ValueError("Input DataFrame must contain 'code' column for grouping.")
        if 'close' not in x.columns:
            raise ValueError("PShapeDiffTransform requires 'close' column.")

        def _p_shape_diff(x_inner: pd.DataFrame) -> float:
            """P-shape diff value for one resample window."""
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
            return (x_inner['close'] - vsa_low).mean() / vsa_low

        original_index = x.index
        records = []
        ts_index = pd.DatetimeIndex(x.index).floor(self.frequency)  # type: ignore[union-abstract]
        for (code, ts), group in x.groupby([x['code'], ts_index]):  # type: ignore[union-abstract]
            val = _p_shape_diff(group)
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

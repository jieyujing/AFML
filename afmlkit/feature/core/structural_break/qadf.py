"""
QADF (Quantile ADF) test - rolling quantile of SADF for noise reduction.

Reference: AFML Chapter 17
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Union, Optional
import pandas as pd

from afmlkit.feature.base import MISOTransform


@njit(nogil=True)
def _rolling_quantile_core(
    x: NDArray[np.float64],
    window: int,
    quantile: float
) -> NDArray[np.float64]:
    """
    Rolling quantile calculation using simple sort.

    :param x: Input series
    :param window: Rolling window size
    :param quantile: Quantile level (0-1)
    :returns: Rolling quantile values
    """
    n = len(x)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(window - 1, n):
        window_data = x[i - window + 1:i + 1]
        valid = window_data[~np.isnan(window_data)]
        if len(valid) > 0:
            result[i] = np.percentile(valid, quantile * 100)

    return result


def qadf_test(
    sadf_values: Union[pd.Series, NDArray[np.float64]],
    window: int = 20,
    quantile: float = 0.95
) -> Union[pd.Series, NDArray[np.float64]]:
    """
    QADF test - rolling quantile of SADF values.

    :param sadf_values: SADF statistic series
    :param window: Rolling window for quantile (default 20)
    :param quantile: Quantile level (default 0.95)
    :returns: QADF series
    """
    is_pandas = isinstance(sadf_values, pd.Series)
    index = sadf_values.index if is_pandas else None

    arr = np.asarray(sadf_values, dtype=np.float64)
    result = _rolling_quantile_core(arr, window, quantile)

    if is_pandas:
        return pd.Series(result, index=index, name='qadf')
    return result


class QADFTest(MISOTransform):
    """
    QADF Transform for FeatureKit pipeline.

    Computes rolling quantile of SADF values for smoother bubble detection.

    **Important**: This transform requires a pre-computed SADF column. The input
    DataFrame must contain a 'sadf' column (or the column specified via input_cols).

    Typical pipeline: SADF -> QADF -> CADF

    Example:
        >>> # First compute SADF
        >>> sadf = SADFTest(input_col='close')
        >>> df['sadf'] = sadf.fit_transform(df)
        >>> # Then compute QADF from SADF
        >>> qadf = QADFTest(window=20, quantile=0.95)
        >>> df['qadf'] = qadf.fit_transform(df)

    :param window: Rolling window for quantile (default 20)
    :param quantile: Quantile level (default 0.95)
    """

    def __init__(
        self,
        window: int = 20,
        quantile: float = 0.95,
        input_cols: Optional[list] = None
    ):
        inputs = input_cols if input_cols else ['sadf']
        super().__init__(inputs, 'qadf')
        self.window = window
        self.quantile = quantile

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        sadf = x[self.requires[0]].values
        result = _rolling_quantile_core(sadf, self.window, self.quantile)
        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)

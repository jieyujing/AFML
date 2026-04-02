"""
Theil Index Imbalance Factor — 买卖集中度不对称性因子

核心思路：
    用 Theil index (KL divergence from uniform) 分别度量买方/卖方成交量
    在时间 bar 上的集中度，二者的比率反映 "谁在集中执行"。

    机构倾向于在特定窗口集中下单 → T_buy 高；散户更均匀。

    当没有真实 buy_volume 时，用 bar 内价格结构做连续分拆：
    - CLV (Chaikin): close 在 high-low 中的位置，无参数
    - BVC (Easley, López de Prado & O'Hara): 标准化 price change 的正态 CDF

References
----------
    Easley, D., López de Prado, M. M., & O'Hara, M. (2012).
    The volume clock: Insights into the high-frequency paradigm.
    Journal of Portfolio Management, 39(1), 19-29.
"""
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from typing import Tuple
import pandas as pd

from afmlkit.feature.base import MISOTransform


# ---------------------------------------------------------------------------
# Numba Backend - Volume Split Functions
# ---------------------------------------------------------------------------

@njit(nogil=True)
def clv_split(
    close: NDArray[np.float64],
    low: NDArray[np.float64],
    high: NDArray[np.float64],
    volume: NDArray[np.float64],
    eps: float = 1e-10
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    CLV (Close Location Value) 连续分拆，仅需 OHLCV。

    CLV = [(close - low) - (high - close)] / (high - low) ∈ [-1, 1]

    解释:
        close 贴近 high → CLV → +1 → 买方主导
        close 贴近 low  → CLV → -1 → 卖方主导

    buy_ratio = (1 + CLV) / 2 ∈ [0, 1]

    :param close: 收盘价数组
    :param low: 最低价数组
    :param high: 最高价数组
    :param volume: 成交量数组
    :param eps: 防止除零的小数值
    :returns: (buy_volume, sell_volume) 买方和卖方成交量数组
    """
    n = len(close)
    buy_vol = np.empty(n, dtype=np.float64)
    sell_vol = np.empty(n, dtype=np.float64)

    for i in range(n):
        hl_range = high[i] - low[i]

        if hl_range > eps:
            clv = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
        else:
            clv = 0.0

        buy_ratio = (1.0 + clv) / 2.0
        buy_vol[i] = volume[i] * buy_ratio
        sell_vol[i] = volume[i] * (1.0 - buy_ratio)

    return buy_vol, sell_vol


@njit(nogil=True)
def bvc_split(
    close: NDArray[np.float64],
    volume: NDArray[np.float64],
    sigma_window: int = 20,
    eps: float = 1e-10
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    BVC (Bulk Volume Classification) 概率分拆。

    来源: Easley, López de Prado & O'Hara (2012)

    核心思想:
        buy_ratio = Φ(Δp / σ)
        其中 Δp 是价格变化，σ 是波动率（用 rolling return std）

    :param close: 收盘价数组
    :param volume: 成交量数组
    :param sigma_window: 计算波动率的滚动窗口大小
    :param eps: 防止除零的小数值
    :returns: (buy_volume, sell_volume) 买方和卖方成交量数组
    """
    n = len(close)
    buy_vol = np.empty(n, dtype=np.float64)
    sell_vol = np.empty(n, dtype=np.float64)

    # 计算对数收益率
    log_close = np.log(np.maximum(close, eps))
    ret = np.empty(n, dtype=np.float64)
    ret[0] = 0.0
    for i in range(1, n):
        ret[i] = log_close[i] - log_close[i - 1]

    # 计算滚动波动率
    sigma = np.empty(n, dtype=np.float64)
    sigma[:sigma_window] = eps
    for i in range(sigma_window, n):
        sum_ret = 0.0
        sum_ret2 = 0.0
        for j in range(i - sigma_window + 1, i + 1):
            sum_ret += ret[j]
            sum_ret2 += ret[j] * ret[j]
        mean_ret = sum_ret / sigma_window
        var_ret = sum_ret2 / sigma_window - mean_ret * mean_ret
        sigma[i] = max(np.sqrt(max(var_ret, 0.0)), eps)

    # 前 sigma_window 用全局 std
    if n > sigma_window:
        global_sum = 0.0
        global_sum2 = 0.0
        count = 0
        for i in range(sigma_window, n):
            global_sum += ret[i]
            global_sum2 += ret[i] * ret[i]
            count += 1
        if count > 0:
            global_mean = global_sum / count
            global_var = global_sum2 / count - global_mean * global_mean
            global_std = max(np.sqrt(max(global_var, 0.0)), eps)
            for i in range(sigma_window):
                sigma[i] = global_std

    # 标准化价格变化并使用正态 CDF 近似
    for i in range(n):
        z = ret[i] / sigma[i] if sigma[i] > eps else 0.0
        # 正态 CDF 近似 (Abramowitz and Stegun)
        # Φ(z) ≈ 0.5 * (1 + sign(z) * (1 - exp(-2*z^2/π)))
        if z >= 0:
            buy_ratio = 0.5 * (1.0 + (1.0 - np.exp(-2.0 * z * z / np.pi)))
        else:
            buy_ratio = 0.5 * (1.0 - (1.0 - np.exp(-2.0 * z * z / np.pi)))

        buy_vol[i] = volume[i] * buy_ratio
        sell_vol[i] = volume[i] * (1.0 - buy_ratio)

    return buy_vol, sell_vol


@njit(nogil=True)
def direction_split(
    close: NDArray[np.float64],
    open_: NDArray[np.float64],
    volume: NDArray[np.float64],
    eps: float = 1e-10
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    最简单的方向分类法：bar 涨 → volume 归 buy，bar 跌 → volume 归 sell。

    解释:
        这是一个粗糙的方法，但作为 baseline 有参考价值。
        仅根据涨跌方向判断，不考虑幅度。

    :param close: 收盘价数组
    :param open_: 开盘价数组
    :param volume: 成交量数组
    :param eps: 防止除零的小数值
    :returns: (buy_volume, sell_volume) 买方和卖方成交量数组
    """
    n = len(close)
    buy_vol = np.empty(n, dtype=np.float64)
    sell_vol = np.empty(n, dtype=np.float64)

    for i in range(n):
        if close[i] >= open_[i]:
            buy_vol[i] = volume[i]
            sell_vol[i] = eps
        else:
            buy_vol[i] = eps
            sell_vol[i] = volume[i]

    return buy_vol, sell_vol


# ---------------------------------------------------------------------------
# Numba Backend - Theil Index Calculation
# ---------------------------------------------------------------------------

@njit(nogil=True)
def theil_index(shares: NDArray[np.float64], total: float, eps: float = 1e-10) -> float:
    """
    计算单个窗口的 Theil Index (KL divergence from uniform)。

    Theil Index 公式:
        T = Σ p_i * log(p_i / q_i)
        其中 p_i 是实际分布，q_i 是均匀分布 (1/n)

    简化后:
        T = (1/n) * Σ p_i * log(p_i * n)
        其中 p_i = shares_i / total

    特殊情况:
        T = 0: 完全均匀（每个 bar 成交量相等）
        T = log(n): 完全集中（所有成交量在一个 bar）

    :param shares: 成交量份额数组
    :param total: 总成交量
    :param eps: 防止除零的小数值
    :returns: Theil Index 值 ∈ [0, log(n)]
    """
    n = len(shares)
    if n < 2 or total < eps:
        return np.nan

    theil = 0.0
    for i in range(n):
        if shares[i] > eps:
            p_i = shares[i] / total
            theil += p_i * np.log(p_i * n)

    return theil


@njit(nogil=True, parallel=True)
def rolling_theil_imbalance(
    buy_volume: NDArray[np.float64],
    sell_volume: NDArray[np.float64],
    window: int,
    eps: float = 1e-10
) -> NDArray[np.float64]:
    """
    计算滚动 Theil Imbalance Factor。

    T_buy = Theil index of buy volume distribution across bars
    T_sell = Theil index of sell volume distribution across bars

    T_ratio = T_buy / (T_buy + T_sell)

    解释:
        T_ratio > 0.5: 买方更集中 (机构买入特征)
        T_ratio < 0.5: 卖方更集中 (机构卖出特征)
        T_ratio = 0.5: 对称

    :param buy_volume: 买方成交量数组
    :param sell_volume: 卖方成交量数组
    :param window: 滚动窗口大小
    :param eps: 防止除零的小数值
    :returns: T_ratio 序列 ∈ [0, 1]
    """
    n = len(buy_volume)
    result = np.empty(n, dtype=np.float64)
    result[:] = np.nan

    if n < window:
        return result

    for t in prange(window - 1, n):
        # 提取窗口数据
        buy_window = buy_volume[t - window + 1: t + 1]
        sell_window = sell_volume[t - window + 1: t + 1]

        # 计算总成交量
        buy_total = 0.0
        sell_total = 0.0
        for i in range(window):
            buy_total += buy_window[i]
            sell_total += sell_window[i]

        # 计算 Theil Index
        t_buy = theil_index(buy_window, buy_total, eps)
        t_sell = theil_index(sell_window, sell_total, eps)

        # 计算 T_ratio
        denom = t_buy + t_sell
        if denom > eps:
            result[t] = t_buy / denom
        else:
            result[t] = 0.5  # 对称情况

    return result


@njit(nogil=True, parallel=True)
def rolling_theil_decomposed(
    buy_volume: NDArray[np.float64],
    sell_volume: NDArray[np.float64],
    window: int,
    eps: float = 1e-10
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    计算滚动 Theil Imbalance 完整分解。

    返回:
        T_buy — 买方集中度 (KL from uniform)
        T_sell — 卖方集中度
        T_ratio — T_buy / (T_buy + T_sell)
        T_diff — T_buy - T_sell (原始差值)

    :param buy_volume: 买方成交量数组
    :param sell_volume: 卖方成交量数组
    :param window: 滚动窗口大小
    :param eps: 防止除零的小数值
    :returns: (T_buy, T_sell, T_ratio, T_diff) 四个数组
    """
    n = len(buy_volume)
    t_buy_arr = np.empty(n, dtype=np.float64)
    t_sell_arr = np.empty(n, dtype=np.float64)
    t_ratio_arr = np.empty(n, dtype=np.float64)
    t_diff_arr = np.empty(n, dtype=np.float64)

    t_buy_arr[:] = np.nan
    t_sell_arr[:] = np.nan
    t_ratio_arr[:] = np.nan
    t_diff_arr[:] = np.nan

    if n < window:
        return t_buy_arr, t_sell_arr, t_ratio_arr, t_diff_arr

    for t in prange(window - 1, n):
        # 提取窗口数据
        buy_window = buy_volume[t - window + 1: t + 1]
        sell_window = sell_volume[t - window + 1: t + 1]

        # 计算总成交量
        buy_total = 0.0
        sell_total = 0.0
        for i in range(window):
            buy_total += buy_window[i]
            sell_total += sell_window[i]

        # 计算 Theil Index
        t_buy = theil_index(buy_window, buy_total, eps)
        t_sell = theil_index(sell_window, sell_total, eps)

        t_buy_arr[t] = t_buy
        t_sell_arr[t] = t_sell

        # 计算 T_ratio 和 T_diff
        denom = t_buy + t_sell
        if denom > eps:
            t_ratio_arr[t] = t_buy / denom
        else:
            t_ratio_arr[t] = 0.5

        t_diff_arr[t] = t_buy - t_sell

    return t_buy_arr, t_sell_arr, t_ratio_arr, t_diff_arr


# ---------------------------------------------------------------------------
# Transform Classes
# ---------------------------------------------------------------------------

class TheilImbalanceTransform(MISOTransform):
    """
    Theil Imbalance Factor Transform.

    计算买卖集中度不对称性因子，使用 Theil Index (KL divergence from uniform)。

    Parameters
    ----------
    ohlcv_cols : list[str]
        列名列表 [open, high, low, close, volume]
    window : int
        滚动窗口大小
    split_method : str
        成交量分拆方法: 'clv' (默认), 'bvc', 'direction'
    output_col : str, optional
        输出列名后缀，默认 'theil_imb_{window}'

    Examples
    --------
    >>> from afmlkit.feature.core.theil_imbalance import TheilImbalanceTransform
    >>> transform = TheilImbalanceTransform(
    ...     ohlcv_cols=['open', 'high', 'low', 'close', 'volume'],
    ...     window=20,
    ...     split_method='clv'
    ... )
    >>> t_ratio = transform(bars, backend='nb')
    """

    def __init__(
        self,
        ohlcv_cols: list,
        window: int,
        split_method: str = 'clv',
        output_col: str = None
    ):
        self.window = window
        self.split_method = split_method
        if output_col is None:
            output_col = f'theil_imb_{window}'
        super().__init__(ohlcv_cols, output_col)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        """验证输入列存在。"""
        for col in self.requires:
            if col not in x.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        return True

    def _prepare_input_nb(self, x: pd.DataFrame) -> dict:
        """准备 Numba 输入。"""
        return {
            'open': x[self.requires[0]].values.astype(np.float64),
            'high': x[self.requires[1]].values.astype(np.float64),
            'low': x[self.requires[2]].values.astype(np.float64),
            'close': x[self.requires[3]].values.astype(np.float64),
            'volume': x[self.requires[4]].values.astype(np.float64),
        }

    def _prepare_output_nb(self, idx: pd.Index, y: NDArray) -> pd.Series:
        """准备输出 Series。"""
        return pd.Series(y, index=idx, name=self.output_name)

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        """Pandas 实现。"""
        inputs = self._prepare_input_nb(x)
        result = self._nb(x)
        return result

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        """Numba 实现。"""
        inputs = self._prepare_input_nb(x)

        # 分拆成交量
        if self.split_method == 'clv':
            buy_vol, sell_vol = clv_split(
                inputs['close'], inputs['low'], inputs['high'], inputs['volume']
            )
        elif self.split_method == 'bvc':
            buy_vol, sell_vol = bvc_split(inputs['close'], inputs['volume'])
        elif self.split_method == 'direction':
            buy_vol, sell_vol = direction_split(
                inputs['close'], inputs['open'], inputs['volume']
            )
        else:
            raise ValueError(f"Unknown split_method: {self.split_method}")

        # 计算 Theil Imbalance
        result = rolling_theil_imbalance(buy_vol, sell_vol, self.window)

        return self._prepare_output_nb(x.index, result)

    @property
    def output_name(self) -> str:
        """输出列名。"""
        return self.produces[0]


class TheilDecomposedTransform(MISOTransform):
    """
    Theil Imbalance 完整分解 Transform。

    返回 T_buy, T_sell, T_ratio, T_diff 四个特征。

    Parameters
    ----------
    ohlcv_cols : list[str]
        列名列表 [open, high, low, close, volume]
    window : int
        滚动窗口大小
    split_method : str
        成交量分拆方法: 'clv' (默认), 'bvc', 'direction'
    """

    def __init__(
        self,
        ohlcv_cols: list,
        window: int,
        split_method: str = 'clv'
    ):
        self.window = window
        self.split_method = split_method
        output_cols = [
            f'theil_buy_{window}',
            f'theil_sell_{window}',
            f'theil_ratio_{window}',
            f'theil_diff_{window}'
        ]
        super().__init__(ohlcv_cols, output_cols)

    def _validate_input(self, x: pd.DataFrame) -> bool:
        """验证输入列存在。"""
        for col in self.requires:
            if col not in x.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        return True

    def _prepare_input_nb(self, x: pd.DataFrame) -> dict:
        """准备 Numba 输入。"""
        return {
            'open': x[self.requires[0]].values.astype(np.float64),
            'high': x[self.requires[1]].values.astype(np.float64),
            'low': x[self.requires[2]].values.astype(np.float64),
            'close': x[self.requires[3]].values.astype(np.float64),
            'volume': x[self.requires[4]].values.astype(np.float64),
        }

    def _prepare_output_nb(self, idx: pd.Index, y: tuple) -> tuple:
        """准备输出 Series。"""
        return tuple(
            pd.Series(arr, index=idx, name=name)
            for arr, name in zip(y, self.produces)
        )

    def _pd(self, x: pd.DataFrame) -> tuple:
        """Pandas 实现。"""
        return self._nb(x)

    def _nb(self, x: pd.DataFrame) -> tuple:
        """Numba 实现。"""
        inputs = self._prepare_input_nb(x)

        # 分拆成交量
        if self.split_method == 'clv':
            buy_vol, sell_vol = clv_split(
                inputs['close'], inputs['low'], inputs['high'], inputs['volume']
            )
        elif self.split_method == 'bvc':
            buy_vol, sell_vol = bvc_split(inputs['close'], inputs['volume'])
        elif self.split_method == 'direction':
            buy_vol, sell_vol = direction_split(
                inputs['close'], inputs['open'], inputs['volume']
            )
        else:
            raise ValueError(f"Unknown split_method: {self.split_method}")

        # 计算 Theil 分解
        t_buy, t_sell, t_ratio, t_diff = rolling_theil_decomposed(
            buy_vol, sell_vol, self.window
        )

        return self._prepare_output_nb(x.index, (t_buy, t_sell, t_ratio, t_diff))

    @property
    def output_name(self) -> list:
        """输出列名列表。"""
        return self.produces


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def theil_imbalance(
    bars: pd.DataFrame,
    window: int,
    split_method: str = 'clv',
    decomposed: bool = False
) -> pd.DataFrame:
    """
    计算 Theil Imbalance 因子（便捷函数）。

    :param bars: OHLCV DataFrame
    :param window: 滚动窗口大小
    :param split_method: 分拆方法 'clv', 'bvc', 'direction'
    :param decomposed: True 返回完整分解，False 仅返回 T_ratio
    :returns: 特征 DataFrame
    """
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']

    if decomposed:
        transform = TheilDecomposedTransform(ohlcv_cols, window, split_method)
        t_buy, t_sell, t_ratio, t_diff = transform(bars, backend='nb')
        return pd.DataFrame({
            f'feat_theil_buy_{window}': t_buy,
            f'feat_theil_sell_{window}': t_sell,
            f'feat_theil_ratio_{window}': t_ratio,
            f'feat_theil_diff_{window}': t_diff,
        })
    else:
        transform = TheilImbalanceTransform(ohlcv_cols, window, split_method)
        t_ratio = transform(bars, backend='nb')
        return pd.DataFrame({
            f'feat_theil_imb_{window}': t_ratio,
        })

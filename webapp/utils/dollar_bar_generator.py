"""Dollar Bar 生成器 - 基于动态阈值的流式处理"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Callable, List
from numba import njit
from numba.typed import List as NumbaList
from pathlib import Path


@njit(nogil=True)
def _stream_dynamic_dollar_bar_indexer(
    prices: np.ndarray,
    volumes: np.ndarray,
    thresholds: np.ndarray,
    carry_over: float,
) -> Tuple[NumbaList, float]:
    """
    动态阈值 Dollar Bar indexer，支持跨分块 carry-over。

    :param prices: 当前月 tick 价格
    :param volumes: 当前月 tick 成交量
    :param thresholds: 当前月每 tick 对应的阈值
    :param carry_over: 上月残余累积金额
    :returns: (close_indices_list, new_carry_over)
    """
    n = len(prices)
    indices = NumbaList()

    cum_dollar = carry_over
    for i in range(n):
        cum_dollar += prices[i] * volumes[i]
        if cum_dollar >= thresholds[i]:
            indices.append(i)
            cum_dollar = cum_dollar - thresholds[i]

    return indices, cum_dollar


def _build_ohlcv_from_indices(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    close_indices: np.ndarray,
) -> pd.DataFrame:
    """
    从 close_indices 构建 OHLCV DataFrame

    :param timestamps: 时间戳数组（纳秒）
    :param prices: 价格数组
    :param volumes: 成交量数组
    :param close_indices: close 索引数组
    :return: OHLCV DataFrame
    """
    if len(close_indices) < 2:
        return pd.DataFrame()

    # 使用 afmlkit 的基础函数
    try:
        from afmlkit.bar.base import comp_bar_ohlcv
        from afmlkit.bar.logic import _get_bar_ohlcv

        # 计算 OHLCV
        ohlcv_tuple = _get_bar_ohlcv(prices, volumes, close_indices)

        # close 时间戳
        close_ts = timestamps[close_indices[1:]]

        ohlcv_df = pd.DataFrame({
            "open": ohlcv_tuple[0],
            "high": ohlcv_tuple[1],
            "low": ohlcv_tuple[2],
            "close": ohlcv_tuple[3],
            "volume": ohlcv_tuple[4],
            "vwap": ohlcv_tuple[5],
            "trades": ohlcv_tuple[6],
        })

        ohlcv_df.index = pd.to_datetime(close_ts, unit="ns")
        ohlcv_df.index.name = "timestamp"

    except ImportError:
        # 简化实现
        ohlcv_df = _simple_ohlcv_from_indices(prices, volumes, close_indices)
        close_ts = timestamps[close_indices[1:]]
        ohlcv_df.index = pd.to_datetime(close_ts, unit="ns")
        ohlcv_df.index.name = "timestamp"

    return ohlcv_df


def _simple_ohlcv_from_indices(
    prices: np.ndarray,
    volumes: np.ndarray,
    close_indices: np.ndarray,
) -> pd.DataFrame:
    """
    简化版 OHLCV 计算（不依赖 afmlkit）

    :param prices: 价格数组
    :param volumes: 成交量数组
    :param close_indices: close 索引数组
    :return: OHLCV DataFrame
    """
    n_bars = len(close_indices) - 1

    open_prices = np.zeros(n_bars)
    high_prices = np.zeros(n_bars)
    low_prices = np.zeros(n_bars)
    close_prices = np.zeros(n_bars)
    bar_volumes = np.zeros(n_bars)

    for i in range(n_bars):
        start_idx = close_indices[i] if i == 0 else close_indices[i] + 1
        end_idx = close_indices[i + 1] + 1

        bar_prices = prices[start_idx:end_idx]
        bar_vols = volumes[start_idx:end_idx]

        open_prices[i] = bar_prices[0]
        high_prices[i] = np.max(bar_prices)
        low_prices[i] = np.min(bar_prices)
        close_prices[i] = bar_prices[-1]
        bar_volumes[i] = np.sum(bar_vols)

    return pd.DataFrame({
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": bar_volumes,
    })


class DollarBarGenerator:
    """
    Dollar Bar 流式生成器

    基于 Marcos López de Prado 的动态 Dollar Bar 理论，
    根据每日目标 bar 数量自适应调整阈值。
    """

    DEFAULT_TARGET_BARS = [4, 6, 10, 20, 50]
    DEFAULT_EWMA_SPAN = 20

    def __init__(
        self,
        target_daily_bars: Optional[List[int]] = None,
        ewma_span: int = DEFAULT_EWMA_SPAN,
    ):
        """
        初始化 Dollar Bar 生成器

        :param target_daily_bars: 每日目标 bar 数量列表
        :param ewma_span: EWMA 平滑窗口
        """
        self.target_daily_bars = target_daily_bars or self.DEFAULT_TARGET_BARS
        self.ewma_span = ewma_span

    def scan_daily_dollar_volume(
        self,
        df: pd.DataFrame,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> pd.Series:
        """
        阶段 1: 扫描计算每日 dollar volume

        :param df: OHLCV DataFrame（索引为 datetime）
        :param progress_callback: 进度回调函数
        :return: 每日 dollar volume Series
        """
        if progress_callback:
            progress_callback(0.1)

        # 使用 close 价格和 volume 计算 dollar volume
        if 'close' not in df.columns or 'volume' not in df.columns:
            raise ValueError("DataFrame 必须包含 'close' 和 'volume' 列")

        # 计算每行的 dollar volume
        dollar_vol = df['close'].values * df['volume'].values

        # 创建时间索引
        if isinstance(df.index, pd.DatetimeIndex):
            dt_index = df.index
        else:
            dt_index = pd.to_datetime(df.index)

        # 按日聚合
        daily_series = pd.Series(dollar_vol, index=dt_index).resample("D").sum()

        if progress_callback:
            progress_callback(0.3)

        # 处理零值
        daily_series = daily_series.replace(0, np.nan).ffill()

        if progress_callback:
            progress_callback(0.5)

        return daily_series

    def compute_ewma_thresholds(
        self,
        daily_dollar_vol: pd.Series
    ) -> Dict[int, pd.Series]:
        """
        计算 EWMA 阈值

        :param daily_dollar_vol: 每日 dollar volume
        :return: 各频率的阈值字典 {freq: threshold_series}
        """
        # 计算 EWMA
        daily_ewma = daily_dollar_vol.ewm(span=self.ewma_span, min_periods=1).mean()

        # 为每个频率计算阈值
        thresholds = {}
        for freq in self.target_daily_bars:
            thresholds[freq] = daily_ewma / freq

        return thresholds

    def _map_thresholds_to_ticks(
        self,
        tick_dates: pd.DatetimeIndex,
        daily_thresholds: pd.Series
    ) -> np.ndarray:
        """
        将日度阈值映射到每个 tick

        :param tick_dates: tick 日期索引
        :param daily_thresholds: 日度阈值
        :return: 每个 tick 对应的阈值数组
        """
        # 获取 tick 对应的日期（归一化到午夜）
        tick_dates_int = tick_dates.normalize().view(np.int64)

        # 获取阈值日期
        thrs_dates_int = daily_thresholds.index.view(np.int64)
        thrs_values = daily_thresholds.values.astype(np.float64)

        # 使用 searchsorted 映射
        idx = np.searchsorted(thrs_dates_int, tick_dates_int, side="right") - 1
        idx = np.clip(idx, 0, len(thrs_values) - 1)

        return thrs_values[idx]

    def generate_bars(
        self,
        df: pd.DataFrame,
        thresholds: Dict[int, pd.Series],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[int, pd.DataFrame]:
        """
        生成 Dollar Bars

        :param df: OHLCV DataFrame
        :param thresholds: 各频率的阈值字典
        :param progress_callback: 进度回调函数
        :return: 各频率的 bars 字典 {freq: bars_df}
        """
        # 准备数据
        timestamps = df.index.astype(np.int64).values.astype(np.int64)
        prices = df['close'].values.astype(np.float64)
        volumes = df['volume'].values.astype(np.float64)

        # 为每个 tick 映射日期
        tick_dates = df.index

        all_bars = {}
        total_freqs = len(self.target_daily_bars)

        for i, freq in enumerate(self.target_daily_bars):
            if progress_callback:
                progress_callback(0.5 + (i / total_freqs) * 0.4)

            # 获取该频率的阈值
            daily_thrs = thresholds[freq]

            # 映射到 tick 级别
            tick_thresholds = self._map_thresholds_to_ticks(tick_dates, daily_thrs)

            # 生成 bars
            carry_over = 0.0
            indices_list, new_carry = _stream_dynamic_dollar_bar_indexer(
                prices, volumes, tick_thresholds, carry_over
            )

            close_indices = np.array(indices_list, dtype=np.int64)

            # 构建 OHLCV DataFrame
            if len(close_indices) >= 2:
                bars_df = _build_ohlcv_from_indices(
                    timestamps, prices, volumes, close_indices
                )
                all_bars[freq] = bars_df
            else:
                all_bars[freq] = pd.DataFrame()

        if progress_callback:
            progress_callback(1.0)

        return all_bars

    def save_bars(
        self,
        bars_dict: Dict[int, pd.DataFrame],
        output_dir: str = "outputs/dollar_bars"
    ) -> Dict[int, Path]:
        """
        保存各频率的 bars 到 CSV

        :param bars_dict: bars 字典
        :param output_dir: 输出目录
        :return: 保存的文件路径字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        for freq, bars_df in bars_dict.items():
            if len(bars_df) == 0:
                continue

            filepath = output_path / f"dollar_bars_freq{freq}.csv"
            bars_df.to_csv(filepath)
            saved_files[freq] = filepath

        return saved_files

    def run(
        self,
        df: pd.DataFrame,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[int, pd.DataFrame]:
        """
        完整运行 Dollar Bar 生成流程

        :param df: OHLCV DataFrame
        :param progress_callback: 进度回调函数
        :return: 各频率的 bars 字典
        """
        # 阶段 1: 计算日度 dollar volume
        daily_vol = self.scan_daily_dollar_volume(df, progress_callback)

        # 计算阈值
        thresholds = self.compute_ewma_thresholds(daily_vol)

        # 阶段 2: 生成 bars
        bars_dict = self.generate_bars(df, thresholds, progress_callback)

        return bars_dict


def generate_dollar_bars(
    df: pd.DataFrame,
    target_daily_bars: Optional[List[int]] = None,
    ewma_span: int = 20,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Dict[int, pd.DataFrame]:
    """
    便捷函数：生成 Dollar Bars

    :param df: OHLCV DataFrame
    :param target_daily_bars: 每日目标 bar 数量
    :param ewma_span: EWMA 跨度
    :param progress_callback: 进度回调
    :return: 各频率的 bars 字典
    """
    generator = DollarBarGenerator(target_daily_bars, ewma_span)
    return generator.run(df, progress_callback)

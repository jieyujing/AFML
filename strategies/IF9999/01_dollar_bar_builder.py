"""
01_dollar_bar_builder.py - IF9999 Dollar Bars 构建与验证

流程:
1. 加载 1 分钟 OHLCV 数据
2. 转换为 tick-like 格式（每分钟作为一个采样点）
3. 计算动态阈值（EWMA 日均美元交易量 / 目标 Bar 数）
4. 构建 Dollar Bars
5. 三刀验证（独立性、同分布、正态性）
6. 可视化输出
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import (
    DATA_PATH, CONTRACT_MULTIPLIER, TARGET_DAILY_BARS, EWMA_SPAN,
    ACF_LAGS, BARS_DIR, FIGURES_DIR
)

from numba import njit
from numba.typed import List as NumbaList


def load_if_data(data_path: str) -> pd.DataFrame:
    """
    加载 IF9999 1 分钟 OHLCV 数据。

    :param data_path: CSV 文件路径
    :returns: DataFrame with datetime index and OHLCV columns
    :raises FileNotFoundError: 如果数据文件不存在
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    # 计算美元交易量（价格 × 成交量 × 合约乘数）
    df['dollar_volume'] = df['close'] * df['volume'] * CONTRACT_MULTIPLIER

    print(f"加载完成: {len(df)} 行, 时间范围 {df.index.min()} ~ {df.index.max()}")
    return df


def compute_dynamic_thresholds(
    df: pd.DataFrame, target_daily_bars: int, ewma_span: int
) -> pd.Series:
    """
    计算动态 Dollar Bar 阈值。

    阈值 = EWMA(日均美元交易量) / 目标每日 Bar 数

    :param df: OHLCV DataFrame with dollar_volume column
    :param target_daily_bars: 目标每日 Bar 数量
    :param ewma_span: EWMA 平滑窗口
    :returns: Series of daily thresholds indexed by date
    """
    # 计算每日总美元交易量
    daily_dollar = df['dollar_volume'].groupby(df.index.date).sum()
    daily_dollar.index = pd.to_datetime(daily_dollar.index)
    daily_dollar.name = 'daily_dollar'

    # 处理零值（非交易日）
    daily_dollar = daily_dollar.replace(0, np.nan).ffill()

    # EWMA 平滑
    daily_ewma = daily_dollar.ewm(span=ewma_span, min_periods=1).mean()

    # 计算阈值
    thresholds = daily_ewma / target_daily_bars
    thresholds.name = 'threshold'

    print(f"✅ 阈值计算完成: EWMA span={ewma_span}, target={target_daily_bars} bars/day")
    print(f"   平均阈值: {thresholds.mean():,.0f} 元/bar")
    return thresholds


@njit(nogil=True)
def _dynamic_dollar_bar_indexer(
    prices: np.ndarray,
    volumes: np.ndarray,
    thresholds: np.ndarray,
    contract_multiplier: float,
) -> NumbaList:
    """
    动态阈值 Dollar Bar indexer（Numba 加速）。

    :param prices: 价格数组
    :param volumes: 成交量数组
    :param thresholds: 每个时间点对应的阈值数组
    :param contract_multiplier: 合约乘数
    :returns: Bar close 索引列表

    .. note::
        首个元素的 dollar value 在循环外初始化，循环从索引 1 开始。
    """
    n = len(prices)
    indices = NumbaList()
    indices.append(0)  # 第一个点作为起始

    cum_dollar = prices[0] * volumes[0] * contract_multiplier  # 初始化包含首元素
    for i in range(1, n):  # 从索引 1 开始
        cum_dollar += prices[i] * volumes[i] * contract_multiplier
        if cum_dollar >= thresholds[i]:
            indices.append(i)
            cum_dollar = cum_dollar - thresholds[i]

    return indices


def _build_ohlcv_from_indices(df: pd.DataFrame, close_indices: np.ndarray) -> pd.DataFrame:
    """
    从 close 索引构建 OHLCV DataFrame。

    :param df: 原始 1 分钟数据
    :param close_indices: Bar close 索引数组
    :returns: Dollar Bars DataFrame
    """
    if len(close_indices) < 2:
        return pd.DataFrame()

    bars_data = []
    for i in range(len(close_indices) - 1):
        start_idx = close_indices[i]
        end_idx = close_indices[i + 1]

        segment = df.iloc[start_idx:end_idx + 1]

        bar = {
            'timestamp': df.index[end_idx],
            'open': segment['open'].iloc[0],
            'high': segment['high'].max(),
            'low': segment['low'].min(),
            'close': segment['close'].iloc[-1],
            'volume': segment['volume'].sum(),
            'dollar_volume': segment['dollar_volume'].sum(),
            'n_ticks': len(segment),
        }
        bars_data.append(bar)

    bars_df = pd.DataFrame(bars_data)
    bars_df = bars_df.set_index('timestamp')
    return bars_df


def build_dollar_bars(df: pd.DataFrame, thresholds: pd.Series) -> pd.DataFrame:
    """
    从 1 分钟数据构建 Dollar Bars。

    将每分钟视为一个 tick，使用动态阈值构建 Bars。

    :param df: 1 分钟 OHLCV DataFrame
    :param thresholds: 每日阈值 Series
    :returns: Dollar Bars DataFrame
    """
    # 将阈值映射到每分钟
    df = df.copy()
    df['date'] = df.index.normalize()
    threshold_map = thresholds.to_dict()
    df['threshold'] = df['date'].map(lambda x: threshold_map.get(pd.Timestamp(x), thresholds.mean()))
    df['threshold'] = df['threshold'].fillna(thresholds.mean())

    # 准备数组
    prices = df['close'].values.astype(np.float64)
    volumes = df['volume'].values.astype(np.float64)
    threshold_arr = df['threshold'].values.astype(np.float64)

    # 构建 Bar 索引（Numba 加速）
    close_indices = _dynamic_dollar_bar_indexer(
        prices, volumes, threshold_arr, float(CONTRACT_MULTIPLIER)
    )
    close_indices = np.array(close_indices, dtype=np.int64)

    # 从索引构建 OHLCV
    bars = _build_ohlcv_from_indices(df, close_indices)

    print(f"✅ Dollar Bars 构建完成: {len(bars)} bars")
    print(f"   时间范围: {bars.index.min()} ~ {bars.index.max()}")
    return bars


def compute_returns(bars: pd.DataFrame, log: bool = True) -> pd.Series:
    """
    计算收益率序列。

    :param bars: Bar DataFrame with close column
    :param log: 是否使用对数收益率
    :returns: 收益率 Series
    """
    if log:
        returns = np.log(bars['close']).diff().dropna()
    else:
        returns = bars['close'].pct_change().dropna()
    returns.name = 'returns'
    return returns


def validate_independence(returns: pd.Series, acf_lags: list = [1, 5, 10]) -> dict:
    """
    第一刀：独立性验证（序列相关性检验）。

    :param returns: 收益率序列
    :param acf_lags: 自相关检验滞后阶数
    :returns: 验证结果字典
    """
    # AC1 计算
    ac1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]

    # Ljung-Box 检验
    lb_result = acorr_ljungbox(returns, lags=[10], return_df=True)
    lb_pvalue = lb_result['lb_pvalue'].values[0]

    result = {
        'AC1': ac1,
        'Ljung_Box_p': lb_pvalue,
        'pass': abs(ac1) < 0.05 and lb_pvalue > 0.05,
        'description': f"AC1={ac1:.4f} (目标≈0), LB p={lb_pvalue:.4f} (目标>0.05)"
    }
    return result


def validate_identically_distributed(returns: pd.Series) -> dict:
    """
    第二刀：同分布验证（方差的方差检验）。

    :param returns: 收益率序列
    :returns: 验证结果字典
    """
    # 按月分组计算方差
    monthly_vars = returns.groupby(returns.index.to_period('M')).var()
    vov = monthly_vars.var()
    mean_var = monthly_vars.mean()

    result = {
        'VoV': vov,
        'mean_variance': mean_var,
        'VoV_ratio': vov / mean_var if mean_var > 0 else np.nan,
        'pass': vov < mean_var * 0.1,  # VoV < 均值的 10%
        'description': f"VoV={vov:.6f}, mean_var={mean_var:.6f}, ratio={vov/mean_var:.4f} (目标→0)"
    }
    return result


def validate_normality(returns: pd.Series) -> dict:
    """
    第三刀：正态性验证（Jarque-Bera 检验）。

    :param returns: 收益率序列
    :returns: 验证结果字典
    """
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)  # Fisher 定义（正态=0）

    result = {
        'JB_stat': jb_stat,
        'JB_p': jb_pvalue,
        'Skewness': skew,
        'Kurtosis': kurt + 3,  # 转换为 Pearson 定义（正态=3）
        'pass': jb_pvalue > 0.05 or (abs(skew) < 0.5 and abs(kurt) < 1),
        'description': f"JB={jb_stat:.2f}, Skew={skew:.4f}, Kurt={kurt+3:.4f} (目标: Skew≈0, Kurt≈3)"
    }
    return result


def run_three_knife_validation(time_returns: pd.Series, dollar_returns: pd.Series) -> dict:
    """
    对 Time Bars 和 Dollar Bars 进行三刀验证对比。

    :param time_returns: Time Bars 收益率
    :param dollar_returns: Dollar Bars 收益率
    :returns: 完整验证结果字典
    """
    results = {
        'time_bars': {
            'independence': validate_independence(time_returns),
            'identically_distributed': validate_identically_distributed(time_returns),
            'normality': validate_normality(time_returns),
        },
        'dollar_bars': {
            'independence': validate_independence(dollar_returns),
            'identically_distributed': validate_identically_distributed(dollar_returns),
            'normality': validate_normality(dollar_returns),
        }
    }

    # 打印对比结果
    print("\n" + "=" * 60)
    print("  三刀验证结果对比")
    print("=" * 60)
    print(f"\n第一刀（独立性）:")
    print(f"  Time Bars:  {results['time_bars']['independence']['description']}")
    print(f"  Dollar Bars: {results['dollar_bars']['independence']['description']}")
    print(f"\n第二刀（同分布）:")
    print(f"  Time Bars:  {results['time_bars']['identically_distributed']['description']}")
    print(f"  Dollar Bars: {results['dollar_bars']['identically_distributed']['description']}")
    print(f"\n第三刀（正态性）:")
    print(f"  Time Bars:  {results['time_bars']['normality']['description']}")
    print(f"  Dollar Bars: {results['dollar_bars']['normality']['description']}")

    return results


if __name__ == "__main__":
    # Step 1: 加载数据
    df = load_if_data(DATA_PATH)
    assert len(df) > 0, "数据为空"
    assert 'dollar_volume' in df.columns, "缺少 dollar_volume 列"

    # 输出基本统计
    daily_dollar_vol = df['dollar_volume'].groupby(df.index.date).sum()
    print(f"日均美元交易量: {daily_dollar_vol.mean():,.0f}")
    print(f"数据列: {df.columns.tolist()}")

    # Step 2: 计算动态阈值
    thresholds = compute_dynamic_thresholds(df, TARGET_DAILY_BARS, EWMA_SPAN)
    assert len(thresholds) > 0, "阈值为空"
    print(f"阈值范围: {thresholds.min():,.0f} ~ {thresholds.max():,.0f}")

    # Step 3: 构建 Dollar Bars
    bars = build_dollar_bars(df, thresholds)
    assert len(bars) > 0, "Dollar Bars 为空"
    assert 'close' in bars.columns, "缺少 close 列"

    # 验证每日 Bar 数量
    daily_counts = bars.groupby(bars.index.date).size()
    print(f"平均每日 Bar 数: {daily_counts.mean():.1f} (目标: {TARGET_DAILY_BARS})")
    print(f"Dollar Bars 列: {bars.columns.tolist()}")
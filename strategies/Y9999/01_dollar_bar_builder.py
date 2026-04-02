"""
01_dollar_bar_builder.py - Y9999 Dollar Bars 构建与验证

流程:
1. 加载 1 分钟 OHLCV 数据
2. 转换为 tick-like 格式（每分钟作为一个采样点）
3. 计算动态阈值（EWMA 日均美元交易量 / 目标 Bar 数）
4. 构建 Dollar Bars
5. 三刀验证（独立性、同分布、正态性）
6. 可视化输出
7. 参数优化（可选）

参数优化加权评分（AFML 优先级）:
- 独立性 50%: AC1 25% + Ljung-Box p 25%
- 同分布 30%: VoV ratio 30%
- 正态性 20%: JB 10% + Skew 5% + Kurt 5%
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", context="paper")
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.Y9999.config import (
    DATA_PATH, CONTRACT_MULTIPLIER, TARGET_DAILY_BARS, EWMA_SPAN,
    ACF_LAGS, BARS_DIR, FIGURES_DIR, OUTPUT_DIR
)

from numba import njit
from numba.typed import List as NumbaList


def load_if_data(data_path: str) -> pd.DataFrame:
    """
    加载 Y9999 1 分钟 OHLCV 数据。

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
    计算动态 Dollar bar 阈值。

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


# ============================================================
# 三刀验证
# ============================================================

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


# ============================================================
# 参数优化
# ============================================================

def compute_weighted_score(
    ac1: float,
    lb_p: float,
    vov_ratio: float,
    jb_stat: float,
    skew: float,
    kurt: float,
) -> dict:
    """
    计算加权评分（AFML 优先级：独立性 > 同分布 > 正态性）。

    权重分配：
    - 第一刀（独立性）：50%
        - AC1: 25%（目标 ≈ 0，归一化：min-max 到 0-1）
        - Ljung-Box p: 25%（目标 > 0.05，归一化：1 - min(p, 0.05)/0.05）
    - 第二刀（同分布）：30%
        - VoV ratio: 30%（目标 → 0，归一化：min-max 到 0-1）
    - 第三刀（正态性）：20%
        - JB: 10%（目标最低，对数归一化）
        - Skew: 5%（目标 ≈ 0，归一化：abs(skew)/2，上限 1）
        - Kurt: 5%（目标 ≈ 3，归一化：abs(kurt-3)/50，上限 1）

    :returns: 评分详情字典
    """
    # 第一刀：独立性（50%）
    # AC1 归一化：假设合理范围 [0, 0.05]，越小越好
    ac1_score = min(abs(ac1) / 0.05, 1.0)  # 0-1，越小越好
    # Ljung-Box p 归一化：目标 > 0.05，越大越好
    lb_score = 1.0 - min(lb_p, 0.05) / 0.05  # 0-1，越小越好

    independence_score = 0.25 * ac1_score + 0.25 * lb_score  # 权重 50%

    # 第二刀：同分布（30%）
    # VoV ratio 归一化：假设合理范围 [0, 0.1]，越小越好
    vov_score = min(vov_ratio / 0.1, 1.0) if not np.isnan(vov_ratio) else 1.0

    identically_distributed_score = 0.30 * vov_score  # 权重 30%

    # 第三刀：正态性（20%）
    # JB 归一化：对数缩放，假设范围 [1e4, 1e9]
    jb_score = min(np.log10(max(jb_stat, 1)) / 9, 1.0)  # log10(JB)/9，0-1
    # Skew 归一化：假设合理范围 [0, 2]
    skew_score = min(abs(skew) / 2, 1.0)
    # Kurt 归一化：假设合理范围 [3, 50]，偏离 3 的程度
    kurt_score = min(abs(kurt - 3) / 47, 1.0)

    normality_score = 0.10 * jb_score + 0.05 * skew_score + 0.05 * kurt_score  # 权重 20%

    # 综合评分（越低越好，范围 0-1）
    total_score = independence_score + identically_distributed_score + normality_score

    return {
        'ac1_score': ac1_score,
        'lb_score': lb_score,
        'vov_score': vov_score,
        'jb_score': jb_score,
        'skew_score': skew_score,
        'kurt_score': kurt_score,
        'independence_score': independence_score,
        'identically_distributed_score': identically_distributed_score,
        'normality_score': normality_score,
        'weighted_score': total_score,
    }


def evaluate_target_bars(df: pd.DataFrame, target_bars: int, ewma_span: int = 20) -> dict:
    """
    评估指定 TARGET_DAILY_BARS 参数的效果。

    :param df: 1 分钟 OHLCV 数据
    :param target_bars: 目标每日 Bar 数量
    :param ewma_span: EWMA 窗口
    :returns: 评估结果字典
    """
    # 构建 Dollar Bars
    thresholds = compute_dynamic_thresholds(df, target_bars, ewma_span)
    bars = build_dollar_bars(df, thresholds)
    returns = compute_returns(bars)

    # 计算验证指标
    ac1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
    lb_result = acorr_ljungbox(returns, lags=[10], return_df=True)
    lb_p = lb_result['lb_pvalue'].values[0]

    # VoV
    monthly_vars = returns.groupby(returns.index.to_period('M')).var()
    vov = monthly_vars.var()
    mean_var = monthly_vars.mean()
    vov_ratio = vov / mean_var if mean_var > 0 else np.nan

    # JB 和分布统计
    jb_stat, _ = stats.jarque_bera(returns)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns) + 3

    # 计算实际每日 Bar 数
    daily_counts = bars.groupby(bars.index.date).size()
    actual_bars_per_day = daily_counts.mean()

    # 计算加权评分
    scores = compute_weighted_score(ac1, lb_p, vov_ratio, jb_stat, skew, kurt)

    return {
        'target_bars': target_bars,
        'n_bars': len(bars),
        'actual_bars_per_day': actual_bars_per_day,
        'AC1': ac1,
        'Ljung_Box_p': lb_p,
        'VoV_ratio': vov_ratio,
        'JB_stat': jb_stat,
        'Skewness': skew,
        'Kurtosis': kurt,
        # 加权评分详情
        'independence': scores['independence_score'],
        'identically_dist': scores['identically_distributed_score'],
        'normality': scores['normality_score'],
        'weighted_score': scores['weighted_score'],
    }


def run_parameter_optimization(df: pd.DataFrame, target_range: list) -> pd.DataFrame:
    """
    运行参数优化，对比不同 TARGET_DAILY_BARS 的效果。

    :param df: 1 分钟 OHLCV 数据
    :param target_range: 目标每日 Bar 数量范围
    :returns: 对比结果 DataFrame
    """
    results = []

    print("=" * 70)
    print("  Dollar Bars 参数优化")
    print("=" * 70)

    for target in target_range:
        print(f"\n  测试 TARGET_DAILY_BARS = {target}...")
        result = evaluate_target_bars(df, target, EWMA_SPAN)
        results.append(result)
        print(f"    实际日均: {result['actual_bars_per_day']:.1f} bars")
        print(f"    AC1: {result['AC1']:.4f}, JB: {result['JB_stat']:.2e}, 加权评分: {result['weighted_score']:.4f}")

    results_df = pd.DataFrame(results)
    return results_df


def plot_parameter_comparison(results_df: pd.DataFrame, save_path: str):
    """
    绘制参数对比图。

    :param results_df: 对比结果 DataFrame
    :param save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    x = results_df['target_bars']

    # 1. AC1（越接近 0 越好）
    ax = axes[0, 0]
    ax.bar(x.astype(str), np.abs(results_df['AC1']), color='steelblue')
    ax.set_title('AC1 (lower is better)', fontsize=11)
    ax.set_xlabel('Target Bars/Day')
    ax.set_ylabel('|AC1|')

    # 2. Ljung-Box p-value（越大越好）
    ax = axes[0, 1]
    ax.bar(x.astype(str), results_df['Ljung_Box_p'], color='darkorange')
    ax.axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
    ax.set_title('Ljung-Box p-value (higher is better)', fontsize=11)
    ax.set_xlabel('Target Bars/Day')
    ax.set_ylabel('p-value')
    ax.legend()

    # 3. JB 统计量（越低越好）
    ax = axes[0, 2]
    ax.bar(x.astype(str), results_df['JB_stat'] / 1e6, color='forestgreen')
    ax.set_title('JB Statistic (lower is better, in millions)', fontsize=11)
    ax.set_xlabel('Target Bars/Day')
    ax.set_ylabel('JB (×10⁶)')

    # 4. 偏度（越接近 0 越好）
    ax = axes[1, 0]
    ax.bar(x.astype(str), np.abs(results_df['Skewness']), color='purple')
    ax.set_title('Skewness (closer to 0 is better)', fontsize=11)
    ax.set_xlabel('Target Bars/Day')
    ax.set_ylabel('|Skewness|')

    # 5. 峰度（越接近 3 越好）
    ax = axes[1, 1]
    ax.bar(x.astype(str), np.abs(results_df['Kurtosis'] - 3), color='coral')
    ax.set_title('Kurtosis Distance from 3 (lower is better)', fontsize=11)
    ax.set_xlabel('Target Bars/Day')
    ax.set_ylabel('|Kurtosis - 3|')

    # 6. 加权综合评分（按 AFML 优先级）
    ax = axes[1, 2]
    bars = ax.bar(x.astype(str), results_df['weighted_score'], color='teal')
    # 标记最优
    min_idx = results_df['weighted_score'].idxmin()
    bars[min_idx].set_color('gold')
    ax.set_title('Weighted Score (lower is better)\n独立50%+同分布30%+正态20%', fontsize=10)
    ax.set_xlabel('Target Bars/Day')
    ax.set_ylabel('Score')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 参数对比图已保存: {save_path}")


# ============================================================
# 可视化
# ============================================================

def plot_price_comparison(time_df: pd.DataFrame, dollar_bars: pd.DataFrame, save_path: str):
    """
    绘制 Time Bars vs Dollar Bars 价格走势对比图。

    :param time_df: 1 分钟 OHLCV DataFrame
    :param dollar_bars: Dollar Bars DataFrame
    :param save_path: 保存路径
    """
    # 采样 Time Bars 以提高可视化效率（每小时取一个点）
    time_sampled = time_df['close'].resample('H').last()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time Bars
    ax1.plot(time_sampled.index, time_sampled.values, color='steelblue', linewidth=0.8)
    ax1.set_title('Time Bars (1-min, sampled hourly)', fontsize=12)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.tick_params(axis='x', rotation=45)

    # Dollar Bars
    ax2.plot(dollar_bars.index, dollar_bars['close'].values, color='darkorange', linewidth=0.8)
    ax2.set_title(f'Dollar Bars (target={TARGET_DAILY_BARS} bars/day)', fontsize=12)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Price')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存价格对比图: {save_path}")


def plot_validation_metrics(time_returns: pd.Series, dollar_returns: pd.Series,
                            validation_results: dict, save_path: str):
    """
    绘制三刀验证指标对比图。

    :param time_returns: Time Bars 收益率
    :param dollar_returns: Dollar Bars 收益率
    :param validation_results: 验证结果字典
    :param save_path: 保存路径
    """
    fig = plt.figure(figsize=(12, 8))

    # 子图 1: ACF 对比
    ax1 = fig.add_subplot(2, 2, 1)
    plot_acf(time_returns.dropna(), lags=20, ax=ax1, title='Time Bars ACF', color='steelblue')
    ax1.set_ylim(-0.15, 0.15)

    ax2 = fig.add_subplot(2, 2, 2)
    plot_acf(dollar_returns.dropna(), lags=20, ax=ax2, title='Dollar Bars ACF', color='darkorange')
    ax2.set_ylim(-0.15, 0.15)

    # 子图 3: JB 统计量对比
    ax3 = fig.add_subplot(2, 2, 3)
    jb_values = [
        validation_results['time_bars']['normality']['JB_stat'],
        validation_results['dollar_bars']['normality']['JB_stat']
    ]
    bars_jb = ax3.bar(['Time Bars', 'Dollar Bars'], jb_values, color=['steelblue', 'darkorange'])
    ax3.set_title('Jarque-Bera Statistic (lower is better)')
    ax3.set_ylabel('JB Statistic')
    for bar, val in zip(bars_jb, jb_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}',
                 ha='center', va='bottom', fontsize=10)

    # 子图 4: VoV 对比
    ax4 = fig.add_subplot(2, 2, 4)
    vov_values = [
        validation_results['time_bars']['identically_distributed']['VoV'],
        validation_results['dollar_bars']['identically_distributed']['VoV']
    ]
    bars_vov = ax4.bar(['Time Bars', 'Dollar Bars'], vov_values, color=['steelblue', 'darkorange'])
    ax4.set_title('Variance of Variances (VoV) (lower is better)')
    ax4.set_ylabel('VoV')
    for bar, val in zip(bars_vov, vov_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存验证指标图: {save_path}")


def plot_return_distribution(time_returns: pd.Series, dollar_returns: pd.Series, save_path: str):
    """
    绘制收益率分布直方图对比。

    :param time_returns: Time Bars 收益率
    :param dollar_returns: Dollar Bars 收益率
    :param save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Time Bars 分布
    ax.hist(time_returns.dropna(), bins=50, density=True, alpha=0.6,
            color='steelblue', label='Time Bars', edgecolor='white')

    # Dollar Bars 分布
    ax.hist(dollar_returns.dropna(), bins=50, density=True, alpha=0.6,
            color='darkorange', label='Dollar Bars', edgecolor='white')

    # 正态分布参考曲线
    from scipy.stats import norm
    x = np.linspace(dollar_returns.min(), dollar_returns.max(), 100)
    mean, std = dollar_returns.mean(), dollar_returns.std()
    ax.plot(x, norm.pdf(x, mean, std), 'k--', linewidth=1.5, label='Normal Reference')

    ax.set_title('Return Distribution Comparison')
    ax.set_xlabel('Log Returns')
    ax.set_ylabel('Density')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存收益率分布图: {save_path}")


def update_config_target_bars(best_target: int):
    """
    更新 config.py 中的 TARGET_DAILY_BARS 参数。

    :param best_target: 最优参数值
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')
    with open(config_path, 'r') as f:
        config_content = f.read()

    # 替换 TARGET_DAILY_BARS 的值
    new_config = re.sub(
        r'TARGET_DAILY_BARS = \d+',
        f'TARGET_DAILY_BARS = {best_target}',
        config_content
    )
    with open(config_path, 'w') as f:
        f.write(new_config)
    print(f"  ✅ 已更新 config.py: TARGET_DAILY_BARS = {best_target}")


def main(run_optimization: bool = True):
    """
    Y9999 Dollar Bars 构建与验证主流程。

    :param run_optimization: 是否运行参数优化
    """
    print("=" * 60)
    print("  Y9999 Dollar Bars 构建与验证")
    print("=" * 60)

    # 确保输出目录存在
    os.makedirs(BARS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载数据
    print("\n[Step 1] 加载 Y9999 1 分钟数据...")
    df = load_if_data(DATA_PATH)

    # Step 2: 参数优化（可选）
    if run_optimization:
        print("\n[Step 2] 参数优化...")
        target_range = [4, 6, 8, 10, 12, 15, 20, 25, 30]
        results_df = run_parameter_optimization(df, target_range)

        # 打印结果表格
        print("\n" + "=" * 70)
        print("  参数优化结果")
        print("=" * 70)
        print(results_df[['target_bars', 'actual_bars_per_day', 'AC1', 'Ljung_Box_p',
                          'JB_stat', 'Skewness', 'Kurtosis', 'weighted_score']].to_string(index=False))

        # 找出最优参数
        best_row = results_df.loc[results_df['weighted_score'].idxmin()]
        best_target = int(best_row['target_bars'])

        print("\n" + "-" * 70)
        print(f"  最优参数: TARGET_DAILY_BARS = {best_target}")
        print(f"  加权综合评分: {best_row['weighted_score']:.4f}")
        print("-" * 70)

        # 更新 config
        update_config_target_bars(best_target)

        # 保存参数优化结果
        results_df.to_csv(os.path.join(FIGURES_DIR, '01_parameter_optimization.csv'), index=False)
        plot_parameter_comparison(results_df, os.path.join(FIGURES_DIR, '01_parameter_optimization.png'))

        # 使用最优参数构建 Dollar Bars
        target_daily_bars = best_target
    else:
        target_daily_bars = TARGET_DAILY_BARS

    # Step 3: 计算动态阈值
    print("\n[Step 3] 计算动态阈值...")
    thresholds = compute_dynamic_thresholds(df, target_daily_bars, EWMA_SPAN)

    # Step 4: 构建 Dollar Bars
    print("\n[Step 4] 构建 Dollar Bars...")
    bars = build_dollar_bars(df, thresholds)

    # Step 5: 保存 Dollar Bars
    bars_path = os.path.join(BARS_DIR, f'dollar_bars_target{target_daily_bars}.parquet')
    bars.to_parquet(bars_path)
    print(f"✅ Dollar Bars 已保存: {bars_path}")

    # Step 6: 三刀验证
    print("\n[Step 5] 三刀验证...")
    time_returns = compute_returns(df)
    dollar_returns = compute_returns(bars)
    validation_results = run_three_knife_validation(time_returns, dollar_returns)

    # Step 7: 可视化
    print("\n[Step 6] 生成可视化图表...")
    plot_price_comparison(df, bars, os.path.join(FIGURES_DIR, '01_price_comparison.png'))
    plot_validation_metrics(time_returns, dollar_returns, validation_results,
                            os.path.join(FIGURES_DIR, '01_validation_metrics.png'))
    plot_return_distribution(time_returns, dollar_returns,
                             os.path.join(FIGURES_DIR, '01_return_distribution.png'))

    print("\n" + "=" * 60)
    print("  流程完成")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"  - Bars: {bars_path}")
    print(f"  - Figures: {FIGURES_DIR}/")


if __name__ == "__main__":
    main(run_optimization=True)
"""
02_parameter_optimization.py - Dollar Bars 参数优化

对比不同 TARGET_DAILY_BARS 参数的效果，找出最优配置。
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from numba.typed import List as NumbaList

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import (
    DATA_PATH, CONTRACT_MULTIPLIER, EWMA_SPAN, FIGURES_DIR
)

sns.set_theme(style="whitegrid", context="paper")


# ============================================================
# 核心函数（从 01_dollar_bar_builder.py 复用）
# ============================================================

def load_if_data(data_path: str) -> pd.DataFrame:
    """加载 IF9999 1 分钟 OHLCV 数据。"""
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    df['dollar_volume'] = df['close'] * df['volume'] * CONTRACT_MULTIPLIER
    return df


def compute_dynamic_thresholds(df: pd.DataFrame, target_daily_bars: int, ewma_span: int) -> pd.Series:
    """计算动态 Dollar Bar 阈值。"""
    daily_dollar = df['dollar_volume'].groupby(df.index.date).sum()
    daily_dollar.index = pd.to_datetime(daily_dollar.index)
    daily_dollar = daily_dollar.replace(0, np.nan).ffill()
    daily_ewma = daily_dollar.ewm(span=ewma_span, min_periods=1).mean()
    thresholds = daily_ewma / target_daily_bars
    return thresholds


@njit(nogil=True)
def _dynamic_dollar_bar_indexer(
    prices: np.ndarray,
    volumes: np.ndarray,
    thresholds: np.ndarray,
    contract_multiplier: float,
) -> NumbaList:
    """动态阈值 Dollar Bar indexer（Numba 加速）。"""
    n = len(prices)
    indices = NumbaList()
    indices.append(0)
    cum_dollar = prices[0] * volumes[0] * contract_multiplier
    for i in range(1, n):
        cum_dollar += prices[i] * volumes[i] * contract_multiplier
        if cum_dollar >= thresholds[i]:
            indices.append(i)
            cum_dollar = cum_dollar - thresholds[i]
    return indices


def build_dollar_bars(df: pd.DataFrame, thresholds: pd.Series) -> pd.DataFrame:
    """从 1 分钟数据构建 Dollar Bars。"""
    df = df.copy()
    df['date'] = df.index.normalize()
    threshold_map = thresholds.to_dict()
    df['threshold'] = df['date'].map(lambda x: threshold_map.get(pd.Timestamp(x), thresholds.mean()))
    df['threshold'] = df['threshold'].fillna(thresholds.mean())

    prices = df['close'].values.astype(np.float64)
    volumes = df['volume'].values.astype(np.float64)
    threshold_arr = df['threshold'].values.astype(np.float64)

    close_indices = _dynamic_dollar_bar_indexer(prices, volumes, threshold_arr, float(CONTRACT_MULTIPLIER))
    close_indices = np.array(close_indices, dtype=np.int64)

    return _build_ohlcv_from_indices(df, close_indices)


def _build_ohlcv_from_indices(df: pd.DataFrame, close_indices: np.ndarray) -> pd.DataFrame:
    """从 close 索引构建 OHLCV DataFrame。"""
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


def compute_returns(bars: pd.DataFrame, log: bool = True) -> pd.Series:
    """计算收益率序列。"""
    if log:
        returns = np.log(bars['close']).diff().dropna()
    else:
        returns = bars['close'].pct_change().dropna()
    return returns


# ============================================================
# 验证指标计算
# ============================================================

def compute_ac1(returns: pd.Series) -> float:
    """计算一阶自相关系数。"""
    return np.corrcoef(returns[:-1], returns[1:])[0, 1]


def compute_ljung_box_p(returns: pd.Series) -> float:
    """计算 Ljung-Box 检验 p-value。"""
    lb_result = acorr_ljungbox(returns, lags=[10], return_df=True)
    return lb_result['lb_pvalue'].values[0]


def compute_vov(returns: pd.Series) -> float:
    """计算方差的方差（VoV ratio）。"""
    monthly_vars = returns.groupby(returns.index.to_period('M')).var()
    vov = monthly_vars.var()
    mean_var = monthly_vars.mean()
    return vov / mean_var if mean_var > 0 else np.nan


def compute_jb_stat(returns: pd.Series) -> float:
    """计算 Jarque-Bera 统计量。"""
    jb_stat, _ = stats.jarque_bera(returns)
    return jb_stat


def compute_skewness(returns: pd.Series) -> float:
    """计算偏度。"""
    return stats.skew(returns)


def compute_kurtosis(returns: pd.Series) -> float:
    """计算峰度（Pearson 定义，正态=3）。"""
    return stats.kurtosis(returns) + 3


# ============================================================
# 参数优化主流程
# ============================================================

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
    ac1 = compute_ac1(returns)
    lb_p = compute_ljung_box_p(returns)
    vov_ratio = compute_vov(returns)
    jb_stat = compute_jb_stat(returns)
    skew = compute_skewness(returns)
    kurt = compute_kurtosis(returns)

    # 计算实际每日 Bar 数
    daily_counts = bars.groupby(bars.index.date).size()
    actual_bars_per_day = daily_counts.mean()

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
        # 综合评分（越低越好）
        'score': abs(ac1) + jb_stat / 1e8 + abs(skew) + abs(kurt - 3),
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
        print(f"    AC1: {result['AC1']:.4f}, JB: {result['JB_stat']:.2e}")

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

    # 6. 综合评分
    ax = axes[1, 2]
    bars = ax.bar(x.astype(str), results_df['score'], color='teal')
    # 标记最优
    min_idx = results_df['score'].idxmin()
    bars[min_idx].set_color('gold')
    ax.set_title('Composite Score (lower is better)', fontsize=11)
    ax.set_xlabel('Target Bars/Day')
    ax.set_ylabel('Score')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 参数对比图已保存: {save_path}")


def main():
    """主流程。"""
    print("\n[Step 1] 加载数据...")
    df = load_if_data(DATA_PATH)
    print(f"  数据量: {len(df)} 行")

    # 定义测试范围
    target_range = [20, 30, 40, 50, 60, 80, 100]

    print("\n[Step 2] 运行参数优化...")
    results_df = run_parameter_optimization(df, target_range)

    # 打印结果表格
    print("\n" + "=" * 70)
    print("  参数优化结果")
    print("=" * 70)
    print(results_df[['target_bars', 'actual_bars_per_day', 'AC1', 'Ljung_Box_p',
                       'JB_stat', 'Skewness', 'Kurtosis', 'score']].to_string(index=False))

    # 找出最优参数
    best_row = results_df.loc[results_df['score'].idxmin()]
    print("\n" + "-" * 70)
    print(f"  最优参数: TARGET_DAILY_BARS = {int(best_row['target_bars'])}")
    print(f"  实际日均: {best_row['actual_bars_per_day']:.1f} bars")
    print(f"  AC1: {best_row['AC1']:.4f}")
    print(f"  JB: {best_row['JB_stat']:.2e}")
    print(f"  综合评分: {best_row['score']:.4f}")
    print("-" * 70)

    # 保存结果
    os.makedirs(FIGURES_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(FIGURES_DIR, 'parameter_optimization.csv'), index=False)

    # 绘制对比图
    plot_parameter_comparison(results_df, os.path.join(FIGURES_DIR, 'parameter_optimization.png'))

    print("\n完成！")


if __name__ == "__main__":
    main()
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
    ac1 = compute_ac1(returns)
    lb_p = compute_ljung_box_p(returns)
    vov_ratio = compute_vov(returns)
    jb_stat = compute_jb_stat(returns)
    skew = compute_skewness(returns)
    kurt = compute_kurtosis(returns)

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


def main():
    """主流程。"""
    print("\n[Step 1] 加载数据...")
    df = load_if_data(DATA_PATH)
    print(f"  数据量: {len(df)} 行")

    # 定义测试范围（从更小的值开始）
    target_range = [4, 6, 8, 10, 12, 15, 20, 25, 30]

    print("\n[Step 2] 运行参数优化...")
    results_df = run_parameter_optimization(df, target_range)

    # 打印结果表格
    print("\n" + "=" * 70)
    print("  参数优化结果")
    print("=" * 70)
    print(results_df[['target_bars', 'actual_bars_per_day', 'AC1', 'Ljung_Box_p',
                       'JB_stat', 'Skewness', 'Kurtosis', 'weighted_score']].to_string(index=False))

    # 打印各维度评分
    print("\n" + "=" * 70)
    print("  加权评分详情（AFML 优先级）")
    print("=" * 70)
    print(results_df[['target_bars', 'independence', 'identically_dist', 'normality', 'weighted_score']].to_string(index=False))

    # 找出最优参数
    best_row = results_df.loc[results_df['weighted_score'].idxmin()]
    best_target = int(best_row['target_bars'])

    print("\n" + "-" * 70)
    print(f"  最优参数: TARGET_DAILY_BARS = {best_target}")
    print(f"  实际日均: {best_row['actual_bars_per_day']:.1f} bars")
    print(f"  独立性评分: {best_row['independence']:.4f} (权重 50%)")
    print(f"  同分布评分: {best_row['identically_dist']:.4f} (权重 30%)")
    print(f"  正态性评分: {best_row['normality']:.4f} (权重 20%)")
    print(f"  加权综合评分: {best_row['weighted_score']:.4f}")
    print("-" * 70)

    # 保存最优参数的 Dollar Bars
    print("\n[Step 3] 保存最优参数的 Dollar Bars...")
    from strategies.IF9999.config import BARS_DIR

    thresholds = compute_dynamic_thresholds(df, best_target, EWMA_SPAN)
    best_bars = build_dollar_bars(df, thresholds)

    bars_path = os.path.join(BARS_DIR, f'dollar_bars_target{best_target}.parquet')
    best_bars.to_parquet(bars_path)
    print(f"  ✅ 已保存: {bars_path}")
    print(f"     Bar 数量: {len(best_bars)}")
    print(f"     时间范围: {best_bars.index.min()} ~ {best_bars.index.max()}")

    # 更新 config.py 中的默认参数
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')
    with open(config_path, 'r') as f:
        config_content = f.read()

    # 替换 TARGET_DAILY_BARS 的值
    import re
    new_config = re.sub(
        r'TARGET_DAILY_BARS = \d+',
        f'TARGET_DAILY_BARS = {best_target}',
        config_content
    )
    with open(config_path, 'w') as f:
        f.write(new_config)
    print(f"  ✅ 已更新 config.py: TARGET_DAILY_BARS = {best_target}")

    # 保存结果
    os.makedirs(FIGURES_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(FIGURES_DIR, 'parameter_optimization.csv'), index=False)

    # 绘制对比图
    plot_parameter_comparison(results_df, os.path.join(FIGURES_DIR, 'parameter_optimization.png'))

    print("\n完成！")


if __name__ == "__main__":
    main()
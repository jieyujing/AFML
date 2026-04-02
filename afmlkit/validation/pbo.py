"""
PBO (Probability of Backtest Overfitting) 验证模块.

根据 AFML Chapter 14，PBO 用于评估回测过拟合风险。
通过 CPCV 生成多条路径，计算 Sharpe Ratio 分布，
估计策略在样本外的表现概率。

核心公式：
- PBO = P(SR_OOS < SR_IS_max)
- 通过 CPCV 路径的 Sharpe 分布估计

参考文献：
- Bailey & López de Prado (2017) "The Probability of Backtest Overfitting"
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple
from itertools import combinations


def calculate_pbo(
    sharpe_paths: np.ndarray,
    significance_level: float = 0.05
) -> Tuple[float, Dict]:
    """
    计算 PBO (Probability of Backtest Overfitting).

    PBO = P(SR_OOS < SR_IS_max)，表示最佳样本内策略在样本外表现不佳的概率。

    :param sharpe_paths: 各 CPCV 路径的 Sharpe Ratio 数组
    :param significance_level: 显著性水平（默认 5%）
    :returns: (pbo, stats) - PBO 值和详细统计信息
    """
    n_paths = len(sharpe_paths)
    if n_paths < 10:
        raise ValueError(f"需要至少 10 条路径计算 PBO，当前仅 {n_paths} 条")

    # Sharpe 分布统计
    sr_mean = np.mean(sharpe_paths)
    sr_std = np.std(sharpe_paths, ddof=1)
    sr_max = np.max(sharpe_paths)
    sr_min = np.min(sharpe_paths)

    # PBO 计算：样本内最大 Sharpe 在样本外分布中的位置
    # 参考: Bailey & López de Prado (2017) 公式 (8)
    # PBO = Φ( -μ/σ ) 其中 Φ 是标准正态 CDF
    # 这里使用更保守的定义：P(SR_OOS <= 0)
    pbo = norm.cdf(0, loc=sr_mean, scale=sr_std)

    # 替代定义：P(SR_OOS < SR_IS_max / 2)
    # 如果样本外 Sharpe 远低于样本内最佳，说明过拟合
    threshold_half_max = sr_max / 2
    pbo_half_max = norm.cdf(threshold_half_max, loc=sr_mean, scale=sr_std)

    # 统计信息
    stats = {
        'n_paths': n_paths,
        'sr_mean': sr_mean,
        'sr_std': sr_std,
        'sr_max': sr_max,
        'sr_min': sr_min,
        'sr_median': np.median(sharpe_paths),
        'pbo_zero': pbo,  # P(SR_OOS <= 0)
        'pbo_half_max': pbo_half_max,  # P(SR_OOS < SR_IS_max/2)
        'positive_rate': np.mean(sharpe_paths > 0),
        'is_significant': pbo < significance_level,
    }

    return pbo, stats


def calculate_pbo_from_returns(
    returns_paths: List[pd.Series],
    annualization_factor: float = 252
) -> Tuple[float, Dict[str, float]]:
    """
    从收益率路径计算 PBO.

    :param returns_paths: 各 CPCV 路径的收益率序列列表
    :param annualization_factor: 年化因子
    :returns: (pbo, stats) - PBO 值和详细统计信息
    """
    sharpe_paths = []
    for ret in returns_paths:
        if len(ret) < 20:
            continue
        sr = ret.mean() / ret.std(ddof=1) * np.sqrt(annualization_factor)
        sharpe_paths.append(sr)

    if len(sharpe_paths) < 10:
        raise ValueError(f"有效路径数不足: {len(sharpe_paths)}")

    pbo, stats = calculate_pbo(np.array(sharpe_paths))
    return float(pbo), stats


def estimate_optimal_trials(
    param_grid: Dict[str, List]
) -> int:
    """
    估计最优参数试验次数.

    基于参数网格组合数，而非固定经验值。

    :param param_grid: 参数网格字典 {param_name: [values]}
    :returns: 总试验次数
    """
    n_trials = 1
    for values in param_grid.values():
        n_trials *= len(values)
    return n_trials


def generate_backtest_paths(
    cpcv_splitter,
    X: pd.DataFrame,
    strategy_func,
    annualization_factor: float = 252
) -> Tuple[List[pd.Series], np.ndarray]:
    """
    使用 CPCV 生成多条回测路径.

    :param cpcv_splitter: CombinatorialPurgedKFold 实例
    :param X: 特征矩阵（带 DatetimeIndex）
    :param strategy_func: 策略函数，输入 (train_X, test_X) 返回 test 收益率
    :param annualization_factor: 年化因子
    :returns: (returns_paths, sharpe_paths) - 收益率路径和 Sharpe 路径
    """
    returns_paths = []
    sharpe_paths = []

    n_splits = cpcv_splitter.n_splits
    n_test_splits = cpcv_splitter.n_test_splits

    # CPCV 路径分配
    all_combos = list(combinations(range(n_splits), n_test_splits))
    n_paths = len(all_combos) * n_test_splits // n_splits

    # 初始化路径容器
    path_returns = {p: [] for p in range(n_paths)}

    for split_idx, (train_idx, test_idx, test_folds_info) in enumerate(cpcv_splitter.split(X)):
        # 运行策略
        train_X = X.iloc[train_idx]
        test_X = X.iloc[test_idx]
        test_returns = strategy_func(train_X, test_X)

        # 分配到各路径
        for fold_idx, _ in test_folds_info:
            # 找到该 (split_idx, fold_idx) 对应的路径
            path_idx = _get_path_index(split_idx, fold_idx, all_combos, n_splits)
            if path_idx < n_paths:
                path_returns[path_idx].extend(test_returns.values)

    # 转换为 Series
    for p in range(n_paths):
        if len(path_returns[p]) > 0:
            ret_series = pd.Series(path_returns[p])
            returns_paths.append(ret_series)
            sr = ret_series.mean() / ret_series.std(ddof=1) * np.sqrt(annualization_factor)
            sharpe_paths.append(sr)

    return returns_paths, np.array(sharpe_paths)


def _get_path_index(
    split_idx: int,
    fold_idx: int,
    all_combos: List[Tuple],
    n_splits: int
) -> int:
    """
    获取 (split_idx, fold_idx) 对应的路径索引.

    CPCV 路径分配规则：
    - 每条路径包含多个非重叠的 test fold 组合
    - 路径数 = C(n_splits, n_test_splits) * n_test_splits / n_splits
    """
    # 路径索引计算
    combo = all_combos[split_idx]
    fold_pos_in_combo = combo.index(fold_idx)

    # 每个组合贡献 n_test_splits 个路径片段
    # 路径索引 = split_idx * n_test_splits // n_splits + fold_pos_in_combo
    n_test_splits = len(combo)
    base_path = split_idx * n_test_splits // n_splits
    return base_path + fold_pos_in_combo


def pbo_validation_report(
    sharpe_paths: np.ndarray,
    n_trials: int,
    significance_level: float = 0.05
) -> str:
    """
    生成 PBO 验证报告.

    :param sharpe_paths: 各路径的 Sharpe Ratio
    :param n_trials: 参数试验次数
    :param significance_level: 显著性水平
    :returns: 报告文本
    """
    pbo, stats = calculate_pbo(sharpe_paths, significance_level)

    report = []
    report.append("=" * 70)
    report.append("  PBO (Probability of Backtest Overfitting) 验证报告")
    report.append("=" * 70)
    report.append(f"\n路径数量: {stats['n_paths']}")
    report.append(f"参数试验次数: {n_trials}")
    report.append("-" * 70)
    report.append("\nSharpe Ratio 分布:")
    report.append(f"  均值: {stats['sr_mean']:.4f}")
    report.append(f"  标准差: {stats['sr_std']:.4f}")
    report.append(f"  最大: {stats['sr_max']:.4f}")
    report.append(f"  最小: {stats['sr_min']:.4f}")
    report.append(f"  中位数: {stats['sr_median']:.4f}")
    report.append(f"  正值比例: {stats['positive_rate']*100:.1f}%")
    report.append("-" * 70)
    report.append("\nPBO 计算:")
    report.append(f"  P(SR_OOS <= 0): {stats['pbo_zero']:.4f}")
    report.append(f"  P(SR_OOS < SR_max/2): {stats['pbo_half_max']:.4f}")
    report.append("-" * 70)

    # 验证结论
    if pbo < 0.05:
        verdict = "✅ ACCEPT (过拟合风险低)"
    elif pbo < 0.25:
        verdict = "⚠️ BORDERLINE (轻度过拟合风险)"
    elif pbo < 0.50:
        verdict = "⚠️ WARNING (中度过拟合风险)"
    else:
        verdict = "❌ REJECT (高过拟合风险)"

    report.append(f"\n验证结论: {verdict}")
    report.append(f"  PBO = {pbo:.4f} (阈值: {significance_level})")
    report.append("=" * 70)

    return "\n".join(report)
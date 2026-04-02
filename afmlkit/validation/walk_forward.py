"""
Walk-Forward 验证框架.

根据 AFML 推荐的最佳实践，Walk-Forward 验证用于：
1. 验证策略在不同时期的稳定性
2. 模拟实盘滚动训练/预测过程
3. 评估策略退化风险

参考文献：
- AFML Chapter 14: Backtesting
- AFML Chapter 15: Understanding Strategy Risk
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class WalkForwardConfig:
    """Walk-Forward 配置."""
    train_window: int  # 训练窗口大小（样本数）
    test_window: int   # 测试窗口大小（样本数）
    step_size: int     # 滚动步长（样本数）
    embargo_pct: float = 0.01  # Embargo 比例
    min_train_samples: int = 100  # 最小训练样本数


@dataclass
class WalkForwardResult:
    """Walk-Forward 结果."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_samples: int
    test_samples: int
    sharpe: float
    returns: pd.Series
    metrics: Dict[str, float]


def generate_walk_forward_splits(
    X: pd.DataFrame,
    config: WalkForwardConfig
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    生成 Walk-Forward 分割索引.

    :param X: 特征矩阵（带 DatetimeIndex）
    :param config: Walk-Forward 配置
    :returns: [(train_idx, test_idx), ...] 分割列表
    """
    n_samples = len(X)
    splits = []

    # 计算 embargo 大小
    embargo_size = int(config.test_window * config.embargo_pct)

    # 起始位置
    start = 0

    while start + config.train_window + config.test_window <= n_samples:
        # 训练窗口
        train_start = start
        train_end = start + config.train_window
        train_idx = np.arange(train_start, train_end)

        # Embargo 空白期
        embargo_end = train_end + embargo_size

        # 测试窗口（在 embargo 后）
        test_start = embargo_end
        test_end = test_start + config.test_window

        if test_end > n_samples:
            break

        test_idx = np.arange(test_start, test_end)

        splits.append((train_idx, test_idx))

        # 滚动步长
        start += config.step_size

    return splits


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    config: WalkForwardConfig,
    strategy_func: Callable,
    annualization_factor: float = 252
) -> List[WalkForwardResult]:
    """
    执行 Walk-Forward 验证.

    :param X: 特征矩阵
    :param y: 目标变量
    :param config: Walk-Forward 配置
    :param strategy_func: 策略函数，输入 (train_X, train_y, test_X) 返回 test 预测/信号
    :param annualization_factor: 年化因子
    :returns: Walk-Forward 结果列表
    """
    splits = generate_walk_forward_splits(X, config)
    results = []

    for train_idx, test_idx in splits:
        # 提取数据
        train_X = X.iloc[train_idx]
        train_y = y.iloc[train_idx]
        test_X = X.iloc[test_idx]

        # 执行策略
        test_predictions = strategy_func(train_X, train_y, test_X)

        # 计算收益率
        if isinstance(test_predictions, pd.Series):
            test_returns = test_predictions
        else:
            # 假设 predictions 是信号，需要乘以收益率
            test_returns = y.iloc[test_idx] * test_predictions

        # 计算指标
        sharpe = test_returns.mean() / test_returns.std(ddof=1) * np.sqrt(annualization_factor) if len(test_returns) > 1 else 0.0

        metrics = {
            'sharpe': sharpe,
            'mean_return': test_returns.mean(),
            'std_return': test_returns.std(ddof=1),
            'win_rate': (test_returns > 0).mean(),
            'max_return': test_returns.max(),
            'min_return': test_returns.min(),
            'n_trades': len(test_returns),
        }

        result = WalkForwardResult(
            train_start=train_X.index.min(),
            train_end=train_X.index.max(),
            test_start=test_X.index.min(),
            test_end=test_X.index.max(),
            train_samples=len(train_X),
            test_samples=len(test_X),
            sharpe=sharpe,
            returns=test_returns,
            metrics=metrics
        )
        results.append(result)

    return results


def analyze_walk_forward_results(
    results: List[WalkForwardResult]
) -> Dict[str, float]:
    """
    分析 Walk-Forward 结果稳定性.

    :param results: Walk-Forward 结果列表
    :returns: 稳定性指标字典
    """
    sharpes = [r.sharpe for r in results]
    win_rates = [r.metrics['win_rate'] for r in results]

    # Sharpe 稳定性
    sr_mean = np.mean(sharpes)
    sr_std = np.std(sharpes, ddof=1)
    sr_min = np.min(sharpes)

    # 一致性检验
    positive_rate = np.mean(np.array(sharpes) > 0)
    consistent_rate = np.mean(np.array(sharpes) > sr_mean * 0.5)

    # 退化检测（后期 vs 前期）
    n_periods = len(results)
    early_sharpes = sharpes[:n_periods//3]
    late_sharpes = sharpes[-n_periods//3:]

    degradation = np.mean(early_sharpes) - np.mean(late_sharpes) if n_periods >= 3 else 0.0

    stability_metrics = {
        'n_periods': n_periods,
        'sr_mean': sr_mean,
        'sr_std': sr_std,
        'sr_min': sr_min,
        'sr_max': np.max(sharpes),
        'sr_range': np.max(sharpes) - np.min(sharpes),
        'positive_rate': positive_rate,
        'consistent_rate': consistent_rate,
        'degradation': degradation,
        'win_rate_mean': np.mean(win_rates),
        'win_rate_std': np.std(win_rates, ddof=1),
        'is_stable': sr_std < 0.5 and degradation < 0.3,
        'is_degrading': degradation > 0.3,
    }

    return stability_metrics


def walk_forward_report(
    results: List[WalkForwardResult],
    stability: Dict[str, float]
) -> str:
    """
    生成 Walk-Forward 验证报告.

    :param results: Walk-Forward 结果列表
    :param stability: 稳定性指标
    :returns: 报告文本
    """
    report = []
    report.append("=" * 70)
    report.append("  Walk-Forward 验证报告")
    report.append("=" * 70)

    report.append(f"\n验证周期数: {stability['n_periods']}")
    report.append("-" * 70)

    report.append("\n各周期 Sharpe Ratio:")
    for i, r in enumerate(results):
        period_str = f"{r.test_start.strftime('%Y-%m-%d')} ~ {r.test_end.strftime('%Y-%m-%d')}"
        report.append(f"  [{i+1}] {period_str}: SR={r.sharpe:.4f}, WinRate={r.metrics['win_rate']*100:.1f}%")

    report.append("-" * 70)
    report.append("\nSharpe Ratio 分布:")
    report.append(f"  均值: {stability['sr_mean']:.4f}")
    report.append(f"  标准差: {stability['sr_std']:.4f}")
    report.append(f"  范围: [{stability['sr_min']:.4f}, {stability['sr_max']:.4f}]")
    report.append(f"  正值比例: {stability['positive_rate']*100:.1f}%")

    report.append("\n稳定性分析:")
    report.append(f"  一致性比例: {stability['consistent_rate']*100:.1f}%")
    report.append(f"  退化程度: {stability['degradation']:.4f}")

    report.append("-" * 70)

    # 验证结论
    if stability['is_stable'] and stability['sr_mean'] > 0.5:
        verdict = "✅ ACCEPT (策略稳定，无明显退化)"
    elif stability['is_degrading']:
        verdict = "❌ REJECT (策略退化，实盘风险高)"
    elif stability['sr_std'] > 0.5:
        verdict = "⚠️ WARNING (策略波动大，稳定性差)"
    else:
        verdict = "⚠️ BORDERLINE (边缘通过)"

    report.append(f"\n验证结论: {verdict}")
    report.append("=" * 70)

    return "\n".join(report)


def plot_walk_forward_sharpe(
    results: List[WalkForwardResult],
    save_path: Optional[str] = None
):
    """
    绘制 Walk-Forward Sharpe 趋势图.

    :param results: Walk-Forward 结果列表
    :param save_path: 图片保存路径
    """
    import matplotlib.pyplot as plt

    sharpes = [r.sharpe for r in results]
    test_dates = [r.test_start for r in results]

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(test_dates)), sharpes, 'b-o', markersize=6)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    mean_sr = float(np.mean(sharpes))
    plt.axhline(y=mean_sr, color='green', linestyle='-', alpha=0.5, label=f'Mean SR = {mean_sr:.2f}')

    std_sr = float(np.std(sharpes))
    plt.fill_between(range(len(test_dates)), mean_sr - std_sr, mean_sr + std_sr,
                     alpha=0.2, color='green')

    # 设置 x 轴标签为日期
    ax = plt.gca()
    ax.set_xticks(range(len(test_dates)))
    ax.set_xticklabels([d.strftime('%Y-%m') for d in test_dates], rotation=45, ha='right')

    plt.xlabel('Test Period Start')
    plt.ylabel('Sharpe Ratio')
    plt.title('Walk-Forward Sharpe Ratio Trend')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
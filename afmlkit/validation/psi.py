"""
特征分布漂移检测 (Population Stability Index, PSI).

PSI 用于监控特征分布在不同时期的变化，评估模型输入的稳定性。

PSI 阈值解读：
- PSI < 0.10: 分布稳定，无明显漂移
- 0.10 ≤ PSI < 0.25: 轻微漂移，需要关注
- PSI ≥ 0.25: 显著漂移，需要重新训练模型

参考文献：
- AFML Chapter 5: Feature Importance
- Federal Reserve Supervisory Guidance on Model Risk Management (SR 11-7)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PSIResult:
    """PSI 检测结果."""
    feature_name: str
    psi_value: float
    status: str  # 'stable', 'warning', 'critical'
    train_dist: np.ndarray
    test_dist: np.ndarray
    bin_edges: np.ndarray


def calculate_psi(
    train_feature: np.ndarray,
    test_feature: np.ndarray,
    n_bins: int = 10,
    min_pct: float = 0.01
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算单个特征的 PSI (Population Stability Index).

    PSI = Σ (test_pct - train_pct) × ln(test_pct / train_pct)

    :param train_feature: 训练集特征值
    :param test_feature: 测试集特征值
    :param n_bins: 分桶数量
    :param min_pct: 最小百分比（避免除零）
    :returns: (psi_value, train_dist, test_dist, bin_edges)
    """
    # 移除 NaN
    train_clean = train_feature[~np.isnan(train_feature)]
    test_clean = test_feature[~np.isnan(test_feature)]

    if len(train_clean) < 10 or len(test_clean) < 10:
        return 0.0, np.array([]), np.array([]), np.array([])

    # 基于训练集确定分桶边界
    bin_edges = np.percentile(train_clean, np.linspace(0, 100, n_bins + 1))
    # 确保边界唯一
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0, np.array([]), np.array([]), np.array([])

    # 扩展边界以包含所有值
    bin_edges[0] = min(bin_edges[0], min(test_clean))
    bin_edges[-1] = max(bin_edges[-1], max(test_clean))

    # 计算分布
    train_counts, _ = np.histogram(train_clean, bins=bin_edges)
    test_counts, _ = np.histogram(test_clean, bins=bin_edges)

    # 转换为百分比
    train_pct = train_counts / len(train_clean)
    test_pct = test_counts / len(test_clean)

    # 应用最小百分比约束
    train_pct = np.maximum(train_pct, min_pct)
    test_pct = np.maximum(test_pct, min_pct)

    # 重新归一化
    train_pct = train_pct / train_pct.sum()
    test_pct = test_pct / test_pct.sum()

    # 计算 PSI
    psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))

    return psi, train_pct, test_pct, bin_edges


def calculate_psi_for_features(
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    n_bins: int = 10,
    threshold_warning: float = 0.10,
    threshold_critical: float = 0.25
) -> pd.DataFrame:
    """
    批量计算所有特征的 PSI.

    :param train_X: 训练集特征矩阵
    :param test_X: 测试集特征矩阵
    :param n_bins: 分桶数量
    :param threshold_warning: 警告阈值
    :param threshold_critical: 严重阈值
    :returns: PSI 结果 DataFrame
    """
    results = []
    common_features = [c for c in train_X.columns if c in test_X.columns]

    for feature in common_features:
        train_arr = train_X[feature].values.astype(float)
        test_arr = test_X[feature].values.astype(float)

        psi_val, _, _, _ = calculate_psi(
            train_arr, test_arr, n_bins
        )

        if psi_val < threshold_warning:
            status = 'stable'
        elif psi_val < threshold_critical:
            status = 'warning'
        else:
            status = 'critical'

        results.append({
            'feature': feature,
            'psi': psi_val,
            'status': status,
            'train_missing_pct': np.isnan(train_arr).mean(),
            'test_missing_pct': np.isnan(test_arr).mean(),
        })

    df = pd.DataFrame(results)
    df = df.sort_values('psi', ascending=False)
    return df


def psi_report(psi_df: pd.DataFrame) -> str:
    """
    生成 PSI 检测报告.

    :param psi_df: PSI 结果 DataFrame
    :returns: 报告文本
    """
    report = []
    report.append("=" * 70)
    report.append("  特征分布漂移检测 (PSI) 报告")
    report.append("=" * 70)

    n_features = len(psi_df)
    n_stable = (psi_df['status'] == 'stable').sum()
    n_warning = (psi_df['status'] == 'warning').sum()
    n_critical = (psi_df['status'] == 'critical').sum()

    report.append(f"\n总特征数: {n_features}")
    report.append(f"  稳定 (PSI < 0.10): {n_stable}")
    report.append(f"  警告 (0.10 ≤ PSI < 0.25): {n_warning}")
    report.append(f"  严重 (PSI ≥ 0.25): {n_critical}")

    # 显示漂移最严重的特征
    if n_critical > 0:
        report.append("\n" + "-" * 70)
        report.append("严重漂移特征:")
        critical = psi_df[psi_df['status'] == 'critical']
        for _, row in critical.iterrows():
            report.append(f"  - {row['feature']}: PSI = {row['psi']:.4f}")

    if n_warning > 0:
        report.append("\n" + "-" * 70)
        report.append("轻微漂移特征:")
        warning = psi_df[psi_df['status'] == 'warning']
        for _, row in warning.iterrows():
            report.append(f"  - {row['feature']}: PSI = {row['psi']:.4f}")

    report.append("-" * 70)

    # 验证结论
    if n_critical > 0:
        verdict = "❌ CRITICAL: 存在显著分布漂移，需要重新训练模型"
    elif n_warning > n_features * 0.2:
        verdict = "⚠️ WARNING: 较多特征出现漂移，建议检查数据质量"
    else:
        verdict = "✅ PASS: 特征分布稳定"

    report.append(f"\n验证结论: {verdict}")
    report.append("=" * 70)

    return "\n".join(report)


def plot_psi_distribution(
    psi_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    绘制 PSI 分布图.

    :param psi_df: PSI 结果 DataFrame
    :param save_path: 图片保存路径
    """
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(12, 6))

    features = psi_df['feature'].tolist()
    psi_values = psi_df['psi'].tolist()

    colors = []
    for psi in psi_values:
        if psi < 0.10:
            colors.append('#2ecc71')  # green
        elif psi < 0.25:
            colors.append('#f39c12')  # orange
        else:
            colors.append('#e74c3c')  # red

    y_pos = np.arange(len(features))
    ax.barh(y_pos, psi_values, color=colors, edgecolor='white')

    ax.axvline(x=0.10, color='orange', linestyle='--', label='Warning (0.10)')
    ax.axvline(x=0.25, color='red', linestyle='--', label='Critical (0.25)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8)
    ax.set_xlabel('PSI Value')
    ax.set_title('Feature Distribution Drift (PSI)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def detect_feature_drift(
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    n_bins: int = 10
) -> Tuple[pd.DataFrame, str]:
    """
    检测特征漂移并生成报告.

    :param train_X: 训练集特征矩阵
    :param test_X: 测试集特征矩阵
    :param n_bins: 分桶数量
    :returns: (psi_df, report_text)
    """
    psi_df = calculate_psi_for_features(train_X, test_X, n_bins)
    report = psi_report(psi_df)
    return psi_df, report
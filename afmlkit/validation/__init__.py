"""
afmlkit.validation 模块.

提供 AFML 回测验证工具：
- PurgedKFold: 防止信息泄露的交叉验证
- CombinatorialPurgedKFold (CPCV): 生成多条回测路径
- PBO: 回测过拟合概率计算
- Walk-Forward: 滚动窗口验证策略稳定性
- PSI: 特征分布漂移检测
"""

from .purged_cv import PurgedKFold
from .cpcv import CombinatorialPurgedKFold, generate_cpcv_paths
from .pbo import (
    calculate_pbo,
    calculate_pbo_from_returns,
    estimate_optimal_trials,
    generate_backtest_paths,
    pbo_validation_report,
)
from .walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    generate_walk_forward_splits,
    walk_forward_validate,
    analyze_walk_forward_results,
    walk_forward_report,
    plot_walk_forward_sharpe,
)
from .psi import (
    PSIResult,
    calculate_psi,
    calculate_psi_for_features,
    psi_report,
    plot_psi_distribution,
    detect_feature_drift,
)

__all__ = [
    'PurgedKFold',
    'CombinatorialPurgedKFold',
    'generate_cpcv_paths',
    'calculate_pbo',
    'calculate_pbo_from_returns',
    'estimate_optimal_trials',
    'generate_backtest_paths',
    'pbo_validation_report',
    'WalkForwardConfig',
    'WalkForwardResult',
    'generate_walk_forward_splits',
    'walk_forward_validate',
    'analyze_walk_forward_results',
    'walk_forward_report',
    'plot_walk_forward_sharpe',
    'PSIResult',
    'calculate_psi',
    'calculate_psi_for_features',
    'psi_report',
    'plot_psi_distribution',
    'detect_feature_drift',
]

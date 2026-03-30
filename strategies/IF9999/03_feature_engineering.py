"""
03_feature_engineering.py - IF9999 Phase 2 特征工程

流程:
1. 加载 Dollar Bars
2. FracDiff 参数优化（自动搜索最优 d）
3. 应用 FracDiff 生成平稳序列
4. 计算动态 CUSUM 阈值
5. 应用 CUSUM Filter 采样事件点
6. 可视化验证
7. 输出保存
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import (
    BARS_DIR, FIGURES_DIR, FEATURES_DIR,
    FRACDIFF_THRES, FRACDIFF_D_STEP, FRACDIFF_MAX_D,
    CUSUM_WINDOW
)

from afmlkit.feature.core.frac_diff import optimize_d, frac_diff_ffd
from afmlkit.feature.core.structural_break.adf import adf_test
from afmlkit.sampling.filters import cusum_filter_with_state

sns.set_theme(style="whitegrid", context="paper")


# ============================================================
# 后续函数将在后续 Task 中添加
# ============================================================

def main():
    """主流程占位符"""
    print("Phase 2 Feature Engineering - 待实现")


if __name__ == "__main__":
    main()
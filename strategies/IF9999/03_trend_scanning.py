"""
03_trend_scanning.py - IF9999 Phase 3 Trend Scanning 标签生成

流程:
1. 加载 Dollar Bars（原始价格）
2. 加载 CUSUM 事件点
3. 应用 Trend Scanning 生成标签
4. 保存输出
5. 生成诊断可视化

输出:
  - trend_labels.parquet: 包含 t1, t_value, side
  - 03_trend_distribution.png: side 和 t_value 分布
  - 03_trend_example.png: 趋势窗口示例
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import (
    BARS_DIR, FIGURES_DIR, FEATURES_DIR,
    TREND_WINDOWS
)

from afmlkit.feature.core.trend_scan import trend_scan_labels

sns.set_theme(style="whitegrid", context="paper")


# ============================================================
# 数据加载
# ============================================================

def load_dollar_bars(bars_path: str) -> pd.DataFrame:
    """
    加载 Dollar Bars 数据。

    :param bars_path: parquet 文件路径
    :returns: DataFrame with timestamp index
    """
    bars = pd.read_parquet(bars_path)
    print(f"✅ 加载 Dollar Bars: {len(bars)} bars")
    print(f"   时间范围: {bars.index.min()} ~ {bars.index.max()}")
    return bars


def load_cusum_events(events_path: str) -> pd.DatetimeIndex:
    """
    加载 CUSUM 事件点。

    :param events_path: parquet 文件路径
    :returns: 事件点 DatetimeIndex
    """
    events_df = pd.read_parquet(events_path)
    # 使用 timestamp 列构建 DatetimeIndex
    t_events = pd.DatetimeIndex(events_df['timestamp'])
    print(f"✅ 加载 CUSUM 事件点: {len(t_events)} 个")
    print(f"   时间范围: {t_events.min()} ~ {t_events.max()}")
    return t_events
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

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import (
    DATA_PATH, CONTRACT_MULTIPLIER, TARGET_DAILY_BARS, EWMA_SPAN,
    ACF_LAGS, BARS_DIR, FIGURES_DIR
)


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


if __name__ == "__main__":
    # 测试数据加载
    df = load_if_data(DATA_PATH)
    assert len(df) > 0, "数据为空"
    assert 'dollar_volume' in df.columns, "缺少 dollar_volume 列"

    # 输出基本统计
    daily_dollar_vol = df['dollar_volume'].groupby(df.index.date).sum()
    print(f"日均美元交易量: {daily_dollar_vol.mean():,.0f}")
    print(f"数据列: {df.columns.tolist()}")
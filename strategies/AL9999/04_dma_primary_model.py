"""
04_dma_primary_model.py - AL9999 DMA Primary Model 信号生成

基于双均线交叉 (Dual Moving Average Crossover) 在 CUSUM 事件点上生成方向信号。

流程:
1. 加载 Dollar Bars（价格序列）
2. 加载 CUSUM 事件点
3. 在全部 bar 上计算 EMA(5) / EMA(20)
4. 在事件点上采样 DMA 方向
5. 输出 DMA 信号

AFML 元标签方法论中的角色:
- DMA 是探路猎犬 (Primary Model): 提出假设 (Side)
- 不保证每次都对，但保证不漏过动能破裂 (高召回率)
- 趋势扫描 (Trend Scanning) 将在后续验证这些信号

输出:
  - dma_signals.parquet: timestamp, side, ema_fast, ema_slow, dma_spread
"""

import os
import sys
import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.config import (
    BARS_DIR, FIGURES_DIR, FEATURES_DIR,
    TARGET_DAILY_BARS, DMA_PRIMARY_CONFIG
)

from afmlkit.feature.core.ma import ewma, sma

# ============================================================
# 数据加载
# ============================================================


def load_dollar_bars(bars_path: str) -> pd.DataFrame:
    """加载 Dollar Bars。"""
    bars = pd.read_parquet(bars_path)
    print(f"  加载 Dollar Bars: {len(bars)} bars")
    print(f"  时间范围: {bars.index.min()} ~ {bars.index.max()}")
    return bars


def load_cusum_events(events_path: str) -> pd.DataFrame:
    """加载 CUSUM 事件点。"""
    events = pd.read_parquet(events_path)
    if 'timestamp' in events.columns:
        events = events.set_index('timestamp')
    print(f"  加载 CUSUM 事件点: {len(events)} 个")
    return events


# ============================================================
# DMA 信号生成
# ============================================================

def generate_dma_signals(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    在 CUSUM 事件点上生成 Dual Moving Average 方向信号。

    规则:
    - EMA(fast) > EMA(slow) → 多头 (Side = 1)
    - EMA(fast) < EMA(slow) → 空头 (Side = -1)

    :param bars: Dollar Bars DataFrame
    :param events: CUSUM 事件点 DataFrame
    :param config: DMA 配置
    :returns: DMA 信号 DataFrame
    """
    close = bars['close'].values.astype(np.float64)

    fast_window = config.get('fast_window', 5)
    slow_window = config.get('slow_window', 20)
    ma_type = config.get('ma_type', 'ewma')

    # 计算双均线
    if ma_type == 'ewma':
        ema_fast = ewma(close, fast_window)
        ema_slow = ewma(close, slow_window)
        print(f"  EMA: fast={fast_window}, slow={slow_window}")
    else:
        ema_fast = sma(close, fast_window)
        ema_slow = sma(close, slow_window)
        print(f"  SMA: fast={fast_window}, slow={slow_window}")

    # 定位事件点在 bars 中的索引
    event_indices = [bars.index.get_loc(ts) for ts in events.index]
    event_indices = np.array(event_indices, dtype=np.int64)

    # 在事件点上提取均线值和价差
    signals = events.copy()
    signals['ema_fast'] = ema_fast[event_indices]
    signals['ema_slow'] = ema_slow[event_indices]
    signals['dma_spread'] = signals['ema_fast'] - signals['ema_slow']

    # 方向判定
    signals['side'] = np.where(signals['dma_spread'] > 0, 1, -1).astype(np.int8)

    # 统计
    n_long = int((signals['side'] == 1).sum())
    n_short = int((signals['side'] == -1).sum())
    total = len(signals)
    print(f"  DMA 信号生成完成")
    print(f"  总信号数: {total}")
    print(f"  多头信号: {n_long} ({n_long / total * 100:.1f}%)")
    print(f"  空头信号: {n_short} ({n_short / total * 100:.1f}%)")

    return signals


# ============================================================
# Main
# ============================================================

def main():
    """AL9999 DMA Primary Model 信号生成主流程。"""
    print("=" * 70)
    print("  AL9999 Phase 4: DMA Primary Model Signal Generation")
    print("=" * 70)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载数据
    print("\n[Step 1] 加载数据...")
    bars_path = os.path.join(BARS_DIR, f'dollar_bars_target{TARGET_DAILY_BARS}.parquet')
    bars = load_dollar_bars(bars_path)

    events_path = os.path.join(FEATURES_DIR, 'cusum_events.parquet')
    events = load_cusum_events(events_path)

    # Step 2: DMA 信号生成
    print("\n[Step 2] 生成 DMA Primary Model 信号...")
    signals = generate_dma_signals(bars, events, DMA_PRIMARY_CONFIG)

    # Step 3: 保存输出
    print("\n[Step 3] 保存输出...")
    output_path = os.path.join(FEATURES_DIR, 'dma_signals.parquet')
    signals.to_parquet(output_path)
    print(f"  DMA 信号已保存: {output_path}")

    print("\n" + "=" * 70)
    print("  DMA Primary Model 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

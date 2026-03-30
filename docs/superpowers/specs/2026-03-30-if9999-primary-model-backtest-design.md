---
name: IF9999 Primary Model Backtest
description: Trend Scanning Primary Model 回测设计，验证 side 信号有效性
type: project
---

# IF9999 Primary Model Backtest 设计

## 概述

Phase 3.5 验证 Trend Scanning Primary Model 的 side 信号有效性。计算信号收益、统计指标，并分析 t_value 分位数表现，验证 t_value 作为样本权重的合理性。

**Why**: Primary Model 信号必须有一定预测能力，Meta Model 才能学习"什么条件下信号可靠"。回测验证是进入 Phase 4 的必要准备。

**How to apply**: 在 IF9999 策略开发中，Phase 3.5 位于 Trend Scanning 之后、Meta Labeling 之前。

## 核心决策

| 项目 | 决定 | 理由 |
|------|------|------|
| 回测精度 | 统计评估 | 验证信号本身，不考虑交易成本（Meta Model 阶段再细化） |
| 持仓周期 | t1 平仓 | 使用 Trend Scanning 定义的最优窗口长度 |
| 收益计算 | 点数收益 | 期货交易常用，每点 300 元，便于后续计算盈亏 |
| 分析维度 | t_value 分位数 | 验证高置信度信号表现更好，确认 t_value 权重合理性 |
| 脚本位置 | `04_primary_model_backtest.py` | 独立脚本，Phase 3.5 定位清晰 |

## 数据流

```
输入:
  → dollar_bars_target6.parquet (close 价格序列)
  → trend_labels.parquet (side, t1, t_value)

处理:
  1. 对每个 CUSUM 事件点计算收益:
     - entry_price = bars['close'].loc[event_timestamp]
     - exit_price = bars['close'].loc[t1]
     - pnl = side * (exit_price - entry_price)  # 点数收益

  2. 整体统计: 胜率、平均收益、盈亏比、总收益

  3. t_value 分位数分析:
     - Top 10% (高置信度)
     - 10%-50% (中等置信度)
     - Bottom 50% (低置信度)

输出:
  → 统计报告 (文本输出)
  → 04_pnl_distribution.png (收益分布直方图)
  → 04_cumulative_pnl.png (累积收益曲线)
  → 04_tvalue_vs_pnl.png (t_value vs 收益散点图)
```

## 统计指标

### 整体统计

| 指标 | 说明 |
|------|------|
| 总事件数 | 信号数量 |
| 胜率 | pnl > 0 的比例 |
| 平均收益 | pnl 平均值（点数） |
| 盈亏比 | 平均盈利 / |平均亏损| |
| 总收益 | 所有信号 pnl 累计 |
| 最大盈利 | 单笔最大 pnl |
| 最大亏损 | 单笔最小 pnl |

### t_value 分位数分析

对每个分位数组计算：
- 胜率
- 平均收益
- 盈亏比
- 事件数量

**假设验证**: Top 10% |t_value| 应有更高胜率和盈亏比。

## 脚本结构

```python
# 04_primary_model_backtest.py

# 模块:
# 1. 数据加载 - load_trend_labels(), load_dollar_bars()
# 2. 收益计算 - compute_pnl(prices, trend_df)
# 3. 统计分析 - compute_overall_stats(), compute_quantile_stats()
# 4. 可视化 - plot_pnl_distribution(), plot_cumulative_pnl(), plot_tvalue_vs_pnl()
# 5. 主流程 - main()
```

### 关键函数签名

```python
def compute_pnl(
    prices: pd.Series,
    trend_df: pd.DataFrame
) -> pd.DataFrame:
    """
    计算每个信号的点数收益。

    :param prices: Dollar Bars close 价格序列
    :param trend_df: trend_labels DataFrame (t1, t_value, side)
    :returns: DataFrame with columns [pnl, t_value, side, window_length]
    """
```

## 可视化输出

| 文件 | 内容 |
|------|------|
| `04_pnl_distribution.png` | 收益分布直方图（正/负分色） |
| `04_cumulative_pnl.png` | 累积收益曲线（按时间排序） |
| `04_tvalue_vs_pnl.png` | |t_value| vs pnl 散点图（分位数边界线） |

## 后续阶段

Phase 4: Meta-Labeling
- 输入: trend_labels + Primary Model 回测结果
- 处理: TBM 标签 + Meta Model 训练
- 输出: bet size 信号
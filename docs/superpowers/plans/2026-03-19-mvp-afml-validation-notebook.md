# MVP AFML Validation Notebook Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建一个完整的 Jupyter Notebook，实现 AFML MVP 验证流程，验证棕榈油期货趋势策略有效性。

**Architecture:** 8 个顺序执行的 Notebook Section，从数据加载到回测可视化，使用 AFMLKit 核心组件（DynamicDollarBarKit、cusum_filter、trend_scan_labels、triple_barrier）。

**Tech Stack:** Python, Jupyter Notebook, AFMLKit, pandas, numpy, matplotlib

---

## File Structure

```
notebooks/
└── MVP_AFML_Validation.ipynb    # 主 Notebook 文件
```

---

## Task 1: 创建 Notebook 结构和配置 Section

**Files:**
- Create: `notebooks/MVP_AFML_Validation.ipynb`

- [ ] **Step 1: 创建 notebooks 目录和 Notebook 文件**

创建 `notebooks/MVP_AFML_Validation.ipynb`，包含基本 metadata：

```json
{
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```

- [ ] **Step 2: 添加 Title 和 Overview Cell**

Markdown cell:

```markdown
# MVP AFML Validation Notebook

## 概述

本 Notebook 实现 AFML (Advances in Financial Machine Learning) MVP 验证流程：

1. **Dynamic Dollar Bars** - 动态成交额采样
2. **CUSUM Filter** - 事件检测
3. **Trend Scanning** - 趋势方向识别
4. **Triple Barrier Method** - 标签生成
5. **简单回测** - 策略验证

**数据源**: P9999.XDCE 棕榈油期货 1 分钟数据 (2023-2026)
```

- [ ] **Step 3: 添加 Import Cell**

Code cell:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# AFMLKit imports
from afmlkit.bar.data_model import TradesData
from afmlkit.bar.kit import DynamicDollarBarKit
from afmlkit.sampling.filters import cusum_filter
from afmlkit.feature.core.trend_scan import trend_scan_labels
from afmlkit.label.tbm import triple_barrier

# 设置
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

%matplotlib inline
```

- [ ] **Step 4: 添加 Configuration Cell**

Code cell:

```python
# ============================================
# 可调参数配置
# ============================================

# Dollar Bars
TARGET_DAILY_BARS = 50      # 目标每天采样 Bar 数
EWMA_SPAN = 20              # 动态阈值 EWMA 平滑周期

# CUSUM Filter
CUSUM_THRESHOLD_MULT = 2.0  # 阈值 = sigma * multiplier
VOL_SPAN = 20               # 波动率 EWMA 平滑周期

# Trend Scanning
L_WINDOWS = [10, 20, 30, 50, 100]  # 候选窗口长度

# Triple Barrier
STOP_LOSS_MULT = 1.5        # 止损 = volatility * STOP_LOSS_MULT
TAKE_PROFIT_MULT = 1.5      # 止盈 = volatility * TAKE_PROFIT_MULT
MAX_HOLD_BARS = 50          # 最大持仓 Bar 数
MIN_CLOSE_TIME_SEC = 0.0    # 最小持仓时间（Bar 数据设为 0）
MIN_RET = 0.0               # 最小收益率阈值（不过滤）

# 数据路径
DATA_PATH = Path('../data/csv/P9999.XDCE-2023-1-1-To-2026-03-11-1m.csv')
```

---

## Task 2: 数据加载 Section

**Files:**
- Modify: `notebooks/MVP_AFML_Validation.ipynb`

- [ ] **Step 1: 添加 Section Header**

Markdown cell:

```markdown
## 1. 数据加载

加载棕榈油期货 1 分钟 OHLCV 数据，进行基础预处理。
```

- [ ] **Step 2: 添加数据加载 Cell**

Code cell:

```python
# 加载数据
df = pd.read_csv(DATA_PATH, parse_dates=['datetime'], index_col='datetime')

# 检查数据
print(f"数据形状: {df.shape}")
print(f"时间范围: {df.index.min()} 至 {df.index.max()}")
print(f"\n列名: {df.columns.tolist()}")
print(f"\n前 5 行:")
display(df.head())
print(f"\n数据类型:")
print(df.dtypes)
```

- [ ] **Step 3: 添加数据质量检查 Cell**

Code cell:

```python
# 数据质量检查
print("=== 数据质量检查 ===")
print(f"缺失值:\n{df.isnull().sum()}")
print(f"\n重复时间戳: {df.index.duplicated().sum()}")
print(f"\n基础统计:")
display(df.describe())

# 计算 Dollar Volume
df['dollar_volume'] = df['close'] * df['volume']
print(f"\nDollar Volume 统计:")
print(df['dollar_volume'].describe())
```

---

## Task 3: Dynamic Dollar Bars Section

**Files:**
- Modify: `notebooks/MVP_AFML_Validation.ipynb`

- [ ] **Step 1: 添加 Section Header**

Markdown cell:

```markdown
## 2. Dynamic Dollar Bars

将 1 分钟 K 线转换为 Dollar Bars，使用动态阈值实现每天约 50 根 Bar。
```

- [ ] **Step 2: 添加 TradesData 转换 Cell**

Code cell:

```python
# 将 1min OHLCV 转换为伪 Tick 数据
# 每根 1min K 线视为一个 "tick"，成交额为 close * volume

timestamps = df.index.astype(np.int64).values  # 纳秒时间戳
prices = df['close'].values
volumes = df['volume'].values

# 创建 TradesData 实例
trades_data = TradesData(
    ts=timestamps,
    px=prices,
    qty=volumes
)

print(f"伪 Tick 数据: {len(trades_data.data)} 条")
print(f"时间范围: {pd.to_datetime(trades_data.data['timestamp'].iloc[0], unit='ns')} 至 {pd.to_datetime(trades_data.data['timestamp'].iloc[-1], unit='ns')}")
```

- [ ] **Step 3: 添加 Dynamic Dollar Bar 构建 Cell**

Code cell:

```python
# 构建 Dynamic Dollar Bars
dollar_bar_builder = DynamicDollarBarKit(
    trades=trades_data,
    target_daily_bars=TARGET_DAILY_BARS,
    ewma_span=EWMA_SPAN
)

# 获取 Dollar Bars DataFrame (使用 build_ohlcv 方法)
dollar_bars = dollar_bar_builder.build_ohlcv()

print(f"Dollar Bars 数量: {len(dollar_bars)}")
print(f"\nDollar Bars 样本:")
display(dollar_bars.head(10))
```

- [ ] **Step 4: 添加 Dollar Bars 验证 Cell**

Code cell:

```python
# 验证 Dollar Bars 频率
total_days = (dollar_bars.index[-1] - dollar_bars.index[0]).days
avg_bars_per_day = len(dollar_bars) / total_days if total_days > 0 else 0

print(f"=== Dollar Bars 验证 ===")
print(f"总 Bar 数: {len(dollar_bars)}")
print(f"总交易日: {total_days}")
print(f"平均每天 Bar 数: {avg_bars_per_day:.1f}")
print(f"目标每天 Bar 数: {TARGET_DAILY_BARS}")
print(f"偏差: {abs(avg_bars_per_day - TARGET_DAILY_BARS) / TARGET_DAILY_BARS * 100:.1f}%")

# 按日期统计 Bar 数
bars_per_day = dollar_bars.groupby(dollar_bars.index.date).size()
print(f"\n每日 Bar 数统计:")
print(f"  最小: {bars_per_day.min()}")
print(f"  最大: {bars_per_day.max()}")
print(f"  中位数: {bars_per_day.median():.0f}")
```

---

## Task 4: CUSUM Filter Section

**Files:**
- Modify: `notebooks/MVP_AFML_Validation.ipynb`

- [ ] **Step 1: 添加 Section Header**

Markdown cell:

```markdown
## 3. CUSUM Filter

在 Dollar Bars 收益率序列上应用 CUSUM Filter，检测结构性突变事件。
```

- [ ] **Step 2: 添加收益率和波动率计算 Cell**

Code cell:

```python
# 计算 log returns
close = dollar_bars['close']
log_returns = np.log(close).diff().dropna()

# 计算 EWMA 波动率
volatility = log_returns.ewm(span=VOL_SPAN).std()

print(f"收益率统计:")
print(f"  均值: {log_returns.mean():.6f}")
print(f"  标准差: {log_returns.std():.6f}")
print(f"\n波动率统计:")
print(f"  均值: {volatility.mean():.6f}")
print(f"  最大: {volatility.max():.6f}")
```

- [ ] **Step 3: 添加 CUSUM Filter 执行 Cell**

Code cell:

```python
# 准备 CUSUM 输入
diff_series = log_returns.values.astype(np.float64)
threshold = (volatility.values * CUSUM_THRESHOLD_MULT).astype(np.float64)

# 应用 CUSUM Filter
event_indices = cusum_filter(diff_series, threshold)

# 转换为 DatetimeIndex
event_times = log_returns.index[event_indices]

print(f"=== CUSUM Filter 结果 ===")
print(f"检测到事件数: {len(event_indices)}")
print(f"事件占比: {len(event_indices) / len(log_returns) * 100:.1f}%")
print(f"\n事件时间样例:")
for i, t in enumerate(event_times[:5]):
    print(f"  {i+1}. {t}")
```

- [ ] **Step 4: 添加事件分布可视化 Cell**

Code cell:

```python
# 可视化事件在价格序列上的分布
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(dollar_bars.index, dollar_bars['close'], 'b-', alpha=0.7, label='Close Price')
ax.scatter(event_times, dollar_bars.loc[event_times, 'close'],
           c='red', s=20, alpha=0.6, label='CUSUM Events')

ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.set_title(f'Dollar Bars Close Price with CUSUM Events (Total: {len(event_indices)})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Task 5: Trend Scanning Section

**Files:**
- Modify: `notebooks/MVP_AFML_Validation.ipynb`

- [ ] **Step 1: 添加 Section Header**

Markdown cell:

```markdown
## 4. Trend Scanning

对每个 CUSUM 事件点执行 Trend Scanning，确定趋势方向（side）和置信度（t-value）。
```

- [ ] **Step 2: 添加 Trend Scanning 执行 Cell**

Code cell:

```python
# 准备输入数据
price_series = dollar_bars['close'].copy()

# 执行 Trend Scanning
trend_df = trend_scan_labels(
    price_series=price_series,
    t_events=event_times,
    L_windows=L_WINDOWS
)

print(f"=== Trend Scanning 结果 ===")
print(f"成功处理事件数: {len(trend_df)}")
print(f"\n结果样例:")
display(trend_df.head(10))
```

- [ ] **Step 3: 添加 Trend Scanning 统计 Cell**

Code cell:

```python
# 统计分析
n_up = (trend_df['side'] == 1).sum()
n_down = (trend_df['side'] == -1).sum()
n_flat = (trend_df['side'] == 0).sum()

print(f"=== Trend Scanning 统计 ===")
print(f"看涨信号 (side=+1): {n_up} ({n_up/len(trend_df)*100:.1f}%)")
print(f"看跌信号 (side=-1): {n_down} ({n_down/len(trend_df)*100:.1f}%)")
print(f"无趋势 (side=0): {n_flat} ({n_flat/len(trend_df)*100:.1f}%)")
print(f"\nt-value 分布:")
print(f"  均值: {trend_df['t_value'].mean():.2f}")
print(f"  标准差: {trend_df['t_value'].std():.2f}")
print(f"  最大绝对值: {trend_df['t_value'].abs().max():.2f}")
print(f"\n最优窗口分布:")
print(trend_df['t1'].value_counts().sort_index())
```

---

## Task 6: Triple Barrier Method Section

**Files:**
- Modify: `notebooks/MVP_AFML_Validation.ipynb`

- [ ] **Step 1: 添加 Section Header**

Markdown cell:

```markdown
## 5. Triple Barrier Method

使用动态波动率倍数设置止盈止损，生成交易标签。
```

- [ ] **Step 2: 添加 TBM 参数准备 Cell**

Code cell:

```python
# 计算平均 Bar 持续时间（秒）
bar_timestamps = dollar_bars.index.astype(np.int64).values
avg_bar_duration_sec = np.mean(np.diff(bar_timestamps)) / 1e9

# 计算每个事件点的波动率 target
# 需要将事件时间映射到 volatility Series 的位置
volatility_targets = volatility.loc[trend_df.index].values.astype(np.float64)

# 准备事件索引（在 close 数组中的位置）
event_positions = dollar_bars.index.get_indexer(trend_df.index)

print(f"=== TBM 参数 ===")
print(f"平均 Bar 持续时间: {avg_bar_duration_sec:.1f} 秒")
print(f"止损倍数: {STOP_LOSS_MULT}")
print(f"止盈倍数: {TAKE_PROFIT_MULT}")
print(f"最大持仓时间: {MAX_HOLD_BARS} 根 Bar ({MAX_HOLD_BARS * avg_bar_duration_sec:.0f} 秒)")
print(f"\n波动率 Target 统计:")
print(f"  均值: {np.nanmean(volatility_targets):.6f}")
print(f"  最小: {np.nanmin(volatility_targets):.6f}")
print(f"  最大: {np.nanmax(volatility_targets):.6f}")
```

- [ ] **Step 3: 添加 TBM 执行 Cell**

Code cell:

```python
# 执行 Triple Barrier Method
labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
    timestamps=bar_timestamps,
    close=dollar_bars['close'].values.astype(np.float64),
    event_idxs=event_positions.astype(np.int64),
    targets=volatility_targets,
    horizontal_barriers=(STOP_LOSS_MULT, TAKE_PROFIT_MULT),
    vertical_barrier=MAX_HOLD_BARS * avg_bar_duration_sec,
    min_close_time_sec=MIN_CLOSE_TIME_SEC,
    side=None,  # Side prediction 模式
    min_ret=MIN_RET
)

# 创建结果 DataFrame
tbm_df = pd.DataFrame({
    'label': labels,
    'touch_idx': touch_idxs,
    'return': rets,
    'max_rb_ratio': max_rb_ratios
}, index=trend_df.index)

print(f"=== Triple Barrier Method 结果 ===")
print(f"处理事件数: {len(tbm_df)}")
print(f"\n结果样例:")
display(tbm_df.head(10))
```

- [ ] **Step 4: 添加 TBM 统计 Cell**

Code cell:

```python
# 标签分布统计
n_profit = (tbm_df['label'] == 1).sum()
n_loss = (tbm_df['label'] == -1).sum()
n_timeout = (tbm_df['label'] == 0).sum()

print(f"=== TBM 标签分布 ===")
print(f"止盈 (+1): {n_profit} ({n_profit/len(tbm_df)*100:.1f}%)")
print(f"止损 (-1): {n_loss} ({n_loss/len(tbm_df)*100:.1f}%)")
print(f"时间到期 (0): {n_timeout} ({n_timeout/len(tbm_df)*100:.1f}%)")

# 持仓时间统计
hold_times = []
for idx, row in tbm_df.iterrows():
    event_idx = event_positions[tbm_df.index.get_loc(idx)]
    touch_idx = row['touch_idx']
    if touch_idx >= 0 and touch_idx < len(dollar_bars):
        start_time = dollar_bars.index[event_idx]
        end_time = dollar_bars.index[touch_idx]
        hold_times.append((end_time - start_time).total_seconds() / 3600)  # 小时

print(f"\n持仓时间统计 (小时):")
print(f"  平均: {np.mean(hold_times):.2f}")
print(f"  中位数: {np.median(hold_times):.2f}")
print(f"  最长: {np.max(hold_times):.2f}")
```

---

## Task 7: 简单回测 Section

**Files:**
- Modify: `notebooks/MVP_AFML_Validation.ipynb`

- [ ] **Step 1: 添加 Section Header**

Markdown cell:

```markdown
## 6. 简单回测

基于 Trend Scanning 信号和 TBM 标签进行简单策略回测。
```

- [ ] **Step 2: 添加交易收益计算 Cell**

Code cell:

```python
# 合并 Trend Scanning 和 TBM 结果
# trend_df 列: ['t1', 't_value', 'side']
# tbm_df 列: ['label', 'touch_idx', 'return', 'max_rb_ratio']
results_df = pd.concat([trend_df, tbm_df], axis=1)
# 最终列: ['t1', 't_value', 'side', 'label', 'touch_idx', 'return', 'max_rb_ratio']

# 计算策略收益
# 每笔交易收益 = side * return
# side: +1 做多, -1 做空
# return: TBM 计算的收益率（已考虑方向）
results_df['trade_return'] = results_df['side'] * results_df['return']

print(f"=== 交易结果 ===")
print(f"总交易数: {len(results_df)}")
print(f"\n交易收益统计:")
print(f"  均值: {results_df['trade_return'].mean():.6f}")
print(f"  标准差: {results_df['trade_return'].std():.6f}")
print(f"  最大: {results_df['trade_return'].max():.6f}")
print(f"  最小: {results_df['trade_return'].min():.6f}")

# 胜率计算
winning_trades = (results_df['trade_return'] > 0).sum()
losing_trades = (results_df['trade_return'] < 0).sum()
win_rate = winning_trades / len(results_df) * 100

print(f"\n胜率: {win_rate:.1f}% ({winning_trades}/{len(results_df)})")
```

- [ ] **Step 3: 添加累计收益曲线 Cell**

Code cell:

```python
# 计算累计收益
cumulative_returns = (1 + results_df['trade_return']).cumprod()

# 计算最大回撤
running_max = cumulative_returns.expanding().max()
drawdown = (cumulative_returns - running_max) / running_max
max_drawdown = drawdown.min()

# 计算 Sharpe Ratio
# 年化因子：基于实际交易频率
total_days = (results_df.index[-1] - results_df.index[0]).days
trades_per_year = len(results_df) / total_days * 252
ann_factor = np.sqrt(trades_per_year)

sharpe_ratio = results_df['trade_return'].mean() / results_df['trade_return'].std() * ann_factor

print(f"=== 策略绩效 ===")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"最大回撤: {max_drawdown:.2%}")
print(f"累计收益: {(cumulative_returns.iloc[-1] - 1):.2%}")
print(f"年化收益: {(cumulative_returns.iloc[-1] ** (252/total_days) - 1):.2%}")
```

---

## Task 8: 可视化 Section

**Files:**
- Modify: `notebooks/MVP_AFML_Validation.ipynb`

- [ ] **Step 1: 添加 Section Header**

Markdown cell:

```markdown
## 7. 可视化分析
```

- [ ] **Step 2: 添加累计收益曲线图 Cell**

Code cell:

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 累计收益曲线
ax1 = axes[0, 0]
ax1.plot(cumulative_returns.index, cumulative_returns.values, 'b-', linewidth=1.5)
ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax1.set_title('Cumulative Returns', fontsize=12)
ax1.set_xlabel('Time')
ax1.set_ylabel('Cumulative Return')
ax1.grid(True, alpha=0.3)

# 2. 回撤曲线
ax2 = axes[0, 1]
ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
ax2.plot(drawdown.index, drawdown.values, 'r-', linewidth=1)
ax2.set_title('Drawdown', fontsize=12)
ax2.set_xlabel('Time')
ax2.set_ylabel('Drawdown')
ax2.grid(True, alpha=0.3)

# 3. 标签分布
ax3 = axes[1, 0]
label_counts = tbm_df['label'].value_counts()
colors = ['green' if x == 1 else 'red' if x == -1 else 'gray' for x in label_counts.index]
ax3.bar(['Profit (+1)', 'Loss (-1)', 'Timeout (0)'] if len(label_counts) == 3 else label_counts.index,
        label_counts.values, color=colors)
ax3.set_title('TBM Label Distribution', fontsize=12)
ax3.set_ylabel('Count')

# 4. 持仓时间分布
ax4 = axes[1, 1]
ax4.hist(hold_times, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
ax4.axvline(x=np.mean(hold_times), color='red', linestyle='--', label=f'Mean: {np.mean(hold_times):.1f}h')
ax4.set_title('Holding Time Distribution', fontsize=12)
ax4.set_xlabel('Holding Time (hours)')
ax4.set_ylabel('Frequency')
ax4.legend()

plt.tight_layout()
plt.savefig('mvp_validation_results.png', dpi=150, bbox_inches='tight')
plt.show()
```

- [ ] **Step 3: 添加结果汇总表格 Cell**

Code cell:

```python
# 输出最终汇总表格
print("=" * 60)
print("            MVP VALIDATION RESULTS")
print("=" * 60)
print(f"{'Data Period:':<25} {dollar_bars.index[0].date()} to {dollar_bars.index[-1].date()}")
print(f"{'Dollar Bars:':<25} {len(dollar_bars):,} (avg {avg_bars_per_day:.1f}/day)")
print(f"{'CUSUM Events:':<25} {len(event_indices):,}")
print(f"{'Trend Scan Signals:':<25} {n_up} up / {n_down} down")
print(f"{'TBM Labels:':<25} +1:{n_profit}, -1:{n_loss}, 0:{n_timeout}")
print("-" * 60)
print(f"{'Strategy Sharpe:':<25} {sharpe_ratio:.2f}")
print(f"{'Max Drawdown:':<25} {max_drawdown:.2%}")
print(f"{'Win Rate:':<25} {win_rate:.1f}%")
print(f"{'Total Trades:':<25} {len(results_df):,}")
print("=" * 60)

# 成功判定
success = True
issues = []

if sharpe_ratio < 1.0:
    success = False
    issues.append("Sharpe Ratio < 1.0")
if abs(max_drawdown) > 0.30:
    success = False
    issues.append("Max Drawdown > 30%")
if win_rate < 45:
    success = False
    issues.append("Win Rate < 45%")

if success:
    print("\n✅ MVP 验证通过！趋势策略在该标的上显示有效性。")
    print("   建议下一步：添加 FracDiff、Meta-Labeling、Purged CV 等高级组件。")
else:
    print(f"\n❌ MVP 验证未通过。问题：{', '.join(issues)}")
    print("   建议：检查参数配置，或尝试其他标的/时间段。")
```

---

## Task 9: 最终提交

**Files:**
- Modify: `notebooks/MVP_AFML_Validation.ipynb`

- [ ] **Step 1: 运行完整 Notebook 验证**

在 Jupyter 环境中运行整个 Notebook，确保所有 Cell 正确执行。

- [ ] **Step 2: 提交 Notebook 文件**

```bash
git add notebooks/MVP_AFML_Validation.ipynb
git commit -m "feat: add MVP AFML validation notebook for P9999.XDCE"
```

---

## Verification Checklist

完成实现后，验证以下内容：

- [ ] Notebook 可以从头到尾顺序执行，无错误
- [ ] Dollar Bars 平均每天约 50 根
- [ ] CUSUM 事件数量合理（非过度密集或稀疏）
- [ ] Trend Scanning 输出 up/down 信号
- [ ] TBM 标签分布三类都有样本
- [ ] 策略指标正确计算（Sharpe、MaxDD、Win Rate）
- [ ] 可视化图表正确生成
- [ ] 最终汇总表格输出完整
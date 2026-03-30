# IF9999 Primary Model Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 Primary Model 回测脚本，验证 Trend Scanning side 信号有效性。

**Architecture:** 独立脚本读取 trend_labels 和 dollar_bars，计算每个信号的点数收益，输出统计报告和 3 张可视化图表。

**Tech Stack:** Python, Pandas, NumPy, Matplotlib, afmlkit

---

## File Structure

| 文件 | 操作 | 责任 |
|------|------|------|
| `strategies/IF9999/04_primary_model_backtest.py` | Create | 回测主脚本 |
| `strategies/IF9999/README.md` | Modify | 添加 Phase 3.5 文档 |

---

### Task 1: 创建脚本骨架和数据加载模块

**Files:**
- Create: `strategies/IF9999/04_primary_model_backtest.py`

- [ ] **Step 1: 创建脚本骨架和导入部分**

```python
"""
04_primary_model_backtest.py - IF9999 Phase 3.5 Primary Model 回测

流程:
1. 加载 Dollar Bars（价格序列）
2. 加载 Trend Labels（side, t1, t_value）
3. 计算每个信号的点数收益
4. 统计分析（整体 + t_value 分位数）
5. 生成可视化图表

输出:
  - 统计报告（文本）
  - 04_pnl_distribution.png: 收益分布直方图
  - 04_cumulative_pnl.png: 累积收益曲线
  - 04_tvalue_vs_pnl.png: |t_value| vs pnl 散点图
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
    BARS_DIR, FIGURES_DIR, FEATURES_DIR
)

sns.set_theme(style="whitegrid", context="paper")
```

- [ ] **Step 2: 添加数据加载函数**

```python
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


def load_trend_labels(labels_path: str) -> pd.DataFrame:
    """
    加载 Trend Labels 数据。

    :param labels_path: parquet 文件路径
    :returns: DataFrame with columns [t1, t_value, side]
    """
    labels = pd.read_parquet(labels_path)
    print(f"✅ 加载 Trend Labels: {len(labels)} 个信号")
    print(f"   时间范围: {labels.index.min()} ~ {labels.index.max()}")
    print(f"   Side 分布: +1={(labels['side']==1).sum()}, -1={(labels['side']==-1).sum()}")
    return labels
```

- [ ] **Step 3: 验证数据加载模块**

```bash
cd /Users/link/Documents/AFMLKIT
uv run python -c "
from strategies.IF9999.config import BARS_DIR, FEATURES_DIR
import pandas as pd

bars = pd.read_parquet(f'{BARS_DIR}/dollar_bars_target6.parquet')
labels = pd.read_parquet(f'{FEATURES_DIR}/trend_labels.parquet')

print(f'Bars columns: {list(bars.columns)}')
print(f'Labels columns: {list(labels.columns)}')
print(f'Bars index type: {type(bars.index).__name__}')
print(f'Labels index type: {type(labels.index).__name__}')
"
```

Expected: 看到数据结构正确

- [ ] **Step 4: Commit**

```bash
git add strategies/IF9999/04_primary_model_backtest.py
git commit -m "feat(IF9999): add Primary Model backtest script skeleton

- Add data loading functions for Dollar Bars and Trend Labels
- Define visualization and analysis modules

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: 实现收益计算模块

**Files:**
- Modify: `strategies/IF9999/04_primary_model_backtest.py`

- [ ] **Step 1: 添加收益计算函数**

```python
# ============================================================
# 收益计算
# ============================================================

def compute_pnl(
    prices: pd.Series,
    trend_df: pd.DataFrame
) -> pd.DataFrame:
    """
    计算每个信号的点数收益。

    :param prices: Dollar Bars close 价格序列
    :param trend_df: Trend Labels DataFrame (t1, t_value, side)
    :returns: DataFrame with columns [pnl, t_value, side, window_length, entry_price, exit_price]
    """
    results = []

    for event_ts, row in trend_df.iterrows():
        t1 = row['t1']
        side = row['side']
        t_value = row['t_value']

        # 入场价格（事件点 close）
        try:
            entry_price = prices.loc[event_ts]
        except KeyError:
            # 事件点可能在 bars 中不存在（边界情况）
            continue

        # 出场价格（t1 时间点 close）
        try:
            exit_price = prices.loc[t1]
        except KeyError:
            # t1 可能超出 bars 范围，取最近的
            if t1 > prices.index.max():
                exit_price = prices.iloc[-1]
            else:
                # 找 t1 之前最近的 bar
                valid_idx = prices.index[prices.index <= t1]
                if len(valid_idx) == 0:
                    continue
                exit_price = prices.loc[valid_idx[-1]]

        # 点数收益 = side * (exit - entry)
        pnl = side * (exit_price - entry_price)

        # 窗口长度（bars 数量）
        window_bars = len(prices.loc[event_ts:t1]) - 1  # 排除起点

        results.append({
            'timestamp': event_ts,
            'pnl': pnl,
            't_value': t_value,
            'side': side,
            'window_length': window_bars,
            'entry_price': entry_price,
            'exit_price': exit_price
        })

    pnl_df = pd.DataFrame(results)
    pnl_df = pnl_df.set_index('timestamp')

    print(f"\n✅ 收益计算完成: {len(pnl_df)} 个有效信号")
    print(f"   总收益: {pnl_df['pnl'].sum():.2f} 点")
    print(f"   平均收益: {pnl_df['pnl'].mean():.2f} 点")

    return pnl_df
```

- [ ] **Step 2: 运行脚本验证收益计算**

在 main() 函数骨架后添加临时测试：

```python
def main():
    """临时验证"""
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target6.parquet')
    bars = load_dollar_bars(bars_path)
    prices = bars['close']

    labels_path = os.path.join(FEATURES_DIR, 'trend_labels.parquet')
    trend_df = load_trend_labels(labels_path)

    pnl_df = compute_pnl(prices, trend_df)
    print(f"\n样本数据:")
    print(pnl_df.head())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行验证**

```bash
cd /Users/link/Documents/AFMLKIT
uv run python strategies/IF9999/04_primary_model_backtest.py
```

Expected: 看到 pnl_df 输出，包含 pnl, t_value, side 等列

- [ ] **Step 4: Commit**

```bash
git add strategies/IF9999/04_primary_model_backtest.py
git commit -m "feat(IF9999): implement PnL computation for Primary Model

- Compute point-based returns using side * (exit - entry)
- Handle edge cases for missing timestamps

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: 实现统计分析模块

**Files:**
- Modify: `strategies/IF9999/04_primary_model_backtest.py`

- [ ] **Step 1: 添加整体统计函数**

```python
# ============================================================
# 统计分析
# ============================================================

def compute_overall_stats(pnl_df: pd.DataFrame) -> dict:
    """
    计算整体统计指标。

    :param pnl_df: 收益 DataFrame
    :returns: 统计指标字典
    """
    n_total = len(pnl_df)
    n_win = int((pnl_df['pnl'] > 0).sum())
    n_loss = int((pnl_df['pnl'] < 0).sum())
    n_flat = int((pnl_df['pnl'] == 0).sum())

    win_rate = n_win / n_total if n_total > 0 else 0
    avg_pnl = pnl_df['pnl'].mean()
    total_pnl = pnl_df['pnl'].sum()

    # 盈亏比
    wins = pnl_df[pnl_df['pnl'] > 0]['pnl']
    losses = pnl_df[pnl_df['pnl'] < 0]['pnl']
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    max_win = pnl_df['pnl'].max()
    max_loss = pnl_df['pnl'].min()

    stats = {
        'n_total': n_total,
        'n_win': n_win,
        'n_loss': n_loss,
        'n_flat': n_flat,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'max_win': max_win,
        'max_loss': max_loss
    }

    return stats


def print_overall_stats(stats: dict):
    """
    打印整体统计报告。

    :param stats: 统计指标字典
    """
    print("\n" + "=" * 70)
    print("  Primary Model Backtest - 整体统计")
    print("=" * 70)
    print(f"总信号数: {stats['n_total']}")
    print(f"盈利信号: {stats['n_win']} ({stats['win_rate']*100:.1f}%)")
    print(f"亏损信号: {stats['n_loss']} ({stats['n_loss']/stats['n_total']*100:.1f}%)")
    print(f"持平信号: {stats['n_flat']}")
    print("-" * 70)
    print(f"胜率: {stats['win_rate']*100:.1f}%")
    print(f"平均收益: {stats['avg_pnl']:.2f} 点")
    print(f"总收益: {stats['total_pnl']:.2f} 点")
    print(f"盈亏比: {stats['profit_loss_ratio']:.2f}")
    print(f"平均盈利: {stats['avg_win']:.2f} 点")
    print(f"平均亏损: {stats['avg_loss']:.2f} 点")
    print(f"最大盈利: {stats['max_win']:.2f} 点")
    print(f"最大亏损: {stats['max_loss']:.2f} 点")
    print("=" * 70)
```

- [ ] **Step 2: 添加 t_value 分位数分析函数**

```python
def compute_quantile_stats(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 t_value 分位数分析。

    分为三组:
      - Top 10%: 高置信度信号
      - 10%-50%: 中等置信度
      - Bottom 50%: 低置信度

    :param pnl_df: 收益 DataFrame
    :returns: 分位数统计 DataFrame
    """
    t_abs = pnl_df['t_value'].abs()

    # 分位数边界
    top10_thresh = t_abs.quantile(0.90)
    mid_thresh = t_abs.quantile(0.50)

    # 分组
    pnl_df['t_group'] = pd.cut(
        t_abs,
        bins=[0, mid_thresh, top10_thresh, t_abs.max() + 1],
        labels=['Bottom 50%', '10%-50%', 'Top 10%']
    )

    # 计算每组统计
    groups = ['Top 10%', '10%-50%', 'Bottom 50%']
    results = []

    for group in groups:
        group_df = pnl_df[pnl_df['t_group'] == group]
        if len(group_df) == 0:
            continue

        n = len(group_df)
        n_win = int((group_df['pnl'] > 0).sum())
        win_rate = n_win / n

        avg_pnl = group_df['pnl'].mean()
        total_pnl = group_df['pnl'].sum()

        wins = group_df[group_df['pnl'] > 0]['pnl']
        losses = group_df[group_df['pnl'] < 0]['pnl']
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        results.append({
            'group': group,
            'n': n,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'pl_ratio': pl_ratio
        })

    quantile_stats = pd.DataFrame(results)
    return quantile_stats, top10_thresh, mid_thresh


def print_quantile_stats(quantile_stats: pd.DataFrame, top_thresh: float, mid_thresh: float):
    """
    打印分位数统计报告。

    :param quantile_stats: 分位数统计 DataFrame
    :param top_thresh: Top 10% 阈值
    :param mid_thresh: 50% 阈值
    """
    print("\n" + "=" * 70)
    print("  t_value 分位数分析")
    print("=" * 70)
    print(f"Top 10% 阈值: |t| > {top_thresh:.2f}")
    print(f"50% 阈值: |t| > {mid_thresh:.2f}")
    print("-" * 70)
    print(f"{'分组':<12} {'数量':>6} {'胜率':>8} {'平均收益':>10} {'盈亏比':>8}")
    print("-" * 70)

    for _, row in quantile_stats.iterrows():
        print(f"{row['group']:<12} {row['n']:>6} {row['win_rate']*100:>7.1f}% {row['avg_pnl']:>9.2f}点 {row['pl_ratio']:>8.2f}")

    print("=" * 70)
    print("\n验证假设: Top 10% |t_value| 应有更高胜率和盈亏比")
```

- [ ] **Step 3: 更新 main() 函数测试统计模块**

```python
def main():
    """IF9999 Primary Model Backtest 主流程."""
    print("=" * 70)
    print("  IF9999 Phase 3.5 Primary Model Backtest")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载数据
    print("\n[Step 1] 加载数据...")
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target6.parquet')
    bars = load_dollar_bars(bars_path)
    prices = bars['close']

    labels_path = os.path.join(FEATURES_DIR, 'trend_labels.parquet')
    trend_df = load_trend_labels(labels_path)

    # Step 2: 计算收益
    print("\n[Step 2] 计算收益...")
    pnl_df = compute_pnl(prices, trend_df)

    # Step 3: 整体统计
    print("\n[Step 3] 整体统计分析...")
    stats = compute_overall_stats(pnl_df)
    print_overall_stats(stats)

    # Step 4: 分位数分析
    print("\n[Step 4] t_value 分位数分析...")
    quantile_stats, top_thresh, mid_thresh = compute_quantile_stats(pnl_df)
    print_quantile_stats(quantile_stats, top_thresh, mid_thresh)

    # Step 5: 可视化（待实现）
    print("\n[Step 5] 生成可视化图表...")
    print("  (将在 Task 4 实现)")

    print("\n完成！")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行验证**

```bash
cd /Users/link/Documents/AFMLKIT
uv run python strategies/IF9999/04_primary_model_backtest.py
```

Expected: 看到完整的统计报告输出

- [ ] **Step 5: Commit**

```bash
git add strategies/IF9999/04_primary_model_backtest.py
git commit -m "feat(IF9999): implement statistics analysis for Primary Model

- Overall stats: win rate, avg pnl, profit/loss ratio
- t-value quantile analysis: Top 10%, 10-50%, Bottom 50%

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: 实现可视化模块

**Files:**
- Modify: `strategies/IF9999/04_primary_model_backtest.py`

- [ ] **Step 1: 添加收益分布直方图函数**

```python
# ============================================================
# 可视化模块
# ============================================================

def plot_pnl_distribution(pnl_df: pd.DataFrame, save_path: str):
    """
    绘制收益分布直方图。

    :param pnl_df: 收益 DataFrame
    :param save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 分离正负收益
    wins = pnl_df[pnl_df['pnl'] > 0]['pnl']
    losses = pnl_df[pnl_df['pnl'] < 0]['pnl']

    # 绘制直方图（分色）
    bins = np.linspace(pnl_df['pnl'].min(), pnl_df['pnl'].max(), 50)

    if len(losses) > 0:
        ax.hist(losses, bins=bins, color='#ff6b6b', alpha=0.7, label=f'Loss ({len(losses)})')
    if len(wins) > 0:
        ax.hist(wins, bins=bins, color='#4ecdc4', alpha=0.7, label=f'Win ({len(wins)})')

    # 标注
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(pnl_df['pnl'].mean(), color='blue', linestyle='-', linewidth=2,
                label=f'Mean: {pnl_df["pnl"].mean():.1f}点')

    ax.set_title('PnL Distribution (Primary Model)', fontsize=12)
    ax.set_xlabel('PnL (points)')
    ax.set_ylabel('Count')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 收益分布图已保存: {save_path}")
```

- [ ] **Step 2: 添加累积收益曲线函数**

```python
def plot_cumulative_pnl(pnl_df: pd.DataFrame, save_path: str):
    """
    绘制累积收益曲线。

    :param pnl_df: 收益 DataFrame
    :param save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # 按时间排序计算累积收益
    cumulative = pnl_df['pnl'].cumsum()

    ax.plot(pnl_df.index, cumulative.values, color='steelblue', linewidth=1.5)

    # 标注关键点
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    max_cum = cumulative.max()
    min_cum = cumulative.min()
    final_cum = cumulative.iloc[-1]

    ax.scatter([pnl_df.index[-1]], [final_cum], color='red', s=100, zorder=5)
    ax.annotate(f'Final: {final_cum:.1f}点',
                xy=(pnl_df.index[-1], final_cum),
                xytext=(10, 10), textcoords='offset points')

    ax.set_title('Cumulative PnL (Primary Model)', fontsize=12)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative PnL (points)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 累积收益图已保存: {save_path}")
```

- [ ] **Step 3: 添加 t_value vs pnl 散点图函数**

```python
def plot_tvalue_vs_pnl(
    pnl_df: pd.DataFrame,
    top_thresh: float,
    mid_thresh: float,
    save_path: str
):
    """
    绘制 |t_value| vs PnL 散点图。

    :param pnl_df: 收益 DataFrame
    :param top_thresh: Top 10% 阈值
    :param mid_thresh: 50% 阈值
    :param save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    t_abs = pnl_df['t_value'].abs()

    # 分组绘制
    top_mask = t_abs > top_thresh
    mid_mask = (t_abs > mid_thresh) & (t_abs <= top_thresh)
    bottom_mask = t_abs <= mid_thresh

    # 散点图（不同颜色）
    ax.scatter(t_abs[bottom_mask], pnl_df.loc[bottom_mask, 'pnl'],
               c='#999999', alpha=0.5, s=30, label='Bottom 50%')
    ax.scatter(t_abs[mid_mask], pnl_df.loc[mid_mask, 'pnl'],
               c='#ffa500', alpha=0.5, s=30, label='10%-50%')
    ax.scatter(t_abs[top_mask], pnl_df.loc[top_mask, 'pnl'],
               c='#4ecdc4', alpha=0.7, s=40, label='Top 10%')

    # 分位数边界线
    ax.axvline(mid_thresh, color='gray', linestyle='--', linewidth=1.5,
                label=f'50% threshold: {mid_thresh:.2f}')
    ax.axvline(top_thresh, color='green', linestyle='--', linewidth=1.5,
                label=f'90% threshold: {top_thresh:.2f}')

    # 零收益线
    ax.axhline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_title('|t-value| vs PnL (Confidence Analysis)', fontsize=12)
    ax.set_xlabel('|t-value|')
    ax.set_ylabel('PnL (points)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ t_value vs PnL 图已保存: {save_path}")
```

- [ ] **Step 4: 更新 main() 函数调用可视化**

在 Step 5 部分：

```python
    # Step 5: 可视化
    print("\n[Step 5] 生成可视化图表...")
    plot_pnl_distribution(
        pnl_df,
        os.path.join(FIGURES_DIR, '04_pnl_distribution.png')
    )
    plot_cumulative_pnl(
        pnl_df,
        os.path.join(FIGURES_DIR, '04_cumulative_pnl.png')
    )
    plot_tvalue_vs_pnl(
        pnl_df, top_thresh, mid_thresh,
        os.path.join(FIGURES_DIR, '04_tvalue_vs_pnl.png')
    )
```

更新完成输出：

```python
    print("\n" + "=" * 70)
    print("  Phase 3.5 Primary Model Backtest 完成")
    print("=" * 70)
    print(f"图表目录: {FIGURES_DIR}")
    print(f"  - 04_pnl_distribution.png")
    print(f"  - 04_cumulative_pnl.png")
    print(f"  - 04_tvalue_vs_pnl.png")
```

- [ ] **Step 5: 运行完整脚本**

```bash
cd /Users/link/Documents/AFMLKIT
uv run python strategies/IF9999/04_primary_model_backtest.py
```

Expected: 看到完整统计报告 + 3 张图表生成

- [ ] **Step 6: 验证图表文件**

```bash
ls -la strategies/IF9999/output/figures/04_*.png
```

Expected: 看到三个 PNG 文件

- [ ] **Step 7: Commit**

```bash
git add strategies/IF9999/04_primary_model_backtest.py strategies/IF9999/output/figures/04_*.png
git commit -m "feat(IF9999): add Primary Model visualization

- PnL distribution histogram (win/loss colored)
- Cumulative PnL curve
- |t-value| vs PnL scatter plot with quantile thresholds

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: 更新 README 文档

**Files:**
- Modify: `strategies/IF9999/README.md`

- [ ] **Step 1: 添加 Phase 3.5 文档**

在 Phase 3 文档后添加：

```markdown
## Phase 3.5: Primary Model 回测验证

运行 Primary Model 回测验证信号有效性：

```bash
uv run python strategies/IF9999/04_primary_model_backtest.py
```

### 回测方法

| 项目 | 说明 |
|------|------|
| 收益计算 | 点数收益 = side × (close_t1 - close_entry) |
| 持仓周期 | 使用 t1 作为平仓时间（Trend Scanning 最优窗口） |
| 分析维度 | 整体统计 + t_value 分位数分析 |

### 统计指标

| 指标 | 说明 |
|------|------|
| 胜率 | 盈利信号比例 |
| 平均收益 | 所有信号平均点数收益 |
| 盈亏比 | 平均盈利 / 平均亏损 |
| 总收益 | 所有信号累计点数 |

### t_value 分位数分析

验证假设：高 |t_value| 信号应有更好的表现。

| 分组 | 阈值 |
|------|------|
| Top 10% | |t_value| > 90% 分位数 |
| 10%-50% | |t_value| > 50% 分位数 |
| Bottom 50% | |t_value| ≤ 50% 分位数 |

### 可视化输出

| 文件 | 说明 |
|------|------|
| `04_pnl_distribution.png` | 收益分布直方图 |
| `04_cumulative_pnl.png` | 累积收益曲线 |
| `04_tvalue_vs_pnl.png` | |t_value| vs 收益散点图 |
```

- [ ] **Step 2: 更新项目结构**

更新项目结构部分：

```markdown
## 项目结构

```
IF9999/
├── 01_dollar_bar_builder.py    # Dollar Bars 构建与验证
├── 02_feature_engineering.py   # Phase 2: FracDiff + CUSUM
├── 03_trend_scanning.py        # Phase 3: Trend Scanning 标签
├── 04_primary_model_backtest.py # Phase 3.5: Primary Model 回测
├── config.py                   # 配置参数
├── output/
│   ├── bars/                   # 生成的 Bar 数据
│   ├── features/               # 特征和标签输出
│   └── figures/                # 可视化图表
└── README.md
```
```

- [ ] **Step 3: Commit**

```bash
git add strategies/IF9999/README.md
git commit -m "docs(IF9999): update README with Phase 3.5 Primary Model backtest

- Add backtest methodology documentation
- Update project structure

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- ✅ 数据加载（dollar_bars + trend_labels）→ Task 1
- ✅ 收益计算（点数收益）→ Task 2
- ✅ 整体统计（胜率、盈亏比等）→ Task 3
- ✅ t_value 分位数分析 → Task 3
- ✅ 可视化（3 张图表）→ Task 4
- ✅ README 文档 → Task 5

**2. Placeholder scan:**
- ✅ 无 TBD、TODO
- ✅ 所有代码块完整
- ✅ 所有命令有预期输出

**3. Type consistency:**
- ✅ pnl_df 列名一致：pnl, t_value, side, window_length
- ✅ stats 字典键名一致
- ✅ 函数签名与调用匹配

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-03-30-if9999-primary-model-backtest-plan.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** - 我为每个 Task 派发新 subagent，Task 间 review，快速迭代

**2. Inline Execution** - 在当前 session 用 executing-plans 执行，批量执行带 checkpoint

**选择哪种方式？**
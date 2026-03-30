# IF9999 Phase 3 Trend Scanning Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 IF9999 Phase 3 Trend Scanning 标签生成脚本，输出 side 和 t_value 供 Meta Model 学习。

**Architecture:** 独立脚本调用已实现的 `afmlkit.feature.core.trend_scan.trend_scan_labels()`，读取 Phase 2 输出（Dollar Bars + CUSUM 事件），生成标签和可视化。

**Tech Stack:** Python, Pandas, NumPy, Matplotlib, afmlkit

---

## File Structure

| 文件 | 操作 | 责任 |
|------|------|------|
| `strategies/IF9999/config.py` | Modify | 添加 TREND_WINDOWS 参数 |
| `strategies/IF9999/03_trend_scanning.py` | Create | Trend Scanning 主脚本 |
| `strategies/IF9999/README.md` | Modify | 更新 Phase 3 文档 |
| `strategies/IF9999/output/features/trend_labels.parquet` | Create | 标签输出文件 |

---

### Task 1: 更新配置参数

**Files:**
- Modify: `strategies/IF9999/config.py`

- [ ] **Step 1: 添加 TREND_WINDOWS 参数到 config.py**

在 Phase 2 参数区块后添加 Phase 3 配置：

```python
# ============================================================
# Phase 3: Trend Scanning 参数
# ============================================================

# Trend Scanning 窗口范围（Bars 数量）
TREND_WINDOWS = [5, 10, 20, 30, 50]  # 短期趋势，适合期货
```

- [ ] **Step 2: 验证配置可导入**

```bash
cd /Users/link/Documents/AFMLKIT
uv run python -c "from strategies.IF9999.config import TREND_WINDOWS; print(TREND_WINDOWS)"
```

Expected: `[5, 10, 20, 30, 50]`

- [ ] **Step 3: Commit**

```bash
git add strategies/IF9999/config.py
git commit -m "feat(IF9999): add TREND_WINDOWS config for Phase 3 Trend Scanning

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: 创建脚本骨架和数据加载模块

**Files:**
- Create: `strategies/IF9999/03_trend_scanning.py`

- [ ] **Step 1: 创建脚本骨架和导入部分**

```python
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
```

- [ ] **Step 3: 验证数据加载模块**

```bash
cd /Users/link/Documents/AFMLKIT
uv run python -c "
from strategies.IF9999.config import BARS_DIR, FEATURES_DIR
import pandas as pd

# 检查输入文件是否存在
bars_path = f'{BARS_DIR}/dollar_bars_target6.parquet'
events_path = f'{FEATURES_DIR}/cusum_events.parquet'

bars = pd.read_parquet(bars_path)
events = pd.read_parquet(events_path)

print(f'Dollar Bars: {len(bars)} rows, columns: {list(bars.columns)}')
print(f'CUSUM Events: {len(events)} rows, columns: {list(events.columns)}')
print(f'Events timestamp dtype: {events[\"timestamp\"].dtype}')
"
```

Expected: 看到文件存在且格式正确

- [ ] **Step 4: Commit**

```bash
git add strategies/IF9999/03_trend_scanning.py
git commit -m "feat(IF9999): add Phase 3 Trend Scanning script skeleton

- Add data loading functions for Dollar Bars and CUSUM events
- Import trend_scan_labels from afmlkit

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: 实现 Trend Scanning 处理和输出保存

**Files:**
- Modify: `strategies/IF9999/03_trend_scanning.py`

- [ ] **Step 1: 添加 Trend Scanning 处理函数**

```python
# ============================================================
# Trend Scanning 处理
# ============================================================

def run_trend_scanning(
    prices: pd.Series,
    t_events: pd.DatetimeIndex,
    L_windows: list
) -> pd.DataFrame:
    """
    运行 Trend Scanning 标签生成。

    :param prices: 原始价格序列（close）
    :param t_events: CUSUM 事件点 DatetimeIndex
    :param L_windows: 窗口长度列表
    :returns: 标签 DataFrame (t1, t_value, side)
    """
    print("\n[Trend Scanning] 开始处理...")
    print(f"  窗口范围: {L_windows}")

    # 调用 afmlkit 的 trend_scan_labels
    trend_df = trend_scan_labels(
        price_series=prices,
        t_events=t_events,
        L_windows=L_windows
    )

    # 统计结果
    n_up = int((trend_df['side'] == 1).sum())
    n_down = int((trend_df['side'] == -1).sum())
    n_flat = int((trend_df['side'] == 0).sum())

    print(f"\n[Trend Scanning] 处理完成")
    print(f"  总事件数: {len(trend_df)}")
    print(f"  上涨趋势 (+1): {n_up} ({n_up/len(trend_df)*100:.1f}%)")
    print(f"  下跌趋势 (-1): {n_down} ({n_down/len(trend_df)*100:.1f}%)")
    print(f"  无趋势 (0): {n_flat} ({n_flat/len(trend_df)*100:.1f}%)")
    print(f"  平均 |t_value|: {trend_df['t_value'].abs().mean():.2f}")

    return trend_df
```

- [ ] **Step 2: 添加输出保存函数**

```python
# ============================================================
# 输出保存
# ============================================================

def save_outputs(trend_df: pd.DataFrame):
    """
    保存 Trend Scanning 标签。

    :param trend_df: 标签 DataFrame
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # 保存趋势标签
    labels_path = os.path.join(FEATURES_DIR, 'trend_labels.parquet')
    trend_df.to_parquet(labels_path)
    print(f"\n✅ Trend Labels 已保存: {labels_path}")
    print(f"   行数: {len(trend_df)}")
    print(f"   列: {list(trend_df.columns)}")
```

- [ ] **Step 3: 添加主函数骨架**

```python
def main():
    """
    IF9999 Phase 3 Trend Scanning 主流程。
    """
    print("=" * 70)
    print("  IF9999 Phase 3 Trend Scanning Labels")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载 Dollar Bars
    print("\n[Step 1] 加载 Dollar Bars...")
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target6.parquet')
    bars = load_dollar_bars(bars_path)
    prices = bars['close']

    # Step 2: 加载 CUSUM 事件点
    print("\n[Step 2] 加载 CUSUM 事件点...")
    events_path = os.path.join(FEATURES_DIR, 'cusum_events.parquet')
    t_events = load_cusum_events(events_path)

    # Step 3: Trend Scanning
    print("\n[Step 3] 应用 Trend Scanning...")
    trend_df = run_trend_scanning(prices, t_events, TREND_WINDOWS)

    # Step 4: 保存输出
    print("\n[Step 4] 保存输出文件...")
    save_outputs(trend_df)

    # Step 5: 可视化（待实现）
    print("\n[Step 5] 生成可视化图表...")
    print("  (将在 Task 4 实现)")

    # 完成
    print("\n" + "=" * 70)
    print("  Phase 3 Trend Scanning 完成")
    print("=" * 70)
    print(f"输出文件: {FEATURES_DIR}/trend_labels.parquet")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行脚本验证 Trend Scanning 处理**

```bash
cd /Users/link/Documents/AFMLKIT
uv run python strategies/IF9999/03_trend_scanning.py
```

Expected:
- 加载 Dollar Bars 和 CUSUM 事件成功
- Trend Scanning 处理完成，显示 up/down/flat 统计
- trend_labels.parquet 保存成功

- [ ] **Step 5: 验证输出文件格式**

```bash
cd /Users/link/Documents/AFMLKIT
uv run python -c "
import pandas as pd
trend_df = pd.read_parquet('strategies/IF9999/output/features/trend_labels.parquet')
print(f'Columns: {list(trend_df.columns)}')
print(f'Index type: {type(trend_df.index).__name__}')
print(f't1 type: {trend_df[\"t1\"].dtype}')
print(f'side unique: {trend_df[\"side\"].unique()}')
print(f'Sample:\\n{trend_df.head()}')
"
```

Expected:
- Columns: `['t1', 't_value', 'side']`
- Index: DatetimeIndex
- t1: datetime64[ns] 或 Datetime
- side: int8 values in [-1, 0, 1]

- [ ] **Step 6: Commit**

```bash
git add strategies/IF9999/03_trend_scanning.py strategies/IF9999/output/features/trend_labels.parquet
git commit -m "feat(IF9999): implement Phase 3 Trend Scanning processing

- Call trend_scan_labels() with TREND_WINDOWS [5, 10, 20, 30, 50]
- Output trend_labels.parquet with t1, t_value, side columns

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: 实现可视化模块

**Files:**
- Modify: `strategies/IF9999/03_trend_scanning.py`

- [ ] **Step 1: 添加 side 和 t_value 分布可视化函数**

```python
# ============================================================
# 可视化模块
# ============================================================

def plot_trend_distribution(trend_df: pd.DataFrame, save_path: str):
    """
    绘制 side 和 t_value 分布图。

    :param trend_df: 标签 DataFrame
    :param save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Side 分布
    side_counts = trend_df['side'].value_counts().sort_index()
    colors = ['#ff6b6b', '#gray', '#4ecdc4']  # down(-1), flat(0), up(1)
    labels = ['Down (-1)', 'Flat (0)', 'Up (+1)']

    # 确保所有 side 值都有对应颜色
    present_sides = side_counts.index.tolist()
    bar_colors = []
    bar_labels = []
    for side in present_sides:
        if side == -1:
            bar_colors.append('#ff6b6b')
            bar_labels.append('Down (-1)')
        elif side == 0:
            bar_colors.append('#999999')
            bar_labels.append('Flat (0)')
        else:
            bar_colors.append('#4ecdc4')
            bar_labels.append('Up (+1)')

    ax1.bar(range(len(side_counts)), side_counts.values, color=bar_colors)
    ax1.set_xticks(range(len(side_counts)))
    ax1.set_xticklabels(bar_labels)
    ax1.set_title('Trend Direction Distribution', fontsize=12)
    ax1.set_ylabel('Count')

    # t_value 分布（绝对值）
    t_abs = trend_df['t_value'].abs()
    ax2.hist(t_abs, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(t_abs.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean |t|={t_abs.mean():.2f}')
    ax2.axvline(t_abs.median(), color='orange', linestyle='--', linewidth=1.5,
                label=f'Median |t|={t_abs.median():.2f}')
    ax2.set_title('|t-value| Distribution (Sample Weight)', fontsize=12)
    ax2.set_xlabel('|t-value|')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 趋势分布图已保存: {save_path}")


def plot_trend_example(
    prices: pd.Series,
    trend_df: pd.DataFrame,
    n_examples: int = 5,
    save_path: str = None
):
    """
    绘制趋势窗口示例图。

    :param prices: 价格序列
    :param trend_df: 标签 DataFrame
    :param n_examples: 示例数量
    :param save_path: 保存路径
    """
    # 选择 t_value 绝对值最大的几个事件作为示例
    top_events = trend_df['t_value'].abs().nlargest(n_examples)

    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3*n_examples), sharex=False)

    for i, (event_ts, t_abs) in enumerate(top_events.items()):
        ax = axes[i] if n_examples > 1 else axes

        row = trend_df.loc[event_ts]
        t1 = row['t1']
        side = row['side']
        t_value = row['t_value']

        # 获取趋势窗口内的价格
        if pd.notna(t1):
            window_prices = prices.loc[event_ts:t1]
            ax.plot(window_prices.index, window_prices.values,
                    color='steelblue', linewidth=1.5)

            # 标记起点和终点
            ax.scatter([event_ts], [prices.loc[event_ts]],
                       color='green', s=100, marker='o', label='Start', zorder=5)
            ax.scatter([t1], [prices.loc[t1]],
                       color='red', s=100, marker='x', label='End (t1)', zorder=5)

            # 标注 side 和 t_value
            direction = '↑ Up' if side == 1 else '↓ Down' if side == -1 else '— Flat'
            ax.set_title(f'{i+1}. {direction} (t={t_value:.2f}) | {event_ts.strftime("%Y-%m-%d %H:%M")} → {t1.strftime("%H:%M")}',
                        fontsize=10)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 趋势示例图已保存: {save_path}")
```

- [ ] **Step 2: 更新主函数调用可视化**

在 main() 函数的 Step 5 部分，替换为：

```python
    # Step 5: 可视化
    print("\n[Step 5] 生成可视化图表...")
    plot_trend_distribution(
        trend_df,
        os.path.join(FIGURES_DIR, '03_trend_distribution.png')
    )
    plot_trend_example(
        prices, trend_df, n_examples=5,
        save_path=os.path.join(FIGURES_DIR, '03_trend_example.png')
    )
```

并更新完成输出：

```python
    # 完成
    print("\n" + "=" * 70)
    print("  Phase 3 Trend Scanning 完成")
    print("=" * 70)
    print(f"输出文件: {FEATURES_DIR}")
    print(f"  - trend_labels.parquet")
    print(f"图表目录: {FIGURES_DIR}")
    print(f"  - 03_trend_distribution.png")
    print(f"  - 03_trend_example.png")
```

- [ ] **Step 3: 运行完整脚本**

```bash
cd /Users/link/Documents/AFMLKIT
uv run python strategies/IF9999/03_trend_scanning.py
```

Expected:
- 所有 5 个步骤完成
- 生成 03_trend_distribution.png 和 03_trend_example.png

- [ ] **Step 4: 检查输出图表**

```bash
ls -la strategies/IF9999/output/figures/03_*.png
```

Expected: 看到两个 PNG 文件

- [ ] **Step 5: Commit**

```bash
git add strategies/IF9999/03_trend_scanning.py strategies/IF9999/output/figures/03_*.png
git commit -m "feat(IF9999): add Phase 3 Trend Scanning visualization

- Side distribution (up/down/flat counts)
- t-value distribution histogram with mean/median
- Trend window examples with top |t-value| events

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: 更新 README 文档

**Files:**
- Modify: `strategies/IF9999/README.md`

- [ ] **Step 1: 添加 Phase 3 文档**

在 "Phase 2: 特征工程" 部分后添加：

```markdown
## Phase 3: Trend Scanning 标签生成

运行 Trend Scanning 生成 Primary Model 输出：

```bash
uv run python strategies/IF9999/03_trend_scanning.py
```

### Trend Scanning 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| TREND_WINDOWS | [5, 10, 20, 30, 50] | 窗口长度范围（Bars） |

### 输出文件

| 文件 | 说明 |
|------|------|
| `trend_labels.parquet` | Trend Scanning 标签（t1, t_value, side） |

### 标签格式

| 列 | 类型 | 说明 |
|----|------|------|
| `t1` | Datetime | 趋势窗口结束时间 |
| `t_value` | float64 | OLS 斜率 t 统计量（样本权重） |
| `side` | int8 | +1（上涨）/ -1（下跌）/ 0（无趋势） |

### 可视化

| 文件 | 说明 |
|------|------|
| `03_trend_distribution.png` | side 和 t_value 分布 |
| `03_trend_example.png` | 趋势窗口示例（高 |t_value| 事件） |

## 后续阶段

- Phase 4: Meta-Labeling 模型训练
```

并删除或修改 "后续阶段" 部分中的 Phase 3 描述（因为已经实现）。

- [ ] **Step 2: 更新项目结构说明**

更新项目结构部分：

```markdown
## 项目结构

```
IF9999/
├── 01_dollar_bar_builder.py    # Dollar Bars 构建与验证
├── 02_feature_engineering.py   # Phase 2: FracDiff + CUSUM
├── 03_trend_scanning.py        # Phase 3: Trend Scanning 标签
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
git commit -m "docs(IF9999): update README with Phase 3 Trend Scanning

- Add Phase 3 section with parameters and outputs
- Update project structure
- Mark Phase 3 as implemented

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- ✅ TREND_WINDOWS config parameter → Task 1
- ✅ 03_trend_scanning.py script → Tasks 2-4
- ✅ Data loading (Dollar Bars + CUSUM events) → Task 2
- ✅ trend_scan_labels() call → Task 3
- ✅ Output trend_labels.parquet → Task 3
- ✅ Visualization (distribution + example) → Task 4
- ✅ README documentation → Task 5

**2. Placeholder scan:**
- ✅ No TBD, TODO, or vague descriptions
- ✅ All code blocks have complete implementation
- ✅ All commands have expected output descriptions

**3. Type consistency:**
- ✅ trend_df columns: t1 (Datetime), t_value (float64), side (int8)
- ✅ TREND_WINDOWS: list[int]
- ✅ t_events: pd.DatetimeIndex
- ✅ prices: pd.Series

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-03-30-if9999-trend-scanning-plan.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** - 我为每个 Task 派发新 subagent，Task 间 review，快速迭代

**2. Inline Execution** - 在当前 session 用 executing-plans 执行，批量执行带 checkpoint

**选择哪种方式？**
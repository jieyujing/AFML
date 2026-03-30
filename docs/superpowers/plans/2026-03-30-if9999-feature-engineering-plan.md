# IF9999 Phase 2 特征工程实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 IF9999 Dollar Bars 应用 FracDiff 平稳化 + CUSUM Filter 事件采样，生成可用于后续 ML 模型的特征数据。

**Architecture:** 单脚本架构，复用 AFMLKit 现有模块（frac_diff, sampling/filters），无需重新开发核心算法。

**Tech Stack:** Python, pandas, numpy, matplotlib, seaborn, AFMLKit

---

## 文件结构

| 文件 | 负责内容 |
|------|----------|
| `strategies/IF9999/config.py` | 更新配置参数（添加 Phase 2 参数） |
| `strategies/IF9999/03_feature_engineering.py` | 主流程脚本（新建） |
| `strategies/IF9999/output/features/` | 特征输出目录 |
| `strategies/IF9999/output/figures/` | 可视化输出目录 |

---

## Task 1: 更新配置文件

**Files:**
- Modify: `strategies/IF9999/config.py`

- [ ] **Step 1: 添加 Phase 2 配置参数**

在 `config.py` 末尾添加：

```python
# ============================================================
# Phase 2: Feature Engineering 参数
# ============================================================

# FracDiff 参数
FRACDIFF_THRES = 1e-4      # FFD 权重截断阈值
FRACDIFF_D_STEP = 0.05     # d 搜索步长
FRACDIFF_MAX_D = 1.0       # d 最大值

# CUSUM Filter 参数
CUSUM_WINDOW = 20          # 动态阈值滚动窗口

# Phase 2 输出路径
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
```

- [ ] **Step 2: 验证配置可导入**

```bash
uv run python -c "from strategies.IF9999.config import FRACDIFF_THRES, CUSUM_WINDOW, FEATURES_DIR; print(f'THRES={FRACDIFF_THRES}, WINDOW={CUSUM_WINDOW}')"
```

Expected output:
```
THRES=0.0001, WINDOW=20
```

- [ ] **Step 3: 提交**

```bash
git add strategies/IF9999/config.py
git commit -m "feat(IF9999): add Phase 2 feature engineering config

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: 创建输出目录和脚本框架

**Files:**
- Create: `strategies/IF9999/output/features/` (目录)
- Create: `strategies/IF9999/03_feature_engineering.py` (脚本框架)

- [ ] **Step 1: 创建输出目录**

```bash
mkdir -p strategies/IF9999/output/features
```

- [ ] **Step 2: 编写脚本框架**

创建 `strategies/IF9999/03_feature_engineering.py`：

```python
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
```

- [ ] **Step 3: 验证脚本框架可运行**

```bash
uv run python strategies/IF9999/03_feature_engineering.py
```

Expected output:
```
Phase 2 Feature Engineering - 待实现
```

- [ ] **Step 4: 提交**

```bash
git add strategies/IF9999/output/features strategies/IF9999/03_feature_engineering.py
git commit -m "feat(IF9999): add Phase 2 feature engineering script skeleton

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: 编写数据加载和 FracDiff 模块

**Files:**
- Modify: `strategies/IF9999/03_feature_engineering.py`

- [ ] **Step 1: 编写数据加载函数**

在脚本框架的 `# ============================================================` 注释后添加：

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
```

- [ ] **Step 2: 编写 FracDiff 优化和验证函数**

继续添加：

```python
# ============================================================
# FracDiff 模块
# ============================================================

def run_fracdiff_optimization(prices: pd.Series) -> tuple:
    """
    运行 FracDiff 参数优化。

    :param prices: 价格序列
    :returns: (optimal_d, fracdiff_series, adf_result)
    """
    print("\n[FracDiff] 搜索最优 d 值...")

    # 搜索最优 d
    optimal_d = optimize_d(
        prices,
        thres=FRACDIFF_THRES,
        d_step=FRACDIFF_D_STEP,
        max_d=FRACDIFF_MAX_D,
        min_corr=0.0
    )
    print(f"  最优 d = {optimal_d:.4f}")

    # 应用 FracDiff
    fracdiff_series = frac_diff_ffd(prices, d=optimal_d, thres=FRACDIFF_THRES)
    print(f"  FracDiff 序列长度: {len(fracdiff_series)} (截断 {len(prices) - len(fracdiff_series)} 个)")

    # ADF 验证
    valid_series = fracdiff_series.dropna()
    if len(valid_series) > 10:
        t_stat, p_value, _ = adf_test(valid_series)
        print(f"  ADF 检验: t={t_stat:.4f}, p={p_value:.4f}")
        if p_value < 0.05:
            print(f"  ✅ 序列平稳 (p < 0.05)")
        else:
            print(f"  ⚠️ 序列非平稳 (p >= 0.05)")

    return optimal_d, fracdiff_series, (t_stat, p_value)
```

- [ ] **Step 3: 测试数据加载和 FracDiff**

在 `main()` 函数中临时添加测试代码：

```python
def main():
    """测试 FracDiff 模块"""
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target6.parquet')

    # 加载数据
    bars = load_dollar_bars(bars_path)
    prices = bars['close']

    # FracDiff
    optimal_d, fracdiff_series, adf_result = run_fracdiff_optimization(prices)

    print(f"\n测试完成: optimal_d={optimal_d:.4f}")
```

- [ ] **Step 4: 运行测试**

```bash
uv run python strategies/IF9999/03_feature_engineering.py
```

Expected output:
```
✅ 加载 Dollar Bars: XXX bars
   时间范围: XXX ~ XXX

[FracDiff] 搜索最优 d 值...
  最优 d = XXX
  FracDiff 序列长度: XXX (截断 XXX 个)
  ADF 检验: t=XXX, p=XXX
  ✅ 序列平稳 (p < 0.05)

测试完成: optimal_d=XXX
```

- [ ] **Step 5: 提交**

```bash
git add strategies/IF9999/03_feature_engineering.py
git commit -m "feat(IF9999): add FracDiff optimization module

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: 编写 CUSUM Filter 模块

**Files:**
- Modify: `strategies/IF9999/03_feature_engineering.py`

- [ ] **Step 1: 编写动态阈值计算函数**

在 FracDiff 模块后添加：

```python
# ============================================================
# CUSUM Filter 模块
# ============================================================

def compute_dynamic_cusum_threshold(fracdiff_series: pd.Series, window: int) -> pd.Series:
    """
    计算动态 CUSUM 阈值。

    基于滚动波动率，与 Dollar Bars 动态阈值风格一致。

    :param fracdiff_series: FracDiff 序列
    :param window: 滚动窗口
    :returns: 阈值序列
    """
    # 滚动标准差
    threshold_series = fracdiff_series.rolling(window).std()

    # 前 window 个点用全局 std 填充
    global_std = fracdiff_series.std()
    threshold_series = threshold_series.fillna(global_std)

    print(f"\n[CUSUM] 动态阈值计算完成")
    print(f"  窗口: {window}")
    print(f"  平均阈值: {threshold_series.mean():.6f}")
    print(f"  阈值范围: {threshold_series.min():.6f} ~ {threshold_series.max():.6f}")

    return threshold_series
```

- [ ] **Step 2: 编写 CUSUM Filter 应用函数**

继续添加：

```python
def run_cusum_filter(fracdiff_series: pd.Series, threshold_series: pd.Series) -> tuple:
    """
    应用 CUSUM Filter 采样事件点。

    :param fracdiff_series: FracDiff 序列
    :param threshold_series: 阈值序列
    :returns: (event_indices, s_pos, s_neg, n_events)
    """
    # 准备输入数组（去除 NaN）
    valid_idx = fracdiff_series.dropna().index
    diff_arr = fracdiff_series.dropna().values.astype(np.float64)
    threshold_arr = threshold_series.loc[valid_idx].values.astype(np.float64)

    # 应用 CUSUM Filter
    event_indices, s_pos, s_neg, thr = cusum_filter_with_state(diff_arr, threshold_arr)

    n_events = len(event_indices)
    print(f"\n[CUSUM] Filter 应用完成")
    print(f"  事件点数量: {n_events}")
    print(f"  事件率: {n_events / len(diff_arr) * 100:.2f}%")

    return event_indices, s_pos, s_neg, n_events
```

- [ ] **Step 3: 更新 main 函数测试 CUSUM**

```python
def main():
    """测试完整 FracDiff + CUSUM 流程"""
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target6.parquet')

    # 加载数据
    bars = load_dollar_bars(bars_path)
    prices = bars['close']

    # FracDiff
    optimal_d, fracdiff_series, adf_result = run_fracdiff_optimization(prices)

    # CUSUM 阈值
    threshold_series = compute_dynamic_cusum_threshold(fracdiff_series, CUSUM_WINDOW)

    # CUSUM Filter
    event_indices, s_pos, s_neg, n_events = run_cusum_filter(fracdiff_series, threshold_series)

    print(f"\n完整流程测试完成")
    print(f"  optimal_d = {optimal_d:.4f}")
    print(f"  n_events = {n_events}")
```

- [ ] **Step 4: 运行测试**

```bash
uv run python strategies/IF9999/03_feature_engineering.py
```

Expected output (包含 CUSUM 部分):
```
...
[CUSUM] 动态阈值计算完成
  窗口: 20
  平均阈值: XXX
  阈值范围: XXX ~ XXX

[CUSUM] Filter 应用完成
  事件点数量: XXX
  事件率: XXX%

完整流程测试完成
  optimal_d = XXX
  n_events = XXX
```

- [ ] **Step 5: 提交**

```bash
git add strategies/IF9999/03_feature_engineering.py
git commit -m "feat(IF9999): add CUSUM filter module

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: 编写可视化模块

**Files:**
- Modify: `strategies/IF9999/03_feature_engineering.py`

- [ ] **Step 1: 编写 FracDiff 对比图函数**

在 CUSUM 模块后添加：

```python
# ============================================================
# 可视化模块
# ============================================================

def plot_fracdiff_comparison(
    prices: pd.Series,
    fracdiff_series: pd.Series,
    optimal_d: float,
    save_path: str
):
    """
    绘制价格 vs FracDiff 对比图。

    :param prices: 原始价格序列
    :param fracdiff_series: FracDiff 序列
    :param optimal_d: 最优 d 值
    :param save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 价格序列
    ax1.plot(prices.index, prices.values, color='steelblue', linewidth=0.8)
    ax1.set_title('Original Price Series', fontsize=12)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)

    # FracDiff 序列
    ax2.plot(fracdiff_series.index, fracdiff_series.values, color='darkorange', linewidth=0.8)
    ax2.set_title(f'FracDiff Series (d={optimal_d:.4f})', fontsize=12)
    ax2.set_ylabel('FracDiff')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)

    # 添加零线
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ FracDiff 对比图已保存: {save_path}")
```

- [ ] **Step 2: 编写 CUSUM 状态曲线图函数**

继续添加：

```python
def plot_cusum_state(
    fracdiff_series: pd.Series,
    s_pos: np.ndarray,
    s_neg: np.ndarray,
    threshold_series: pd.Series,
    event_indices: np.ndarray,
    save_path: str
):
    """
    绘制 CUSUM 状态曲线图。

    :param fracdiff_series: FracDiff 序列（用于获取 index）
    :param s_pos: 正向累积数组
    :param s_neg: 负向累积数组
    :param threshold_series: 阈值序列
    :param event_indices: 事件点索引
    :param save_path: 保存路径
    """
    valid_idx = fracdiff_series.dropna().index

    fig, ax = plt.subplots(figsize=(14, 6))

    # CUSUM 状态曲线
    ax.plot(valid_idx, s_pos, color='green', linewidth=0.8, label='S+ (positive)')
    ax.plot(valid_idx, np.abs(s_neg), color='red', linewidth=0.8, label='S- (negative)')

    # 阈值线（取平均阈值）
    mean_threshold = threshold_series.mean()
    ax.axhline(y=mean_threshold, color='gray', linestyle='--', linewidth=1.5, label=f'Threshold (mean={mean_threshold:.4f})')

    # 事件点标记
    event_timestamps = valid_idx[event_indices]
    ax.scatter(event_timestamps, [mean_threshold] * len(event_indices),
               color='black', s=20, marker='^', label=f'Events (n={len(event_indices)})', zorder=5)

    ax.set_title('CUSUM Filter State Curves', fontsize=12)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Sum')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ CUSUM 状态图已保存: {save_path}")
```

- [ ] **Step 3: 编写事件点分布图函数**

继续添加：

```python
def plot_event_distribution(
    bars: pd.DataFrame,
    event_indices: np.ndarray,
    fracdiff_series: pd.Series,
    save_path: str
):
    """
    绘制事件点分布图（价格序列上标记事件点）。

    :param bars: Dollar Bars DataFrame
    :param event_indices: 事件点索引（对应 fracdiff_series）
    :param fracdiff_series: FracDiff 序列（用于映射索引）
    :param save_path: 保存路径
    """
    valid_idx = fracdiff_series.dropna().index
    event_timestamps = valid_idx[event_indices]

    # 事件点对应的价格
    event_prices = bars['close'].loc[event_timestamps]

    fig, ax = plt.subplots(figsize=(14, 6))

    # 价格序列
    ax.plot(bars.index, bars['close'].values, color='steelblue', linewidth=0.8, label='Price')

    # 事件点标记
    ax.scatter(event_timestamps, event_prices.values,
               color='red', s=30, marker='o', label=f'CUSUM Events (n={len(event_indices)})', zorder=5)

    ax.set_title('CUSUM Event Distribution on Price Series', fontsize=12)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 事件分布图已保存: {save_path}")
```

- [ ] **Step 4: 提交**

```bash
git add strategies/IF9999/03_feature_engineering.py
git commit -m "feat(IF9999): add visualization functions for FracDiff and CUSUM

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: 编写输出保存和完整主流程

**Files:**
- Modify: `strategies/IF9999/03_feature_engineering.py`

- [ ] **Step 1: 编写输出保存函数**

在可视化模块后添加：

```python
# ============================================================
# 输出保存
# ============================================================

def save_outputs(
    bars: pd.DataFrame,
    fracdiff_series: pd.Series,
    optimal_d: float,
    event_indices: np.ndarray,
    fracdiff_series_clean: pd.Series,  # 无 NaN 的序列
    adf_p_value: float  # ADF 检验 p 值
):
    """
    保存 FracDiff 序列和 CUSUM 事件点。

    :param bars: Dollar Bars DataFrame
    :param fracdiff_series: FracDiff 序列（含 NaN）
    :param optimal_d: 最优 d 值
    :param event_indices: 事件点索引
    :param fracdiff_series_clean: 无 NaN 的 FracDiff 序列
    :param adf_p_value: ADF 检验 p 值
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # 1. FracDiff 序列
    fracdiff_df = pd.DataFrame({
        'fracdiff': fracdiff_series,
    })
    fracdiff_path = os.path.join(FEATURES_DIR, 'fracdiff_series.parquet')
    fracdiff_df.to_parquet(fracdiff_path)
    print(f"\n✅ FracDiff 序列已保存: {fracdiff_path}")
    print(f"   长度: {len(fracdiff_df)}")

    # 保存参数信息
    params_df = pd.DataFrame({
        'optimal_d': [optimal_d],
        'n_events': [len(event_indices)],
        'adf_p_value': [adf_p_value]
    })
    params_path = os.path.join(FEATURES_DIR, 'fracdiff_params.parquet')
    params_df.to_parquet(params_path)
    print(f"✅ 参数信息已保存: {params_path}")

    # 2. CUSUM 事件点
    valid_idx = fracdiff_series_clean.index
    event_timestamps = valid_idx[event_indices]

    events_df = pd.DataFrame({
        'timestamp': event_timestamps,
        'price': bars['close'].loc[event_timestamps].values,
        'fracdiff': fracdiff_series_clean.iloc[event_indices].values,
    })
    events_path = os.path.join(FEATURES_DIR, 'cusum_events.parquet')
    events_df.to_parquet(events_path)
    print(f"✅ CUSUM 事件点已保存: {events_path}")
    print(f"   事件数量: {len(events_df)}")
```

- [ ] **Step 2: 编写完整主流程**

替换 `main()` 函数：

```python
def main():
    """
    IF9999 Phase 2 特征工程主流程。
    """
    print("=" * 70)
    print("  IF9999 Phase 2 Feature Engineering")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载 Dollar Bars
    print("\n[Step 1] 加载 Dollar Bars...")
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target6.parquet')
    bars = load_dollar_bars(bars_path)
    prices = bars['close']

    # Step 2: FracDiff 优化
    print("\n[Step 2] FracDiff 参数优化...")
    optimal_d, fracdiff_series, adf_result = run_fracdiff_optimization(prices)

    # Step 3: CUSUM 阈值
    print("\n[Step 3] 计算 CUSUM 动态阈值...")
    threshold_series = compute_dynamic_cusum_threshold(fracdiff_series, CUSUM_WINDOW)

    # Step 4: CUSUM Filter
    print("\n[Step 4] 应用 CUSUM Filter...")
    event_indices, s_pos, s_neg, n_events = run_cusum_filter(fracdiff_series, threshold_series)

    # Step 5: 可视化
    print("\n[Step 5] 生成可视化图表...")
    plot_fracdiff_comparison(
        prices, fracdiff_series, optimal_d,
        os.path.join(FIGURES_DIR, 'fracdiff_comparison.png')
    )

    plot_cusum_state(
        fracdiff_series, s_pos, s_neg, threshold_series, event_indices,
        os.path.join(FIGURES_DIR, 'cusum_state.png')
    )

    plot_event_distribution(
        bars, event_indices, fracdiff_series,
        os.path.join(FIGURES_DIR, 'event_distribution.png')
    )

    # Step 6: 保存输出
    print("\n[Step 6] 保存输出文件...")
    fracdiff_clean = fracdiff_series.dropna()
    save_outputs(
        bars, fracdiff_series, optimal_d, event_indices, fracdiff_clean,
        adf_result[1]  # ADF p-value
    )

    # 完成
    print("\n" + "=" * 70)
    print("  Phase 2 Feature Engineering 完成")
    print("=" * 70)
    print(f"输出目录: {FEATURES_DIR}")
    print(f"  - fracdiff_series.parquet")
    print(f"  - fracdiff_params.parquet")
    print(f"  - cusum_events.parquet")
    print(f"图表目录: {FIGURES_DIR}")
    print(f"  - fracdiff_comparison.png")
    print(f"  - cusum_state.png")
    print(f"  - event_distribution.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行完整流程**

```bash
uv run python strategies/IF9999/03_feature_engineering.py
```

Expected output:
```
======================================================================
  IF9999 Phase 2 Feature Engineering
======================================================================

[Step 1] 加载 Dollar Bars...
✅ 加载 Dollar Bars: XXX bars
   时间范围: XXX ~ XXX

[Step 2] FracDiff 参数优化...
[FracDiff] 搜索最优 d 值...
  最优 d = XXX
  ...
[Step 3] 计算 CUSUM 动态阈值...
...
[Step 4] 应用 CUSUM Filter...
...
[Step 5] 生成可视化图表...
✅ FracDiff 对比图已保存: ...
✅ CUSUM 状态图已保存: ...
✅ 事件分布图已保存: ...

[Step 6] 保存输出文件...
✅ FracDiff 序列已保存: ...
✅ 参数信息已保存: ...
✅ CUSUM 事件点已保存: ...

======================================================================
  Phase 2 Feature Engineering 完成
======================================================================
```

- [ ] **Step 4: 验证输出文件**

```bash
ls -la strategies/IF9999/output/features/
ls -la strategies/IF9999/output/figures/
```

Expected files:
- `fracdiff_series.parquet`
- `fracdiff_params.parquet`
- `cusum_events.parquet`
- `fracdiff_comparison.png`
- `cusum_state.png`
- `event_distribution.png`

- [ ] **Step 5: 提交**

```bash
git add strategies/IF9999/03_feature_engineering.py strategies/IF9999/output/features strategies/IF9999/output/figures
git commit -m "feat(IF9999): complete Phase 2 feature engineering workflow

- FracDiff optimization with auto-d search
- Dynamic threshold CUSUM filter
- Three visualization charts
- Output: fracdiff_series, cusum_events

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 7: 更新 README 文档

**Files:**
- Modify: `strategies/IF9999/README.md`

- [ ] **Step 1: 更新 README 添加 Phase 2 内容**

在现有 README.md 的"项目结构"和"后续阶段"部分更新：

```markdown
## 项目结构

```
IF9999/
├── 01_dollar_bar_builder.py    # Dollar Bars 构建与验证
├── 02_parameter_optimization.py # Dollar Bars 参数优化
├── 03_feature_engineering.py   # Phase 2: FracDiff + CUSUM
├── config.py                   # 配置参数
├── output/
│   ├── bars/                   # 生成的 Bar 数据
│   ├── features/               # Phase 2 特征输出
│   └── figures/                # 可视化图表
└── README.md
```

## Phase 2: 特征工程

运行特征工程流程：

```bash
uv run python strategies/IF9999/03_feature_engineering.py
```

### FracDiff 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| FRACDIFF_THRES | 1e-4 | FFD 权重截断阈值 |
| FRACDIFF_D_STEP | 0.05 | d 搜索步长 |
| FRACDIFF_MAX_D | 1.0 | d 最大值 |

### CUSUM Filter 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| CUSUM_WINDOW | 20 | 动态阈值滚动窗口 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `fracdiff_series.parquet` | FracDiff 平稳化序列 |
| `fracdiff_params.parquet` | 最优 d 值和统计信息 |
| `cusum_events.parquet` | CUSUM 事件点索引 |

## 后续阶段

- Phase 3: 标签生成（Trend Scanning）← 下一步
- Phase 4: Meta-Labeling 模型训练
```

- [ ] **Step 2: 提交**

```bash
git add strategies/IF9999/README.md
git commit -m "docs(IF9999): update README with Phase 2 feature engineering

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## 自检清单

- [x] Spec coverage: 设计文档所有章节已覆盖
- [x] Placeholder scan: 无 TBD、TODO 等占位符
- [x] Type consistency: 函数签名和变量名在各任务中一致
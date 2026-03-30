# IF9999 Phase 2 特征工程设计文档

**Goal**: 为 IF9999 Dollar Bars 应用 FracDiff 平稳化 + CUSUM Filter 事件采样，生成可用于后续 ML 模型的特征数据。

**Architecture**: 单脚本架构，复用 AFMLKit 现有模块，无需重新开发核心算法。

**Tech Stack**: Python, pandas, numpy, AFMLKit (frac_diff, sampling/filters)

---

## 现有模块映射

| 功能 | AFMLKit 模块 | 用途 |
|------|-------------|------|
| FracDiff FFD | `afmlkit/feature/core/frac_diff.py::optimize_d()` | 自动搜索最优 d 值 |
| FracDiff FFD | `afmlkit/feature/core/frac_diff.py::frac_diff_ffd()` | 应用分数差分 |
| CUSUM Filter | `afmlkit/sampling/filters.py::cusum_filter()` | 事件采样 |
| CUSUM Filter | `afmlkit/sampling/filters.py::cusum_filter_with_state()` | 带状态的 CUSUM（可视化） |
| ADF Test | `afmlkit/feature/core/structural_break/adf.py::adf_test()` | 平稳性验证 |

---

## 文件结构

| 文件 | 负责内容 |
|------|----------|
| `strategies/IF9999/03_feature_engineering.py` | 主流程脚本 |
| `strategies/IF9999/output/features/fracdiff_series.parquet` | FracDiff 输出序列 |
| `strategies/IF9999/output/features/cusum_events.parquet` | CUSUM 事件点索引 |
| `strategies/IF9999/output/figures/fracdiff_comparison.png` | 价格 vs FracDiff 对比图 |
| `strategies/IF9999/output/figures/cusum_state.png` | CUSUM 状态曲线图 |
| `strategies/IF9999/output/figures/event_distribution.png` | 事件点分布图 |

---

## 主流程设计

### Step 1: 加载 Dollar Bars

```python
bars = pd.read_parquet('output/bars/dollar_bars_target6.parquet')
prices = bars['close']
```

### Step 2: FracDiff 参数优化

调用 `optimize_d()` 自动搜索使序列平稳的最小 d 值：

```python
from afmlkit.feature.core.frac_diff import optimize_d, frac_diff_ffd

optimal_d = optimize_d(
    prices,
    thres=1e-4,        # FFD 权重截断阈值
    d_step=0.05,       # 搜索步长
    max_d=1.0,         # 最大 d 值
    min_corr=0.0       # 不约束相关性
)
```

输出：
- `optimal_d`: 最优差分阶数（预计 0.3~0.6 范围）

### Step 3: 应用 FracDiff

```python
fracdiff_series = frac_diff_ffd(prices, d=optimal_d, thres=1e-4)
```

验证：
- ADF p-value < 0.05（平稳）
- 与原始序列相关性（可选检查）

### Step 4: 计算动态 CUSUM 阈值

**采用动态阈值策略**，基于滚动波动率计算，与 Dollar Bars 的动态阈值风格一致：

```python
# 动态阈值：使用 FracDiff 序列的滚动标准差
window = 20  # 滚动窗口
threshold_series = fracdiff_series.rolling(window).std().fillna(
    fracdiff_series.std()  # 前 window 个点用全局 std 填充
)
threshold_mean = threshold_series.mean()  # 平均阈值（用于参考）
```

### Step 5: 应用 CUSUM Filter

```python
from afmlkit.sampling.filters import cusum_filter_with_state

# 转换为 numpy 数组
diff_arr = fracdiff_series.dropna().values
threshold_arr = threshold_series.dropna().values  # 动态阈值

event_indices, s_pos, s_neg, thr = cusum_filter_with_state(diff_arr, threshold_arr)
```

输出：
- `event_indices`: 事件发生的 Bar 索引
- `s_pos`, `s_neg`: CUSUM 状态曲线（用于可视化）

### Step 6: 可视化验证

**图 1: FracDiff 对比**
- 原始价格序列 vs FracDiff 序列
- 双轴展示（价格在主轴，FracDiff 在次轴）

**图 2: CUSUM 状态曲线**
- `s_pos`（正向累积）和 `s_neg`（负向累积）
- 阈值线标记
- 事件点标记

**图 3: 事件点分布**
- 价格序列上标记 CUSUM 事件点
- 显示事件触发时的价格位置

### Step 7: 输出保存

```python
# FracDiff 序列
fracdiff_df = pd.DataFrame({
    'fracdiff': fracdiff_series,
    'd': optimal_d
})
fracdiff_df.to_parquet('output/features/fracdiff_series.parquet')

# CUSUM 事件点
events_df = pd.DataFrame({
    'event_idx': event_indices,
    'timestamp': bars.index[event_indices],
    'price': bars['close'].iloc[event_indices]
})
events_df.to_parquet('output/features/cusum_events.parquet')
```

---

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FRACDIFF_THRES` | 1e-4 | FFD 权重截断阈值 |
| `FRACDIFF_D_STEP` | 0.05 | d 搜索步长 |
| `FRACDIFF_MAX_D` | 1.0 | d 最大值 |
| `CUSUM_WINDOW` | 20 | 阈值滚动窗口（动态阈值） |

---

## 验收标准

1. **FracDiff 输出**
   - ADF p-value < 0.05（序列平稳）
   - 无过多 NaN（截断窗口合理）
   - optimal_d 在合理范围（0 < d < 1）

2. **CUSUM 输出**
   - 事件点数量合理（不宜过多或过少）
   - 事件点覆盖多空双向

3. **可视化**
   - 图表清晰，标注完整
   - 可直观理解 FracDiff 和 CUSUM 效果

---

## 后续阶段衔接

Phase 2 输出将用于：
- **Phase 3**: Trend Scanning 标签生成（在 CUSUM 事件点处应用）
- **Phase 4**: Meta-Labeling 模型训练（使用 FracDiff 作为特征）

---

## 参考

- AFML 第 5 章：Fractional Differentiation
- AFML 第 6 章：Entropy and Market Regime Detection（CUSUM Filter）
- `afmlkit/feature/core/frac_diff.py` 源码
- `afmlkit/sampling/filters.py` 源码
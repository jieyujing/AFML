---
name: afmlkit
description: Local codebase analysis for afmlkit — 基于 Advances in Financial Machine Learning 方法论的高性能金融机器学习工具包，支持 K 线构建、特征工程、Trend Scan 主模型、三重屏障标签、Meta-Labeling、采样过滤、样本权重计算、特征聚类与重要性评估（Clustered MDA）、以及带 Purge/Embargo 的交叉验证。当需要使用或修改 afmlkit 代码库、构建 AFML 量化流程（事件驱动采样 → Trend Scan 主模型 → 三重屏障 Meta-Labeling → 样本权重 → 特征重要性分析 → 模型训练）、或理解核心类 API 时使用此 skill。
---

# AFMLKit

基于 *Advances in Financial Machine Learning*（López de Prado, 2018）的高性能 Python 量化工具包。使用 Numba JIT 编译，专为处理大规模高频交易数据设计。

**路径**: `afmlkit/` | **Python 3.13** | **包管理**: uv

## 核心模块

| 模块 | 主要类/函数 | 用途 |
|------|------------|------|
| `bar.data_model` | `TradesData` | 交易数据容器，预处理，HDF5 月度分区存储 |
| `bar.io` | `AddTimeBarH5`, `TimeBarReader`, `H5Inspector` | K 线持久化与读取工具 |
| `bar.kit` | `TimeBarKit`, `VolumeBarKit`, `DollarBarKit` | K 线构建（时间/成交量/金额/Tick） |
| `feature.base` | `SISOTransform`, `MISOTransform` | 自定义 Transform 基类（含双后端支持） |
| `feature.kit` | `Feature`, `FeatureKit` | 特征运算与管道构建 |
| `feature.core.*` | `ewma`, `ewmst`, `atr`, `realized_vol`… | 内置技术指标（Numba 加速） |
| `feature.core.trend_scan` | `_trend_scan_core`, `trend_scan_labels` | Trend Scan 主模型 — OLS t-统计量动态趋势识别（MLAM Ch.3.5）|
| `sampling.filters` | `cusum_filter`, `z_score_peak_filter` | 事件检测过滤器（AFML Ch.2） |
| `label.kit` | `TBMLabel`, `SampleWeights` | 三重屏障标签高层 API（AFML Ch.3/4） |
| `label.tbm` | `triple_barrier` | TBM Numba 核心函数 |
| `label.weights` | `average_uniqueness`, `return_attribution`… | 样本权重 Numba 函数 |
| `validation.purged_cv` | `PurgedKFold` | Purge + Embargo 交叉验证（AFML Ch.7） |
| `importance.clustering` | `cluster_features`, `get_feature_distance_matrix`, `hierarchical_clustering` | 特征聚类（相关性距离 + 层次聚类 + 自动 k）（AFML Ch.4） |
| `importance.mda` | `clustered_mda` | 聚类 MDA 特征重要性（Log-loss + sample_weight）（AFML Ch.8） |

## 标准 AFML 工作流

**完整端到端示例** → 见 `references/workflow.md`（含元标签和 HDF5 存储工作流）

简化流程：

```python
from afmlkit.bar.data_model import TradesData
from afmlkit.bar.kit import DollarBarKit
from afmlkit.feature.kit import Feature, FeatureKit
from afmlkit.sampling.filters import cusum_filter
from afmlkit.feature.core.trend_scan import trend_scan_labels
from afmlkit.label.kit import TBMLabel, SampleWeights
from afmlkit.importance.clustering import cluster_features
from afmlkit.importance.mda import clustered_mda
from afmlkit.validation.purged_cv import PurgedKFold

trades = TradesData.load_trades_h5('data/trades.h5', start_time='2023-01-01')
ohlcv  = DollarBarKit(trades, dollar_thrs=1_000_000).build_ohlcv()
# ... 特征工程 → cusum_filter → trend_scan_labels → TBMLabel(is_meta) → SampleWeights → 特征重要性分析
```

## Transform 命名约定

- `SISOTransform('close', 'sma_20')` → 输出列 `close_sma_20`
- `MISOTransform(['col1', 'col2'], 'output')` → 输出列 `output`
- `FeatureKit.build(df, backend='nb')` — `'nb'` 使用 Numba，`'pd'` 使用 Pandas

## 重要注意事项

- **Numba JIT 首次慢**：测试用 `$env:NUMBA_DISABLE_JIT=1; uv run pytest`
- **时间戳格式**：`triple_barrier` 和权重函数均需纳秒 `int64`，用 `ts.view('int64')` 转换
- **元标签 side**：`is_meta=True` 时 `features` 必须含 `side` 列（值为 -1 或 1）。推荐使用 `trend_scan_labels()` 动态生成 side。
- **竖直屏障权重**：将 `vertical_touch_weights` 传入 `compute_final_weights` 以降低噪声标签权重

## Trend Scan Primary Model（趋势扫描主模型）

基于 MLAM Ch.3.5 的动态趋势识别，用于替代固定窗口的动量/均线方向判定。完整流水线见 `scripts/cusum_filtering.py`。

```python
from afmlkit.feature.core.trend_scan import trend_scan_labels

# 在 CUSUM 事件点上运行 Trend Scan
trend_df = trend_scan_labels(
    price_series=close_prices,   # pd.Series with DatetimeIndex
    t_events=t_events,           # pd.DatetimeIndex from CUSUM
    L_windows=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)
# trend_df 含 3 列:
#   t1      — 最优窗口长度 (int64)
#   t_value — OLS 斜率的 t-统计量 (float64)
#   side    — 趋势方向 +1/-1 (int8)
```

**核心架构决策：**
- **Numba `@njit` 纯数学 OLS**：弃用 `statsmodels.OLS`，直接用累加公式（$S_{xx}, S_{xy}, S_{yy}$）计算斜率 t-统计量，性能提升 1000x+
- **后视扫描（Backward Scan）**：仅在 $t_{events}$ 位置启动，向过去回溯 $L$ 条数据，杜绝前视偏差
- **零方差保护**：`epsilon = 1e-12` 捕获完全横盘段，返回 `t_value=0, side=0`
- **不开 `parallel=True`**：各事件的窗口扫描是路径依赖的穷举搜索，`prange` 反而降速

**与 Meta-Labeling 的集成：**
```python
# 1. 将 side 注入 features DataFrame
features['side'] = trend_df.loc[features.index, 'side'].astype(int)

# 2. 启用 Meta-Labeling
tbm = TBMLabel(features=features, target_ret_col='volatility',
               horizontal_barriers=(1, 1), vertical_barrier=pd.Timedelta(days=1),
               min_ret=0.0, is_meta=True)

# 3. |t_value| → 标准化 → 乘入 avg_uniqueness 作为 trend-weighted sample weight
t_normalized = trend_df['t_value'].abs() / trend_df['t_value'].abs().max()
trend_weighted_uniqueness = avg_uniqueness * t_normalized
```

## 特征重要性分析（Feature Importance）

完整流程见 `scripts/feature_importance_analysis.py`。核心步骤：

```python
from afmlkit.importance.clustering import cluster_features
from afmlkit.importance.mda import clustered_mda

# 1. 特征聚类 — 自动选 k 或手动指定
clusters = cluster_features(X_features, method='ward')       # auto k via Silhouette
clusters = cluster_features(X_features, n_clusters=5)        # manual k

# 2. Clustered MDA — 簇级打乱 + PurgedKFold + Log-loss
df_importance = clustered_mda(
    X=X_features, y=y_labels, clusters=clusters,
    t1=t1_series, sample_weight=weights,
    n_splits=5, embargo_pct=0.01,
)
```

**关键设计选择：**
- **距离矩阵**：$D = \sqrt{0.5 \times (1 - \rho)}$ 满足度量空间特性，数值稳定
- **聚类级打乱**：MDA 打乱整个特征簇而非单个特征，消除多重共线性的"替代效应"
- **Log-loss 评估**：使用带 `sample_weight` 的 Log-loss 而非 Accuracy，因为概率校准质量对 Bet Sizing 至关重要
- **PurgedKFold**：需传入 `t1`（事件结束时间），自动 Purge 时间重叠样本 + Embargo 缓冲期
- **诊断输出**：每个 Fold 打印训练集有效样本比例，<30% 时触发警告

**PurgedKFold 单独使用：**
```python
from afmlkit.validation.purged_cv import PurgedKFold

cv = PurgedKFold(n_splits=5, t1=t1_series, embargo_pct=0.01)
for train_idx, test_idx in cv.split(X):
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    # PurgedKFold 兼容 sklearn API，可直接用于 GridSearchCV
```

## 深度见解：Dollar Bar 频率评估

在评估 Dollar Bars 的 IID 特性时：
1. **独立性 > 正态性**：一阶自相关（Autocorr）接近 0 比 Jarque-Bera (JB) 统计量绝对值更重要。
2. **JB 的样本量偏误**：JB 统计量随样本量 $N$ 增加而自然增大。在全量数据下，不要仅因 JB 高就否定高频 Bars，应同步观察自相关和样本量平衡。
3. **性能建议**：对全量 Tick 映射日度阈值时，使用 `np.searchsorted` 替代 Python 循环，可获得 20x+ 的性能加速。

## 架构与性能工程最佳实践 (AFML 算法设计军规)

1. **底层算法强制解耦**：像 CUSUM 这种事件过滤器，核心是“状态方程”（如 $S^+ = \max(0, S^+ + diff)$），它必须是**纯净无状态的累加器**。底层 `@njit` 核心代码**绝对不要**自作主张去计算对数收益率。外部算好增量（不论是普通收益率、绝对价差、还是经过 FracDiff 的分数阶序列）后直接传入。只有解耦，流水线才能像乐高一样灵活组合。
2. **警惕“伪并发”陷阱**：在使用 Numba 加速时，如果算法是**路径依赖**的（如 CUSUM，今日的状态严密依赖昨日状态），**绝对不能**使用 `prange` 或 `@njit(parallel=True)`。强行并行不仅无效，还会引入高昂的多线程调度开销（Overhead），拖慢性能。
3. **向量化替代滑动切片**：在计算分数阶差分（FracDiff FFD）这类数学本质是一维连续方程的场景中，**永远不要**在大型 Pandas Series 上用 `for` 循环和 `iloc` 做滑动。先生成静态权重矩阵，然后使用 `np.convolve(series, w, mode='valid')` 替代循环，可实现计算耗时从数秒到毫秒级的百倍优化。
4. **量纲对齐原则 (Scale Matching)**：进行 CUSUM 抽样或阈值计算时，极其容易出现“量纲错位”。如果波动的阈值（Threshold）是用对数收益率通过 EWMS 算出来的百分比波动率，那么传入 CUSUM 内部的 `diff` 序列必须严格在同一量纲（即百分比或对数价格的一阶增量）！否则将导致绝大多数有效信号被拦截，或出现事件爆炸的错误。

## 参考文档

按需加载以下参考文件：

- **`references/workflow.md`** — 完整工作流代码（标准流程、元标签、HDF5、自定义 Transform）
- **`references/api_quickref.md`** — API 速查表、函数签名、常见陷阱
- **`references/api_reference/`** — 各模块详细 API 文档
  - `kit.md` — TBMLabel / SampleWeights
  - `data_model.md` — TradesData / FootprintData
  - `tbm.md` — triple_barrier Numba 函数
  - `weights.md` — 权重计算函数
  - `filters.md` — cusum_filter / z_score_peak_filter
  - `volatility.md` — 波动率系列函数
  - `base.md` — BaseTransform / CoreTransform / SISO / MISO
  - `io.md` — H5Inspector / AddTimeBarH5 / TimeBarReader
- **源码文件**（新增模块）
  - `afmlkit/feature/core/trend_scan.py` — Trend Scan 主模型（Numba 核心 + Pandas 前端）
  - `afmlkit/validation/purged_cv.py` — PurgedKFold 实现
  - `afmlkit/importance/clustering.py` — 特征聚类
  - `afmlkit/importance/mda.py` — Clustered MDA
  - `scripts/cusum_filtering.py` — CUSUM → Trend Scan → Meta-Labeling 端到端流水线
  - `scripts/feature_importance_analysis.py` — 端到端集成脚本
  - `tests/features/test_trend_scan.py` — Trend Scan 正确性与无偏差验证（15 tests）

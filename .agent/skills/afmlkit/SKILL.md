---
name: afmlkit
description: Local codebase analysis for afmlkit — 基于 Advances in Financial Machine Learning 方法论的高性能金融机器学习工具包，支持 K 线构建、特征工程、三重屏障标签、采样过滤及样本权重计算。当需要使用或修改 afmlkit 代码库、构建 AFML 量化流程（事件驱动采样 → 三重屏障标签 → 样本权重 → 模型训练）、或理解 TradesData/TBMLabel/FeatureKit 等核心类的 API 时使用此 skill。
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
| `sampling.filters` | `cusum_filter`, `z_score_peak_filter` | 事件检测过滤器（AFML Ch.2） |
| `label.kit` | `TBMLabel`, `SampleWeights` | 三重屏障标签高层 API（AFML Ch.3/4） |
| `label.tbm` | `triple_barrier` | TBM Numba 核心函数 |
| `label.weights` | `average_uniqueness`, `return_attribution`… | 样本权重 Numba 函数 |

## 标准 AFML 工作流

**完整端到端示例** → 见 `references/workflow.md`（含元标签和 HDF5 存储工作流）

简化流程：

```python
from afmlkit.bar.data_model import TradesData
from afmlkit.bar.kit import DollarBarKit
from afmlkit.feature.kit import Feature, FeatureKit
from afmlkit.sampling.filters import cusum_filter
from afmlkit.label.kit import TBMLabel, SampleWeights

trades = TradesData.load_trades_h5('data/trades.h5', start_time='2023-01-01')
ohlcv  = DollarBarKit(trades, dollar_thrs=1_000_000).build_ohlcv()
# ... 特征工程 → cusum_filter → TBMLabel → SampleWeights
```

## Transform 命名约定

- `SISOTransform('close', 'sma_20')` → 输出列 `close_sma_20`
- `MISOTransform(['col1', 'col2'], 'output')` → 输出列 `output`
- `FeatureKit.build(df, backend='nb')` — `'nb'` 使用 Numba，`'pd'` 使用 Pandas

## 重要注意事项

- **Numba JIT 首次慢**：测试用 `$env:NUMBA_DISABLE_JIT=1; uv run pytest`
- **时间戳格式**：`triple_barrier` 和权重函数均需纳秒 `int64`，用 `ts.view('int64')` 转换
- **元标签 side**：`is_meta=True` 时 `features` 必须含 `side` 列（值为 -1 或 1）
- **竖直屏障权重**：将 `vertical_touch_weights` 传入 `compute_final_weights` 以降低噪声标签权重

## 深度见解：Dollar Bar 频率评估

在评估 Dollar Bars 的 IID 特性时：
1. **独立性 > 正态性**：一阶自相关（Autocorr）接近 0 比 Jarque-Bera (JB) 统计量绝对值更重要。
2. **JB 的样本量偏误**：JB 统计量随样本量 $N$ 增加而自然增大。在全量数据下，不要仅因 JB 高就否定高频 Bars，应同步观察自相关和样本量平衡。
3. **性能建议**：对全量 Tick 映射日度阈值时，使用 `np.searchsorted` 替代 Python 循环，可获得 20x+ 的性能加速。

## 参考文档

按需加载以下参考文件：

- **`references/workflow.md`** — 完整工作流代码（标准流程、元标签、HDF5、自定义 Transform）
- **`references/api_quickref.md`** — API 速查表、函数签名、常见陷阱
- **`references/api_reference/`** — 各模块详细 API 文档（20 个 .md 文件）
  - `kit.md` — TBMLabel / SampleWeights
  - `data_model.md` — TradesData / FootprintData
  - `tbm.md` — triple_barrier Numba 函数
  - `weights.md` — 权重计算函数
  - `filters.md` — cusum_filter / z_score_peak_filter
  - `volatility.md` — 波动率系列函数
  - `base.md` — BaseTransform / CoreTransform / SISO / MISO
  - `io.md` — H5Inspector / AddTimeBarH5 / TimeBarReader

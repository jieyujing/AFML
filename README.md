# AFMLKit

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-green.svg)](https://afmlkit.readthedocs.io)

**AFMLKit** 是一个基于 *Advances in Financial Machine Learning*（Marcos López de Prado 著）方法论的高性能金融机器学习工具包。使用 Numba 进行 JIT 编译加速，专为处理大规模金融数据而设计。

## 特性

- **K线构建** — 支持多种 K 线采样策略：时间、Tick、成交量、金额、CUSUM、不平衡、运行条
- **特征工程** — 流式接口的特征工程框架，支持数学运算、函数组合和管道构建
- **标签生成** — 实现三重屏障法（Triple Barrier Method）进行事件标签生成
- **采样过滤** — CUSUM 过滤器等事件检测方法
- **高性能** — 基于 Numba JIT 编译，支持并行计算
- **双后端** — 同时支持 Pandas 和 Numba 计算后端

## 安装

```bash
# 基础安装
pip install afmlkit

# 开发模式安装（包含测试依赖）
pip install -e .[dev]
```

## 快速开始

### K线构建

```python
import pandas as pd
from afmlkit.bar.kit import TimeBarKit, VolumeBarKit, DollarBarKit
from afmlkit.bar.data_model import TradesData

# 加载原始交易数据
trades = TradesData(pd.read_parquet('trades.parquet'))

# 构建 5 分钟 K 线
time_bars = TimeBarKit(trades, pd.Timedelta('5min'))
ohlcv_df = time_bars.build_ohlcv()

# 构建成交量 K 线（每 1000 单位成交量）
volume_bars = VolumeBarKit(trades, volume_ths=1000)
ohlcv_df = volume_bars.build_ohlcv()

# 构建金额 K 线（每 10000 美元）
dollar_bars = DollarBarKit(trades, dollar_thrs=10000)
ohlcv_df = dollar_bars.build_ohlcv()
```

### 特征工程

```python
from afmlkit.feature.kit import Feature, FeatureKit
from afmlkit.feature.core.ma import SimpleMovingAverageTransform
from afmlkit.feature.core.volatility import VolatilityTransform

# 创建特征
price = Feature(SimpleMovingAverageTransform('close', 'sma_20'))
volatility = Feature(VolatilityTransform('close', 'volatility_20'))

# 数学运算
price_to_vol_ratio = price / volatility
normalized = (price - price.rolling_mean(50)) / price.rolling_std(50)

# 构建特征管道
kit = FeatureKit([
    price,
    volatility,
    price_to_vol_ratio,
    normalized
], retain=['close', 'volume'])

# 执行计算
features_df = kit.build(ohlcv_df, backend='nb', timeit=True)
```

### 标签生成（三重屏障法）

```python
from afmlkit.label.kit import TBMLabel, SampleWeights

# 创建标签生成器
tbm = TBMLabel(
    features=features_df,
    target_ret_col='volatility_20',  # 波动率作为目标收益
    min_ret=0.001,  # 最小收益率阈值
    horizontal_barriers=(1.0, 1.0),  # 止损/止盈乘数
    vertical_barrier=pd.Timedelta('1D')  # 时间屏障
)

# 计算标签
features, labels = tbm.compute_labels(trades)

# 计算样本权重
weights = tbm.compute_weights(trades, normalized=True)

# 组合最终权重
final_weights = SampleWeights.compute_final_weights(
    avg_uniqueness=weights['avg_uniqueness'],
    return_attribution=weights['return_attribution'],
    time_decay_intercept=0.5,
    labels=labels['labels']
)
```

### 采样过滤

```python
from afmlkit.sampling.filters import cusum_filter

# 使用 CUSUM 过滤器检测事件
event_indices = cusum_filter(
    raw_time_series=prices,
    threshold=volatility  # 可使用波动率作为动态阈值
)
```

## 项目结构

```
afmlkit/
├── bar/                    # K 线构建模块
│   ├── base.py            # 基础类和核心函数
│   ├── logic.py           # Numba 加速的索引器
│   ├── data_model.py      # 数据模型
│   └── kit.py             # 具体 K 线实现
├── feature/               # 特征工程模块
│   ├── base.py            # Transform 基类
│   ├── core/              # 核心特征实现
│   │   ├── ma.py          # 移动平均
│   │   ├── volatility.py  # 波动率
│   │   ├── momentum.py    # 动量
│   │   ├── trend.py       # 趋势
│   │   ├── reversion.py   # 均值回归
│   │   ├── volume.py      # 成交量
│   │   ├── correlation.py # 相关性
│   │   └── time.py        # 时间特征
│   └── kit.py             # Feature 和 FeatureKit
├── label/                 # 标签生成模块
│   ├── tbm.py            # 三重屏障法核心实现
│   ├── weights.py        # 样本权重计算
│   └── kit.py            # TBMLabel API
├── sampling/              # 采样模块
│   └── filters.py        # CUSUM 过滤器
└── utils/                 # 工具函数
    └── log.py            # 日志工具
```

## 测试

```bash
# 运行测试（推荐禁用 JIT）
NUMBA_DISABLE_JIT=1 pytest -q

# 运行带覆盖率的测试
NUMBA_DISABLE_JIT=1 pytest tests/ --cov=afmlkit --cov-report=term -v

# 运行单个测试文件
pytest tests/bars/test_comp_ohlcv.py -v
```

## 文档

完整文档请访问 [afmlkit.readthedocs.io](https://afmlkit.readthedocs.io)

## 贡献

欢迎贡献代码！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 参考文献

本项目方法论主要基于：

- López de Prado, Marcos. *Advances in Financial Machine Learning*. Wiley, 2018.

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AFMLKit 是一个基于 *Advances in Financial Machine Learning*（Marcos López de Prado 著）方法论的高性能金融机器学习工具包。使用 Numba 进行 JIT 编译加速，专为处理大规模金融数据而设计。

## 常用命令

```bash
# 安装（开发模式）
pip install -e .[dev]

# 测试（推荐禁用 JIT，与 CI 一致）
NUMBA_DISABLE_JIT=1 pytest -q

# 测试带覆盖率
NUMBA_DISABLE_JIT=1 pytest tests/ --cov=afmlkit --cov-report=term -v

# 运行单个测试文件
pytest tests/bars/test_comp_ohlcv.py -v

# 运行单个测试函数
pytest tests/bars/test_comp_ohlcv.py::test_comp_ohlcv -v

# Lint 和格式化
flake8 afmlkit/ && black afmlkit/

# 启动 Web UI
streamlit run webapp/app.py
# 或
python run_webapp.py
```

## Numba JIT 注意事项

**关键**：此代码库使用 Numba 进行高性能计算，测试需要特殊处理。

- **推荐**：设置 `NUMBA_DISABLE_JIT=1` 运行测试
- **原因**：CI 在 JIT 禁用模式下运行（numba 与大规模测试不兼容）
- **本地验证**：JIT 禁用测试通过后，启用 JIT 再次验证生产行为

调试时禁用 JIT（仅用于调试，提交前移除）：
```python
import os
os.environ['NUMBA_DISABLE_JIT'] = "1"  # 必须在导入 numba 函数前
from afmlkit.bar.base import comp_bar_ohlcv
```

## 架构概览

### 核心模块 (`afmlkit/`)

```
afmlkit/
├── bar/           # K 线构建（Time, Tick, Volume, Dollar, CUSUM, Imbalance, Run bars）
│   ├── base.py    # 基础类和核心 Numba 函数
│   ├── logic.py   # Numba 加速的索引器
│   ├── kit.py     # 具体 Bar 实现的 Kit 类
│   └── data_model.py  # TradesData 数据模型
├── feature/       # 特征工程框架（流式接口）
│   ├── base.py    # Transform 基类
│   ├── core/      # 核心特征（MA, Volatility, Momentum, Trend, Reversion, Volume, Correlation, Time）
│   └── kit.py     # Feature 和 FeatureKit API
├── label/         # 标签生成
│   ├── tbm.py     # 三重屏障法核心实现
│   ├── weights.py # 样本权重计算
│   └── kit.py     # TBMLabel API
├── sampling/      # 采样过滤器（CUSUM filter）
└── utils/         # 工具函数
```

### Web UI (`webapp/`)

基于 Streamlit 的交互式研究界面，遵循数据管道模式：

```
数据导入 → K 线构建 → 特征工程 → 标签生成 → 特征分析 → 模型训练 → 回测评估
```

关键文件：
- `app.py` - 主入口
- `session.py` - 会话状态管理（跨页面数据流）
- `pages/` - 各功能页面
- `components/` - 可复用 UI 组件
- `utils/` - 工具函数

## 代码风格

### 导入顺序
三组导入，用空行分隔：标准库 → 第三方 → 本地导入

### 类型注解
```python
from numpy.typing import NDArray
def process_data(
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    threshold: Optional[float] = None
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
```

### 命名约定
- 函数：`snake_case`（如 `comp_bar_ohlcv`）
- 类：`PascalCase`（如 `BarBuilderBase`）
- 私有函数：`_前缀`（如 `_time_bar_indexer`）
- 常量：`UPPER_SNAKE_CASE`

### Numba 函数
```python
@njit(nogil=True)
def _process_array(data: NDArray[np.float64]) -> NDArray[np.float64]:
    ...

@njit(nogil=True, parallel=True)
def _process_parallel(data: NDArray[np.float64]) -> NDArray[np.float64]:
    for i in prange(n):  # 使用 prange 而非 range
        ...
```

### Docstring
使用 reStructuredText 格式：
```python
def compute_returns(prices: NDArray[np.float64], log: bool = True) -> NDArray[np.float64]:
    """
    Compute returns from a price series.

    :param prices: Array of price values.
    :param log: If True, compute log returns.
    :returns: Array of return values.
    :raises ValueError: If prices has fewer than 2 elements.
    """
```

## 关键约定

- **DataFrame 索引**：时间序列使用 datetime index，内部存储为纳秒 (`int64`)
- **数组类型**：`np.float64`（价格/成交量）、`np.int64`（索引/计数）、`np.int8`（分类值）
- **错误处理**：在调用 Numba 函数前验证输入，Numba 函数内部不处理异常
- **日志**：`from afmlkit.utils.log import get_logger`

## 参考资源

- 方法论：*Advances in Financial Machine Learning* by Marcos López de Prado
- 文档：https://afmlkit.readthedocs.io
- 测试指南：[tests/README.md](tests/README.md)
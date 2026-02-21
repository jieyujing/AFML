# AGENTS Instructions

AI 编码代理在 AFMLKit 仓库中工作的必备指南。Python 版本：3.10+

---

## 快速参考

```bash
pip install -e .[dev]                                      # 安装
NUMBA_DISABLE_JIT=1 pytest -q                              # 测试
pytest tests/bars/test_comp_ohlcv.py::test_comp_ohlcv -v   # 单测试
flake8 afmlkit/ && black afmlkit/                          # 检查
```

---

## 构建与测试

```bash
# 安装（开始工作前必须执行）
pip install -e .[dev]

# 测试命令
NUMBA_DISABLE_JIT=1 pytest -q                              # 完整套件
NUMBA_DISABLE_JIT=1 pytest tests/ --cov=afmlkit -v         # 带覆盖率
pytest tests/bars/test_comp_ohlcv.py -v                    # 单文件
pytest tests/bars/test_comp_ohlcv.py::test_comp_ohlcv -v   # 单函数

# 辅助脚本
./local_test.sh          # JIT 启用
./local_test_nojit.sh    # JIT 禁用（CI 风格）

# Lint
flake8 afmlkit/
black afmlkit/
```

---

## Numba JIT 注意事项

**关键**：此代码库使用 Numba 进行高性能计算，测试需要特殊处理。

- **推荐**：设置 `NUMBA_DISABLE_JIT=1` 运行测试
- **原因**：CI 在 JIT 禁用模式下运行（numba 与大规模测试不兼容）
- **本地验证**：JIT 禁用测试通过后，启用 JIT 再次验证生产行为

**调试时禁用 JIT（仅用于调试，提交前移除）：**
```python
import os
os.environ['NUMBA_DISABLE_JIT'] = "1"  # 必须在导入 numba 函数前
from afmlkit.bar.base import comp_bar_ohlcv
```

---

## 代码风格

### 导入顺序

三组导入，用空行分隔：标准库 → 第三方 → 本地导入

```python
# 标准库
from abc import ABC, abstractmethod
from typing import Tuple, Optional

# 第三方
import numpy as np
import pandas as pd
from numba import njit, prange
from numpy.typing import NDArray

# 本地导入
from .data_model import TradesData
from afmlkit.utils.log import get_logger
```

### 类型注解

```python
def process_data(
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    threshold: Optional[float] = None
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    ...
```

### 命名约定

| 类型 | 约定 | 示例 |
|------|------|------|
| 函数 | `snake_case` | `comp_bar_ohlcv` |
| 类 | `PascalCase` | `BarBuilderBase` |
| 私有函数 | `_前缀` | `_time_bar_indexer` |
| 常量 | `UPPER_SNAKE_CASE` | `DEFAULT_THRESHOLD` |

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

### 错误处理

```python
def public_api(prices, volumes):
    # 在调用 Numba 函数前验证
    if len(prices) != len(volumes):
        raise ValueError("Array lengths must match.")
    return _numba_implementation(prices, volumes)
```

---

## 项目结构

```
afmlkit/
├── bar/           # K线构建（time, tick, volume, dollar bars）
├── feature/       # 特征工程框架
├── label/         # 标签方法（Triple Barrier Method）
├── sampling/      # 采样过滤器（CUSUM filter）
└── utils/         # 工具函数

tests/
├── bars/          # bar 模块测试
├── features/      # feature 模块测试
└── utils.py       # 测试工具
```

---

## 关键约定

- **DataFrame 索引**：时间序列使用 datetime index，内部存储为纳秒 (`int64`)
- **数组类型**：`np.float64`（价格/成交量）、`np.int64`（索引/计数）、`np.int8`（分类值）
- **日志**：`from afmlkit.utils.log import get_logger`

---

## Docstring 风格

使用 **reStructuredText** 格式：

```python
def compute_returns(prices: NDArray[np.float64], log: bool = True) -> NDArray[np.float64]:
    """
    Compute returns from a price series.

    :param prices: Array of price values.
    :param log: If True, compute log returns.
    :returns: Array of return values.
    :raises ValueError: If prices has fewer than 2 elements.
    """
    ...
```

---

## 参考资源

- 方法论：*Advances in Financial Machine Learning* by Marcos López de Prado
- 文档：https://afmlkit.readthedocs.io
- 测试指南：[tests/README.md](tests/README.md)

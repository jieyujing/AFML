# Alpha 因子库

聚宽内置了常见的 Alpha101 和 Alpha191 因子。

## 1. Alpha101 因子
**来源**：WorldQuant 发布的 101 个 Alpha 因子。

**调用方法**：
```python
from jqlib.alpha101 import *
# 获取 alpha_001 因子值
res = alpha_001(security_list, date)
```

## 2. Alpha191 因子
**来源**：国泰君安发布的 191 个短周期预测因子。

**调用方法**：
```python
from jqlib.alpha191 import *
# 获取 alpha_001 因子值
res = alpha_001(security_list, date)
```

## 3. 聚宽因子库 (Factor)
聚宽提供了更丰富的预计算因子，支持横截面查询。

**示例**：
```python
from jqfactor import get_factor_values
# 获取 ROE 因子
factors = get_factor_values(securities=['000001.XSHE'], factors=['roe'], start_date='2023-01-01', end_date='2023-01-31')
```

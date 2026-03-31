# ADF 自动滞后选择功能设计

**日期**: 2026-03-31
**状态**: 已批准
**范围**: `afmlkit/feature/core/structural_break/adf.py`

---

## 背景

当前 AFMLKit 的 `adf_test` 只实现基本 DF 检验（lag=0），不支持自动滞后选择。完整的 ADF（Augmented Dickey-Fuller）检验需要包含滞后差分项以消除序列相关性，提高检验准确性。

项目中 `frac_diff.py` 已使用 statsmodels 的 `adfuller`（含自动滞后），但核心 ADF 模块追求 Numba JIT 加速，需要自主实现。

---

## 需求

| 决策点 | 选择 |
|--------|------|
| 滞后选择方法 | AIC 信息准则 |
| 实现策略 | 混合模式（Python 迭代 + Numba 核心） |
| API 设计 | 双 API：简洁版 + 完整版 |
| 默认 maxlag | Schwert 公式自动计算 |
| 错误处理 | 返回 NaN + 日志可观测性 |

---

## API 设计

### 简洁 API（保持现有风格）

```python
def adf_test(y, max_lag=None, trend=True):
    """
    ADF 检验，自动滞后选择。

    :param y: 价格序列
    :param max_lag: None 表示自动选择（Schwert 公式 + AIC）；int 表示固定滞后
    :param trend: 是否包含时间趋势项
    :returns: (t_statistic, p_value, selected_lag)
    """
```

### 完整 API（statsmodels 兼容）

```python
def adf_test_full(y, max_lag=None, trend=True):
    """
    ADF 检验完整结果。

    :param y: 价格序列
    :param max_lag: None 表示自动选择；int 表示固定滞后
    :param trend: 是否包含时间趋势项
    :returns: (adf_stat, pvalue, used_lag, nobs, critical_values, icbest)
              - critical_values: dict {'1%': ..., '5%': ..., '10%': ...}
    """
```

### 行为变更

- `max_lag=None`（默认）→ 自动滞后选择
- `max_lag=int` → 固定使用指定滞后（不自动选择）

---

## 核心实现

### 新增 Numba 函数

```python
@njit(nogil=True)
def _build_adf_design_matrix(y, lag, trend):
    """
    构建 ADF 回归设计矩阵。

    模型: dy_t = alpha + beta*t + gamma*y_{t-1} + sum(delta_i * dy_{t-i}) + epsilon

    :param y: 价格序列
    :param lag: 滞后阶数（差分项数量）
    :param trend: 是否包含时间趋势
    :returns: (X, dy, gamma_idx)
              - X: 设计矩阵 (n_obs, n_features)
              - dy: 目标向量（一阶差分）
              - gamma_idx: gamma 系数在 beta 中的索引
    """

@njit(nogil=True)
def _compute_aic(n_obs, n_params, rss):
    """
    计算 AIC = n * log(rss/n) + 2 * k

    :param n_obs: 观测数量
    :param n_params: 参数数量（含常数项）
    :param rss: 残差平方和
    :returns: AIC 值
    """

@njit(nogil=True)
def _adf_regression_with_lag(y, lag, trend):
    """
    带滞后项的 ADF 回归。

    :param y: 价格序列
    :param lag: 滞后阶数
    :param trend: 是否包含趋势
    :returns: (t_statistic, p_value, rss, n_params, n_obs, success)
              - success: bool，表示计算是否成功（矩阵非奇异）
    """
```

### 新增 Python 函数

```python
def schwert_maxlag(n, method='standard'):
    """
    Schwert 公式计算最大滞后。

    公式: int(12 * (n/100)**(1/4))

    :param n: 样本量
    :param method: 'standard' 或 'modified'
    :returns: 最大滞后阶数
    """

def _select_lag_by_aic(y, max_lag, trend):
    """
    AIC 自动滞后选择。

    Python 层迭代，调用 Numba 核心，选择 AIC 最小的滞后阶。

    :param y: 价格序列
    :param max_lag: 最大滞后
    :param trend: 是否包含趋势
    :returns: (best_lag, t_stat, p_value, best_aic)
    """
```

---

## 数据流

```
用户调用 adf_test(y, max_lag=None, trend=True)
    ↓
计算 Schwert maxlag = int(12 * (n/100)**(1/4))
    ↓
_select_lag_by_aic(y, max_lag, trend)  [Python 层迭代]
    ↓ 循环 lag = 0, 1, 2, ..., max_lag
    _adf_regression_with_lag(y, lag, trend)  [Numba 核心]
        ↓
        _build_adf_design_matrix(y, lag, trend)  → X, dy
        ↓
        OLS 回归 → t_stat, rss, n_params
        ↓
        _compute_aic(n_obs, n_params, rss) → aic
    ↓
选择 AIC 最小的 lag → (best_lag, best_t_stat, best_p_value)
    ↓
返回 (t_stat, p_value, best_lag)
```

---

## 边界情况与错误处理

```python
def adf_test(y, max_lag=None, trend=True):
    try:
        # 1. 样本量检查
        n = len(y)
        if n < 10:
            logger.warning(f"ADF: 样本量过小 ({n})，返回 NaN")
            return (np.nan, np.nan, np.nan)

        # 2. 计算 Schwert maxlag
        if max_lag is None:
            max_lag = schwert_maxlag(n)

        # 3. 滞后截断检查
        if max_lag > n // 2:
            logger.warning(f"ADF: 滞后截断 ({max_lag} -> {n//2})")
            max_lag = n // 2

        # 4. AIC 自动滞后选择
        best_lag, t_stat, p_value, aic = _select_lag_by_aic(y, max_lag, trend)

        # 5. 检查结果有效性
        if np.isnan(t_stat):
            logger.warning("ADF: 设计矩阵奇异或计算失败")
            return (np.nan, np.nan, np.nan)

        logger.debug(f"ADF: 选择滞后={best_lag}, AIC={aic:.2f}")
        return (t_stat, p_value, best_lag)

    except Exception as e:
        logger.error(f"ADF: 未捕获错误 - {e}")
        return (np.nan, np.nan, np.nan)
```

### 错误处理策略

| 场景 | 处理方式 |
|------|----------|
| `n < 10` | 返回 `(nan, nan, nan)` + warning 日志 |
| `max_lag > n // 2` | 自动截断为 `n // 2` + warning 日志 |
| 设计矩阵奇异（det≈0） | 返回 `(nan, nan, nan)` + warning 日志 |
| `lag=0` 时无足够数据 | 返回 `(nan, nan, nan)` |
| AIC 比较中某 lag 失败 | 跳过该 lag，继续比较其他 |
| 未预期异常 | 返回 `(nan, nan, nan)` + error 日志 |

### 日志级别

- `warning`：数据问题（样本量小、滞后截断、奇异矩阵）
- `debug`：正常执行信息（选择的滞后、AIC 值）
- `error`：未预期异常

---

## 测试策略

### 新增测试用例

| 测试 | 内容 |
|------|------|
| `test_schwert_maxlag` | 验证 Schwert 公式计算正确 |
| `test_adf_regression_with_lag` | 验证带滞后项的回归 |
| `test_select_lag_by_aic` | 验证 AIC 选择逻辑 |
| `test_adf_test_auto_lag` | 自动滞后 vs 固定滞后对比 |
| `test_adf_test_full` | 验证完整 API 返回值 |
| `test_adf_test_small_sample` | 边界：样本量 < 10 |
| `test_adf_test_singular_matrix` | 边界：奇异矩阵处理 |
| `test_adf_vs_statsmodels` | 与 statsmodels 结果对比验证 |

### 对比验证

```python
def test_adf_vs_statsmodels():
    """与 statsmodels 结果对比（允许小误差）"""
    y = generate_random_walk(100)

    # statsmodels 结果
    sm_result = adfuller(y, autolag='AIC', regression='ct')

    # AFMLKit 结果
    ak_result = adf_test_full(y, trend=True)

    # 验证核心指标（允许 5% 相对误差）
    assert np.isclose(ak_result[0], sm_result[0], rtol=0.05)  # t_stat
    assert ak_result[2] == sm_result[2]  # used_lag
```

---

## 文件结构变更

```
afmlkit/feature/core/structural_break/
├── adf.py                         # 主要修改
│   ├── schwert_maxlag()           # 新增
│   ├── _build_adf_design_matrix() # 新增
│   ├── _compute_aic()             # 新增
│   ├── _adf_regression_with_lag() # 新增
│   ├── _select_lag_by_aic()       # 新增
│   ├── adf_test()                 # 修改（支持自动滞后）
│   ├── adf_test_full()            # 新增
│   └── adf_test_rolling()         # 修改（可选支持自动滞后）
├── __init__.py                    # 更新导出
│   └── 新增 adf_test_full 导出
tests/structural_breaks/
├── test_adf.py                    # 扩展测试
│   └── 新增 8 个测试用例
```

---

## 不在范围内

- BIC / t-stat 递减法滞后选择（仅实现 AIC）
- 滚动 ADF 的自动滞后（可选，不在本次范围）
- `trend='ctt'` 或 `'n'` 模式（仅支持 `'c'` 和 `'ct'`）

---

## 参考

- Schwert, G.W. (1989). "Tests for Unit Roots: A Monte Carlo Investigation"
- Ng, S. & Perron, P. (1995). "Unit Root Tests in ARMA Models with Data-Dependent Methods for the Selection of the Truncation Lag"
- statsmodels.tsa.stattools.adfuller 实现
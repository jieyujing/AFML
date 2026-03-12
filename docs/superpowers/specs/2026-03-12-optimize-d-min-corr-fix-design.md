# optimize_d min_corr 参数修复设计文档

## 1. 问题描述

`webapp/utils/alpha158_features.py` 调用 `optimize_d()` 时传入 `min_corr` 参数，但该函数不接受此参数，导致 `TypeError`。

**错误信息：**
```
TypeError: optimize_d() got an unexpected keyword argument 'min_corr'
```

## 2. 根本原因

`afmlkit/feature/core/frac_diff.py` 中的 `optimize_d()` 函数签名缺少 `min_corr` 参数，但：
- 调用代码期望该参数存在
- 函数内部已经计算了相关性（第 100-103 行），只是没有用于决策

## 3. 设计方案

### 3.1 修改 `optimize_d` 函数签名

添加 `min_corr` 参数，默认值为 0.0（保持向后兼容）：

```python
def optimize_d(
    series: pd.Series,
    thres: float = 1e-4,
    d_step: float = 0.05,
    max_d: float = 1.0,
    min_corr: float = 0.0,  # 新增参数
) -> float:
```

### 3.2 修改选择逻辑

在找到使序列平稳的 d 值后，检查相关性是否满足 `min_corr` 要求：

```python
if p_val < 0.05:
    # 计算与原始序列的相关性
    common_idx = diff_series.index.intersection(series.index)
    if len(common_idx) >= 10:
        diff_aligned = diff_series.loc[common_idx]
        orig_aligned = series.loc[common_idx].dropna()
        common_idx = diff_aligned.index.intersection(orig_aligned.index)
        if len(common_idx) >= 10:
            corr = np.corrcoef(
                diff_aligned.loc[common_idx].values,
                orig_aligned.loc[common_idx].values
            )[0, 1]

            # 新增：检查相关性是否满足要求
            if corr >= min_corr:
                return float(round(d, 4))
            # 如果不满足相关性要求，继续搜索更大的 d
            continue

    # 如果无法计算相关性但序列平稳，返回当前 d
    return float(round(d, 4))
```

### 3.3 更新文档字符串

```python
"""
寻找使序列平稳的最小 d 值。

:param series: 输入价格序列
:param thres: FFD 权重截断阈值
:param d_step: d 的搜索步长，默认 0.05
:param max_d: d 的最大搜索值，默认 1.0（极端情况下可调至 2.0）
:param min_corr: 与原始序列的最小相关性阈值，默认 0.0（不约束）。
                 推荐值：0.9-0.95 以保持高记忆性
:return: 最优 d 值
"""
```

## 4. 技术细节

### 4.1 向后兼容性
- 默认值 `min_corr=0.0` 确保现有代码行为不变
- 所有现有调用无需修改

### 4.2 搜索逻辑
- 当 `min_corr > 0` 时，算法会找到**同时满足平稳性和相关性要求**的最小 d 值
- 如果某个 d 使序列平稳但相关性不足，继续搜索更大的 d

### 4.3 边界情况处理
- 无法计算相关性（数据不足）：返回当前 d
- 所有 d 都无法满足要求：返回 `max_d`

## 5. 测试验证

### 5.1 单元测试
```python
def test_optimize_d_with_min_corr():
    """测试 min_corr 参数是否生效"""
    # 创建非平稳序列
    prices = pd.Series(np.cumsum(np.random.randn(100)))

    # min_corr=0 时应返回较小的 d
    d_no_corr = optimize_d(prices, min_corr=0.0)

    # min_corr=0.9 时应返回相同或更大的 d
    d_high_corr = optimize_d(prices, min_corr=0.9)

    assert d_high_corr >= d_no_corr
```

### 5.2 集成测试
运行 WebUI 特征工程流程，验证不再报错。

## 6. 文件修改列表

**修改：**
- `afmlkit/feature/core/frac_diff.py:59-109` - `optimize_d` 函数

**无需修改：**
- `webapp/utils/alpha158_features.py` - 调用代码已正确使用参数

## 7. 完成标准

- [ ] `optimize_d` 函数接受 `min_corr` 参数
- [ ] 相关性检查逻辑正确实现
- [ ] 文档字符串已更新
- [ ] 现有测试通过
- [ ] WebUI 特征计算不再报错

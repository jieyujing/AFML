# optimize_d min_corr 参数修复实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `optimize_d()` 函数缺少 `min_corr` 参数导致的 TypeError，使 WebUI 特征计算正常工作。

**Architecture:** 在 `optimize_d()` 函数中添加 `min_corr` 参数和相关性检查逻辑，当序列平稳且与原始序列相关性满足要求时才返回 d 值。

**Tech Stack:** Python, NumPy, Pandas, statsmodels (ADF 检验)

---

## 文件结构

**Modify:**
- `afmlkit/feature/core/frac_diff.py:59-109` - `optimize_d` 函数

**Test:**
- `tests/test_frac_diff_corr.py` - 已有测试文件，可能需要更新

---

### Task 1: 修改 optimize_d 函数

**Files:**
- Modify: `afmlkit/feature/core/frac_diff.py:59-109`

- [ ] **Step 1: 读取当前 optimize_d 函数**

```bash
cat afmlkit/feature/core/frac_diff.py
```

确认当前函数签名和实现。

- [ ] **Step 2: 修改函数签名**

将函数签名从：
```python
def optimize_d(
    series: pd.Series,
    thres: float = 1e-4,
    d_step: float = 0.05,
    max_d: float = 1.0,
) -> float:
```

修改为：
```python
def optimize_d(
    series: pd.Series,
    thres: float = 1e-4,
    d_step: float = 0.05,
    max_d: float = 1.0,
    min_corr: float = 0.0,
) -> float:
```

- [ ] **Step 3: 更新文档字符串**

将 docstring 从：
```python
"""
寻找使序列平稳的最小 d 值。

:param series: 输入价格序列
:param thres: FFD 权重截断阈值
:param d_step: d 的搜索步长，默认 0.05
:param max_d: d 的最大搜索值，默认 1.0（极端情况下可调至 2.0）
:return: 最优 d 值
"""
```

修改为：
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

- [ ] **Step 4: 修改选择逻辑**

将原来的逻辑（第 91-106 行）：
```python
# 熔断机制：p < 0.05 立即停止
if p_val < 0.05:
    # 计算与原始序列的相关性（仅日志观察，不干预选择）
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
            # 日志输出相关性供观察
            # print(f"[optimize_d] d*={d:.4f}, corr={corr:.4f}")
    return float(round(d, 4))
```

修改为：
```python
# 检查平稳性和相关性
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

            # 如果相关性满足要求，返回当前 d
            if corr >= min_corr:
                return float(round(d, 4))
            # 相关性不足，继续搜索更大的 d
            continue

    # 无法计算相关性但序列平稳，返回当前 d
    return float(round(d, 4))
```

- [ ] **Step 5: 提交**

```bash
git add afmlkit/feature/core/frac_diff.py
git commit -m "feat: add min_corr parameter to optimize_d for memory retention"
```

---

### Task 2: 编写并运行单元测试

**Files:**
- Modify: `tests/test_frac_diff_corr.py`
- Test: `tests/test_frac_diff_corr.py::test_optimize_d_with_min_corr`

- [ ] **Step 1: 读取现有测试文件**

```bash
cat tests/test_frac_diff_corr.py
```

查看现有测试结构。

- [ ] **Step 2: 添加测试用例**

在测试文件中添加新测试：
```python
def test_optimize_d_with_min_corr():
    """测试 min_corr 参数是否生效"""
    from afmlkit.feature.core.frac_diff import optimize_d

    # 创建非平稳序列（随机游走）
    np.random.seed(42)
    prices = pd.Series(np.cumsum(np.random.randn(100)))

    # min_corr=0 时应返回某个 d 值
    d_no_corr = optimize_d(prices, min_corr=0.0)
    assert isinstance(d_no_corr, float)
    assert 0.0 <= d_no_corr <= 1.0

    # min_corr=0.9 时应返回相同或更大的 d（更严格的要求）
    d_high_corr = optimize_d(prices, min_corr=0.9)
    assert isinstance(d_high_corr, float)
    assert 0.0 <= d_high_corr <= 1.0

    # 高相关性要求通常需要更大的 d 或相等的 d
    assert d_high_corr >= d_no_corr
```

- [ ] **Step 3: 运行测试验证**

```bash
cd /Users/link/Documents/AFMLKIT
python3 -m pytest tests/test_frac_diff_corr.py::test_optimize_d_with_min_corr -v
```

预期：测试通过

- [ ] **Step 4: 运行所有相关测试**

```bash
python3 -m pytest tests/test_frac_diff_corr.py -v
```

预期：所有测试通过

- [ ] **Step 5: 提交测试**

```bash
git add tests/test_frac_diff_corr.py
git commit -m "test: add min_corr parameter test for optimize_d"
```

---

### Task 3: 验证 WebUI 特征计算

**Files:**
- Verify: `webapp/utils/alpha158_features.py:74`

- [ ] **Step 1: 确认调用代码**

确认调用代码正确使用新参数：
```python
# webapp/utils/alpha158_features.py:74
optimal_d = optimize_d(log_price, thres=thres, d_step=d_step, min_corr=min_corr)
```

这行代码现在应该可以正常工作。

- [ ] **Step 2: 运行集成测试（如果有）**

```bash
python3 -m pytest tests/webapp/test_alpha158_features.py -v
```

预期：测试通过

- [ ] **Step 3: 手动验证（可选）**

如果环境允许，启动 WebUI 并运行特征计算：
```bash
cd /Users/link/Documents/AFMLKIT/webapp
streamlit run app.py
```

然后在浏览器中：
1. 导航到 "特征工程" 页面
2. 执行特征计算
3. 确认不再出现 `TypeError: optimize_d() got an unexpected keyword argument 'min_corr'`

---

## 完成标准

- [ ] Task 1: optimize_d 函数修改完成
- [ ] Task 2: 单元测试通过
- [ ] Task 3: WebUI 特征计算不再报错
- [ ] 所有 git 提交完成

---

## 回滚方案

如果修改后出现问题，使用以下命令回滚：

```bash
git revert HEAD~3..HEAD
```

或者恢复原始函数签名和逻辑。

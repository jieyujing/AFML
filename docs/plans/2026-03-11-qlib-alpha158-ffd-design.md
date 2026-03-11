# Qlib Alpha158 特征改造设计（FFD 适配版）

> 创建日期：2026-03-11
> 状态：已批准
> 实施优先级：Phase 1（核心子集）

---

## 概述

本设计文档描述如何将 Qlib Alpha158 因子库改造为符合 AFMLKIT 规范的特征工程模块。核心改造理念是使用**分数阶差分（FFD）序列**替代传统的一阶差分/收益率因子，以在保持平稳性的同时保留更多历史记忆。

---

## 改造原则

### 三步改造动作

1. **提取底座**
   只对极其底层的非平稳数据（如对数价格 `log_price`）运行固定窗口的分数阶差分（FFD），求出平稳且带记忆的序列 `X̃`。

2. **平替一阶差分特征**
   在特征矩阵里，直接用这个 FFD 序列 `X̃` 替代 Qlib158 里的所有简单动量（Momentum）或收益率（Return）因子。不要再去算平稳序列的收益率。

3. **安全使用平滑与截面指标**
   对于 Qlib 里的横向比较（如 Cross-sectional Rank 截面排行）或平滑操作（如 EMA、SMA 等均线计算），直接把 FFD 序列 `X̃` 喂进去。因为均线和排序不会增加差分阶数，它们只是在平滑噪音或提取相对位置。

### 最优 d* 搜索标准

在实盘代码中，不要去猜。写一个循环，让 d 从 0 开始，每次增加一点点（比如 0.05），对每一个 FFD(d) 的序列计算 ADF 统计量的 p-value。只要 p-value 第一次跌破 5%（0.05 及格线），立刻停止，这个当前的 d 就是你的最优阶数 d*。

---

## 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 实施策略 | 分阶段实施 | 先做核心子集，验证后再扩展 |
| 均线计算 | 直接对 FFD 序列 | `SMA(ffd_log_price)` 保持平稳性 |
| d* 计算频率 | 每只股票一次 | 避免 look-ahead bias，训练期确定后固定 |
| 特征命名 | `ffd_*` 前缀 | 清晰标识基于 FFD 的特征 |
| 动量计算 | FFD 序列本身 | FFD 已提取变化率，无需再次差分 |
| 成交量处理 | 保持原始计算 | 成交量通常已是平稳序列 |
| 截面 Rank | 时序 Rank（单股票） | 当前单品种场景，后续扩展 |
| 集成方式 | 与现有特征并列 | 向后兼容，可对比效果 |

---

## 特征架构

### 数据流

```
输入：OHLCV 数据 (DatetimeIndex)
         ↓
    ┌────┴────┐
    ↓         ↓
log_price   volume/high/low/open
    ↓         ↓
 optimize_d() ↓
    ↓         ↓
  d* 固定     原始量价特征
    ↓         ↓
frac_diff_ffd()  ──→ ffd_log_price
                     ↓
         ┌───────────┼───────────┐
         ↓           ↓           ↓
   波动率类      均线类      时序 Rank
   ffd_vol_*    ffd_ma_*    ffd_rank_*
         ↓           ↓           ↓
         └───────────┼───────────┘
                     ↓
         ┌───────────┴───────────┐
         ↓                       ↓
    量价类 (原始)           合并输出
    ffd_vwap, etc.      (全部 ffd_* 前缀)
         ↓                       ↓
         └───────────┬───────────┘
                     ↓
              NaN 清理 + 元数据
                     ↓
              特征矩阵 DataFrame
```

---

## 特征清单（Phase 1）

### 底座序列

| 特征名 | 计算逻辑 | 说明 |
|--------|----------|------|
| `ffd_log_price` | `frac_diff_ffd(log(close), d=d*)` | 平稳且带记忆的底座序列 |

### 波动率类（基于 FFD 序列）

| 特征名 | 计算逻辑 | 对应 Qlib158 |
|--------|----------|-------------|
| `ffd_vol_std_5` | `STD(ffd_log_price, 5)` | `STD_5` |
| `ffd_vol_std_10` | `STD(ffd_log_price, 10)` | `STD_10` |
| `ffd_vol_std_20` | `STD(ffd_log_price, 20)` | `STD_20` |
| `ffd_vol_ewm_5` | `EWM(ffd_log_price, 5)` | - |
| `ffd_vol_ewm_10` | `EWM(ffd_log_price, 10)` | - |

### 均线类（基于 FFD 序列）

| 特征名 | 计算逻辑 | 对应 Qlib158 |
|--------|----------|-------------|
| `ffd_ma_5` | `SMA(ffd_log_price, 5)` | `MA_5` |
| `ffd_ma_10` | `SMA(ffd_log_price, 10)` | `MA_10` |
| `ffd_ma_20` | `SMA(ffd_log_price, 20)` | `MA_20` |
| `ffd_ema_5` | `EMA(ffd_log_price, 5)` | `EMA_5` |
| `ffd_ema_10` | `EMA(ffd_log_price, 10)` | `EMA_10` |

### 动量类（FFD 序列本身）

| 特征名 | 计算逻辑 | 对应 Qlib158 |
|--------|----------|-------------|
| `ffd_mom` | `ffd_log_price` | 替代 `MOMENTUM_*`, `RETURNS_*` |

### 量价类（保持原始计算）

| 特征名 | 计算逻辑 | 对应 Qlib158 |
|--------|----------|-------------|
| `ffd_vwap` | `(close*volume).sum()/volume.sum()` | `VWAP` |
| `ffd_amount` | `close * volume` | `AMOUNT` |
| `ffd_amplification` | `(high-low)/(open-prev_close)` | `AMPLIFICATION` |

### 时序排序类（单股票时序 Rank）

| 特征名 | 计算逻辑 | 对应 Qlib158 |
|--------|----------|-------------|
| `ffd_rank_ma_5` | `percentile_rank(ffd_ma_5, 20)` | `RANK_*` 类 |
| `ffd_rank_vol_10` | `percentile_rank(ffd_vol_std_10, 20)` | - |

---

## API 设计

### 核心函数

```python
def compute_ffd_base(
    close: pd.Series,
    thres: float = 1e-4,
    d_step: float = 0.05
) -> Tuple[pd.Series, float]:
    """
    计算 FFD 底座序列并返回最优 d*

    Args:
        close: 收盘价序列
        thres: FFD 权重截断阈值
        d_step: d 搜索步长

    Returns:
        tuple: (ffd_log_price 序列，最优 d 值)
    """

def compute_ffd_volatility(
    ffd_series: pd.Series,
    spans: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    基于 FFD 序列计算波动率特征

    Args:
        ffd_series: FFD 处理后的序列
        spans: 波动率计算窗口列表

    Returns:
        DataFrame: 包含 ffd_vol_std_* 和 ffd_vol_ewm_* 列
    """

def compute_ffd_ma(
    ffd_series: pd.Series,
    windows: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    基于 FFD 序列计算均线特征

    Args:
        ffd_series: FFD 处理后的序列
        windows: 均线窗口列表

    Returns:
        DataFrame: 包含 ffd_ma_* 和 ffd_ema_* 列
    """

def compute_ffd_rank(
    df: pd.DataFrame,
    feature_cols: List[str],
    rank_window: int = 20
) -> pd.DataFrame:
    """
    计算时序排序特征（单股票历史百分位）

    Args:
        df: 包含基础特征的 DataFrame
        feature_cols: 要计算 Rank 的特征列
        rank_window: 滚动窗口大小

    Returns:
        DataFrame: 包含 ffd_rank_* 列
    """

def compute_alpha158_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    计算完整 Alpha158 风格特征（FFD 改造版）

    Args:
        df: 输入 DataFrame (必须包含 OHLCV 列)
        config: 特征配置字典

    Returns:
        tuple: (特征矩阵 DataFrame, 元数据字典)

    元数据包含:
        - optimal_d: FFD 最优参数 d*
        - feature_columns: 计算的特征列列表
        - rows_before_clean: 清理前行数
        - rows_after_clean: 清理后行数
    """
```

### 与现有系统集成

```python
# 在 webapp/utils/feature_calculator.py 的 compute_all_features 中

if config.get("alpha158", {}).get("enabled", False):
    from .alpha158_features import compute_alpha158_features
    alpha158_df, alpha158_meta = compute_alpha158_features(df, config)
    # 合并特征（ffd_* 前缀）
    features_df = pd.concat([features_df, alpha158_df], axis=1)
    metadata["alpha158_enabled"] = True
    metadata["alpha158_columns"] = alpha158_df.columns.tolist()
```

---

## 配置示例

```yaml
# 基础 FFD 配置
fractional_diff:
  enabled: true
  threshold: 0.0001       # FFD 权重截断阈值
  d_step: 0.05            # d 值搜索步长
  max_d: 1.0              # 最大 d 值
  min_corr: 0.0           # 最小相关性约束 (0.0-1.0)
                          # 0.0 = 仅平稳性检验
                          # 0.7-0.8 = 推荐默认设置
                          # 0.9-0.95 = 高记忆保留

# Alpha158 配置
alpha158:
  enabled: true
  volatility:
    std_spans: [5, 10, 20]
    ewm_spans: [5, 10]
  ma:
    windows: [5, 10, 20]
    ema_windows: [5, 10]
  volume:
    enabled: true
  rank:
    enabled: true
    window: 20
```

---

## 参数选择指南

### `min_corr` 参数说明

`min_corr` 参数用于在分数差分时平衡**平稳性**和**记忆保留**：

| min_corr 值 | 行为 | 适用场景 |
|-------------|------|----------|
| `0.0` (默认) | 仅使用 ADF 检验，选择最小的平稳 d | 趋势跟踪策略，不需要特征解释性 |
| `0.5-0.7` | 要求中等相关性，适度保留记忆 | 通用场景，平衡性能与解释性 |
| `0.8-0.9` | 要求高相关性，保守差分 | 需要特征与原始价格强相关的场景 |
| `> 0.95` | 可能无法满足，触发回退逻辑 | 不推荐，效果等同于 0.0 |

### 回退机制

当设置的 `min_corr` 过高导致无法同时满足平稳性和相关性时：
1. 算法会遍历所有 d 值 (0.01 → 1.0)
2. 记录第一个满足平稳性的 d
3. 如果找不到满足 `min_corr` 的 d，返回记录的平稳 d

这确保了即使参数设置不合理，算法仍能产生有效结果。

---

## 测试策略

### 单元测试

```python
def test_compute_ffd_base():
    """测试 FFD 底座序列计算"""
    close = pd.Series(np.cumsum(np.random.randn(200)) + 100)
    ffd_series, optimal_d = compute_ffd_base(close)
    assert len(ffd_series) <= len(close)
    assert 0.0 <= optimal_d <= 1.0

def test_compute_ffd_volatility():
    """测试 FFD 波动率特征"""
    ffd_series = pd.Series(np.random.randn(100))
    result = compute_ffd_volatility(ffd_series, spans=[5, 10])
    assert 'ffd_vol_std_5' in result.columns
    assert 'ffd_vol_std_10' in result.columns

def test_compute_alpha158_features():
    """测试完整 Alpha158 特征计算"""
    df = pd.DataFrame({
        'open': np.linspace(100, 110, 200),
        'high': np.linspace(101, 111, 200),
        'low': np.linspace(99, 109, 200),
        'close': np.linspace(100, 110, 200),
        'volume': np.random.exponential(1000, 200)
    }, index=pd.date_range('2023-01-01', periods=200, freq='T'))

    result, metadata = compute_alpha158_features(df)
    assert 'ffd_log_price' in result.columns
    assert 'ffd_ma_5' in result.columns
    assert 'ffd_vol_std_10' in result.columns
    assert 'optimal_d' in metadata
```

### 集成测试

```python
def test_alpha158_with_existing_features():
    """测试 Alpha158 特征与现有特征共存"""
    df, metadata = compute_all_features(
        bar_data,
        config={'alpha158': {'enabled': True}}
    )
    # 验证现有特征存在
    assert 'vol_ewm_10' in df.columns
    # 验证 Alpha158 特征存在
    assert 'ffd_ma_5' in df.columns
```

---

## 实施计划

### Phase 1（核心子集）

- [ ] Task 1: 创建 `webapp/utils/alpha158_features.py` 模块
- [ ] Task 2: 实现 `compute_ffd_base` 函数
- [ ] Task 3: 实现 `compute_ffd_volatility` 函数
- [ ] Task 4: 实现 `compute_ffd_ma` 函数
- [ ] Task 5: 实现 `compute_ffd_rank` 函数
- [ ] Task 6: 实现 `compute_alpha158_features` 入口函数
- [ ] Task 7: 编写单元测试
- [ ] Task 8: 集成到 `feature_calculator.py`
- [ ] Task 9: 更新 UI 配置选项
- [ ] Task 10: 端到端测试

### Phase 2（完整 Alpha158）

- [ ] 扩展更多波动率特征
- [ ] 添加更多量价关系特征
- [ ] 支持多股票截面 Rank
- [ ] 添加特征重要性分析

---

## 依赖关系

```
afmlkit.feature.core.frac_diff
├── frac_diff_ffd
└── optimize_d

webapp.utils.alpha158_features (new)
├── compute_ffd_base
├── compute_ffd_volatility
├── compute_ffd_ma
├── compute_ffd_rank
└── compute_alpha158_features

webapp.utils.feature_calculator (modified)
└── compute_all_features (添加 alpha158 开关)

webapp.pages.03_feature_engineering (modified)
└── 添加 Alpha158 配置选项
```

---

## 验收标准

1. **功能正确性**
   - [ ] 所有 `ffd_*` 特征计算正确
   - [ ] 最优 d* 搜索算法工作正常（p-value < 0.05）
   - [ ] 特征与现有系统兼容共存

2. **性能要求**
   - [ ] 2000 行数据特征计算 < 30 秒
   - [ ] 内存使用合理（无泄漏）

3. **代码质量**
   - [ ] 单元测试覆盖率 > 80%
   - [ ] 通过 ruff 代码检查
   - [ ] 文档完整

---

## 参考资料

- [Qlib Alpha158 文档](https://qlib.readthedocs.io/en/latest/component/alpha158.html)
- [AFMLKIT 分数阶差分规范](../openspec/specs/fractional-differentiation/spec.md)
- [WebUI 特征工程指南](../../webapp/FEATURE_ENGINEERING_GUIDE.md)

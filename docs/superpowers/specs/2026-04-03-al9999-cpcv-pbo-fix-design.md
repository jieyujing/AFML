# AL9999 CPCV/PBO 完整修复设计

日期：2026-04-03

## 背景

经审计分析，AL9999 策略的 CPCV/PBO 实现存在以下问题：

1. **CPCV 路径拼接缺失**：当前实现只切分数据，未正确拼接完整路径
2. **PBO 定义错误**：计算的是 P(SR≤0)，非真正的回测过拟合概率
3. **缺少多策略对比**：PBO 需要多个策略候选进行 IS/OOS 排名对比

本设计定义完整的修复方案，采用渐进式修复策略。

## 设计目标

1. 完成完整审计，输出审计报告
2. 新增正确的 CPCV/PBO 验证脚本
3. 迁移核心逻辑到 afmlkit/validation 模块
4. 标记原有问题脚本为废弃

---

## 第一部分：审计阶段

### 目标

按现有审计设计文档完成完整审计，输出审计报告。

### 输入

**代码文件**：
- `strategies/AL9999/02_feature_engineering.py`
- `strategies/AL9999/04_ma_primary_model.py`
- `strategies/AL9999/06_meta_labels.py`
- `strategies/AL9999/07_meta_model.py`
- `strategies/AL9999/09_pbo_validation.py`
- `strategies/AL9999/10_rolling_backtest.py`
- `strategies/AL9999/feature_compute.py`
- `afmlkit/validation/purged_cv.py`
- `afmlkit/validation/cpcv.py`
- `afmlkit/validation/pbo.py`

**产物文件**：
- `strategies/AL9999/output/features/events_features.parquet`
- `strategies/AL9999/output/features/tbm_results.parquet`
- `strategies/AL9999/output/features/meta_labels.parquet`
- `strategies/AL9999/output/features/rolling_primary_trades.parquet`
- `strategies/AL9999/output/features/rolling_combined_trades.parquet`
- `strategies/AL9999/output/models/meta_oof_signals.parquet`
- `strategies/AL9999/output/models/meta_holdout_signals.parquet`
- `strategies/AL9999/output/models/meta_model.pkl`

### 输出

- `docs/superpowers/audits/2026-04-03-al9999-cpcv-pbo-audit-report.md`
- 包含：逐步骤审计表、前视偏差等级、影响等级、修复优先级

### 审计流程

1. **特征与事件层**：检查特征是否只依赖事件时点及之前的数据
2. **TBM 与标签层**：核对标签生成逻辑，确认未来收益仅作为监督标签
3. **模型训练与验证层**：检查 PurgedKFold 的 purge/embargo 逻辑
4. **回测执行层**：判断是否使用当时可得模型输出
5. **CPCV/PBO 层**：判断是否正确实现路径拼接和 PBO 计算

---

## 第二部分：CPCV 正确实现

### 核心问题

当前 CPCV 只是切分数据，没有正确拼接路径。

### 正确的 CPCV 路径分配

```
n_splits=6, n_test_splits=2 → 15 个组合，10 条路径
每条路径由多个非重叠的 test fold 组成
```

### 路径拼接逻辑

```python
def generate_cpcv_paths(n_splits=6, n_test_splits=2):
    """
    CPCV 路径分配矩阵。
    
    每个组合中的 test fold 分配到不同路径：
    - Fold 1 in combo → Path A
    - Fold 2 in combo → Path B
    """
    from itertools import combinations
    
    all_combos = list(combinations(range(n_splits), n_test_splits))
    n_paths = len(all_combos) * n_test_splits // n_splits
    
    # 路径分配：{(combo_idx, fold_idx): path_idx}
    path_assignments = {}
    for fold in range(n_splits):
        splits_with_fold = [i for i, combo in enumerate(all_combos) if fold in combo]
        for path_idx, split_idx in enumerate(splits_with_fold):
            path_assignments[(split_idx, fold)] = path_idx
    
    return all_combos, n_paths, path_assignments
```

### 新增验证脚本核心流程

```python
# 09b_cpcv_pbo_validation.py 核心流程

# 1. 对每个 CPCV 组合：
#    - 获取 train_idx, test_idx
#    - 对每个参数配置训练模型
#    - 记录 IS 和 OOS Sharpe

# 2. 拼接路径：
#    - 使用 path_assignments 将 test 结果分配到各路径
#    - 每条路径包含完整的 OOS 收益序列

# 3. 计算 PBO：
#    - 对每个策略，计算 IS 最佳配置
#    - 比较该配置在 OOS 的排名
```

---

## 第三部分：PBO 正确实现

### 核心问题

当前 PBO 计算的是 P(SR ≤ 0)，不是真正的回测过拟合概率。

### 正确的 PBO 定义

```
PBO = P[IS 最佳策略在 OOS 排名低于中位数]
```

### 需要的数据结构

```
策略候选（25-100 个）：
  - 参数组合 1: {tbm_barriers: (1.0, 1.0), min_ret: 0.001}
  - 参数组合 2: {tbm_barriers: (1.5, 1.5), min_ret: 0.001}
  - ...

Sharpe 矩阵：
  - sharpe_is[strategy_idx, path_idx]  # 样本内
  - sharpe_oos[strategy_idx, path_idx]  # 样本外
```

### PBO 计算

```python
def calculate_pbo(sharpe_is, sharpe_oos):
    """
    正确的 PBO 计算。
    
    :param sharpe_is: [n_strategies, n_paths] IS Sharpe
    :param sharpe_oos: [n_strategies, n_paths] OOS Sharpe
    :returns: PBO 值 ∈ [0, 1]
    """
    n_strategies, n_paths = sharpe_is.shape
    
    pbo_count = 0
    for p in range(n_paths):
        # 找 IS 最佳策略
        best_is_idx = np.argmax(sharpe_is[:, p])
        
        # 该策略在 OOS 的排名
        oos_rank = np.argsort(np.argsort(sharpe_oos[:, p]))[best_is_idx]
        
        # 如果 OOS 排名低于中位数，则过拟合
        if oos_rank < n_strategies // 2:
            pbo_count += 1
    
    return pbo_count / n_paths
```

### 参数网格（核心参数）

```python
param_grid = {
    'tbm_barriers': [(1.0, 1.0), (1.5, 1.5), (2.0, 2.0), (2.5, 2.5), (3.0, 3.0)],  # 5
    'min_ret': [0.0, 0.001, 0.002, 0.003, 0.005],  # 5
}
# 总组合数：25（可扩展到 50-100）
```

---

## 第四部分：验证流程与产物

### 验证流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        CPCV + PBO 验证流程                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 加载特征与标签数据                                            │
│     ↓                                                            │
│  2. 定义参数网格（25-100 个策略候选）                              │
│     ↓                                                            │
│  3. CPCV 分割数据（n_splits=6, n_test_splits=2 → 15 组合）        │
│     ↓                                                            │
│  4. 对每个组合：                                                  │
│     ├─ Purge 训练集（删除与 test 重叠的样本）                      │
│     ├─ Embargo（留出空白期）                                      │
│     ├─ 对每个参数配置：                                           │
│     │   ├─ 训练 Meta Model（使用 PurgedKFold）                    │
│     │   ├─ 计算 IS Sharpe（训练集内 OOF）                         │
│     │   ├─ 计算 OOS Sharpe（测试集）                              │
│     │   └─ 记录结果                                               │
│     ↓                                                            │
│  5. 拼接 CPCV 路径（15 组合 → 10 条完整路径）                      │
│     ↓                                                            │
│  6. 计算 PBO                                                     │
│     ├─ 对每条路径：找 IS 最佳策略                                  │
│     ├─ 检查该策略在 OOS 的排名                                    │
│     ├─ 统计过拟合次数                                             │
│     ↓                                                            │
│  7. 输出报告                                                     │
│     ├─ Sharpe 分布（IS vs OOS）                                   │
│     ├─ PBO 值与置信区间                                           │
│     ├─ 过拟合风险判断                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 产物输出

| 文件 | 说明 |
|------|------|
| `output/features/cpcv_sharpe_is.parquet` | IS Sharpe 矩阵 [策略 × 路径] |
| `output/features/cpcv_sharpe_oos.parquet` | OOS Sharpe 矩阵 [策略 × 路径] |
| `output/features/cpcv_pbo_results.parquet` | PBO 结果汇总 |
| `output/figures/09b_cpcv_sharpe_distribution.png` | Sharpe 分布对比图 |
| `output/figures/09b_pbo_rank_comparison.png` | IS/OOS 排名对比图 |

---

## 第五部分：迁移与文件结构

### 新增文件结构

```
strategies/AL9999/
├── 09b_cpcv_pbo_validation.py    # 新增：完整 CPCV/PBO 验证
├── 09_pbo_validation.py          # 保留：标记废弃，用于对比
└── output/
    ├── features/
    │   ├── cpcv_sharpe_is.parquet
    │   ├── cpcv_sharpe_oos.parquet
    │   └── cpcv_pbo_results.parquet
    └── figures/
        ├── 09b_cpcv_sharpe_distribution.png
        └── 09b_pbo_rank_comparison.png

afmlkit/validation/
├── cpcv.py                       # 修改：增加路径拼接逻辑
├── pbo.py                        # 修改：支持多策略 PBO
└── __init__.py                   # 更新：导出新函数
```

### 迁移计划

| 阶段 | 内容 | 状态 |
|------|------|------|
| 阶段 1 | 完成 AL9999 审计报告 | 新增 |
| 阶段 2 | 新增 `09b_cpcv_pbo_validation.py` | 新增 |
| 阶段 3 | 验证新脚本输出正确 | 验证 |
| 阶段 4 | 迁移核心函数到 `afmlkit/validation/` | 修改 |
| 阶段 5 | 原 `09_pbo_validation.py` 标记废弃 | 废弃 |

### 废弃标记

```python
# 09_pbo_validation.py 头部添加
"""
⚠️ 已废弃：该文件的 PBO 实现不符合 AFML 定义。

请使用 09b_cpcv_pbo_validation.py 替代。

问题说明：
- CPCV 未正确拼接路径
- PBO 计算的是 P(SR≤0)，非真正的回测过拟合概率
- 缺少多策略对比

保留原因：用于历史对比和审计追溯。
"""
```

---

## 第六部分：afmlkit 迁移细节

### 修改 `afmlkit/validation/cpcv.py`

```python
class CombinatorialPurgedKFold(BaseCrossValidator):
    """CPCV with proper path assembly."""
    
    # 新增方法
    def generate_paths(self, X, strategy_func, y=None):
        """
        生成完整的 CPCV 路径。
        
        :param X: 特征矩阵（带 DatetimeIndex）
        :param strategy_func: 策略函数
            输入 (train_idx, test_idx, param_config)
            返回 (is_sharpe, oos_returns)
        :returns: (returns_paths, sharpe_paths)
        """
        # 初始化路径容器
        path_returns = {p: [] for p in range(self.get_n_paths())}
        
        for split_idx, (train_idx, test_idx, test_folds) in enumerate(self.split(X)):
            # 运行策略
            is_sharpe, oos_returns = strategy_func(train_idx, test_idx)
            
            # 分配到各路径
            for fold_idx, _ in test_folds:
                path_idx = self._get_path_index(split_idx, fold_idx)
                path_returns[path_idx].extend(oos_returns)
        
        # 转换为 Series 并计算 Sharpe
        returns_paths = [pd.Series(r) for r in path_returns.values() if r]
        sharpe_paths = np.array([r.mean()/r.std()*np.sqrt(ann_factor) for r in returns_paths])
        
        return returns_paths, sharpe_paths
```

### 修改 `afmlkit/validation/pbo.py`

```python
def calculate_pbo(
    sharpe_is: np.ndarray,
    sharpe_oos: np.ndarray,
    method: str = 'rank'
) -> Tuple[float, Dict]:
    """
    计算回测过拟合概率。
    
    :param sharpe_is: [n_strategies, n_paths] IS Sharpe
    :param sharpe_oos: [n_strategies, n_paths] OOS Sharpe
    :param method: 'rank' (排名法) 或 'probability' (概率法)
    """
    n_strategies, n_paths = sharpe_is.shape
    
    if method == 'rank':
        # 排名法：IS 最佳策略在 OOS 排名低于中位数的概率
        pbo_count = 0
        for p in range(n_paths):
            best_is_idx = np.argmax(sharpe_is[:, p])
            oos_rank = np.argsort(np.argsort(sharpe_oos[:, p]))[best_is_idx]
            if oos_rank < n_strategies // 2:
                pbo_count += 1
        pbo = pbo_count / n_paths
    
    elif method == 'probability':
        # 概率法：Bailey & López de Prado (2017)
        # PBO = Φ( -μ/σ )
        sr_mean = sharpe_oos.mean(axis=0).mean()
        sr_std = sharpe_oos.mean(axis=0).std()
        pbo = norm.cdf(0, loc=sr_mean, scale=sr_std)
    
    stats = {
        'n_strategies': n_strategies,
        'n_paths': n_paths,
        'pbo': pbo,
        'is_sharpe_mean': sharpe_is.mean(),
        'oos_sharpe_mean': sharpe_oos.mean(),
        'oos_sharpe_std': sharpe_oos.std(),
    }
    
    return pbo, stats
```

### 更新 `afmlkit/validation/__init__.py`

```python
from .cpcv import CombinatorialPurgedKFold, generate_cpcv_paths
from .pbo import calculate_pbo, calculate_pbo_from_returns

__all__ = [
    ...
    'generate_cpcv_paths',  # 新增导出
]
```

---

## 设计总结

| 模块 | 改动 |
|------|------|
| 审计报告 | 新增完整审计文档 |
| AL9999 验证脚本 | 新增 `09b_cpcv_pbo_validation.py` |
| afmlkit/cpcv.py | 新增 `generate_paths()` 方法 |
| afmlkit/pbo.py | 重写 `calculate_pbo()` 支持多策略 |
| 原脚本 | 标记废弃 |

## CPCV 配置

- n_splits = 6
- n_test_splits = 2
- 路径数 = 15

## 参数网格（核心参数）

- tbm_barriers: 5 种
- min_ret: 5 种
- 总组合数: 25（可扩展）
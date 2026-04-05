# AL9999 RF Primary Model 设计文档

**日期**: 2026-04-05
**状态**: Draft → Review
**作者**: Claude Code

## 动机

当前 AL9999 的 Primary Model 使用 CUSUM Direction（`sign(g_up - |g_down|)`）决定交易方向。这是一个**规则式方法**，无法学习多维特征与趋势方向之间的非线性关系。

本设计引入 Random Forest 替代 CUSUM Direction：
- 输入：Phase 2 工程化的技术特征（macro-focused：趋势/动量/形态）
- 标签：Trend Scanning 的 side（±1，剔除此类样本）
- 输出：RF 预测的 trade direction（+1/-1），直接替换现有侧列
- 集成到现有 TBM → Meta Model → Filter-First 回测流程，其余环节不变

设计原则：**特征差异化**——RF Primary 看宏观结构，Meta Model 看微观结构 + RF 置信度，避免两者陷入回音室效应。

## 架构概览

### 数据流

```
03_trend_scanning.py → trend_labels.parquet (side, t_value, t1)
02_feature_engineering.py → events_features.parquet (全部L1/L2/L3特征)
    ↓
04_rf_primary_model.py (新)
    ↓
    - X = events_features (macro subset, 可配置)
    - y = trend_side (±1, side=0 剔除)
    - avgU-weighted Bagging (max_samples = avgU)
    - PurgedKFold + Embargo 交叉验证
    ↓
rf_primary_signals.parquet (side, y_prob, y_pred, t1)
    ↓
04_ma_primary_model.py (修改：新增 'rf' 类型)
    ↓
    - load rf_primary_signals.parquet
    - generate_rf_signals() → 对齐现有信号格式
    - 传递给 TBM 计算（不修改）
    ↓
tbm_results.parquet (格式不变)
    ↓
07_meta_model.py → 10_combined_backtest.py (不变)
```

### 与现有的交互点

| 改动范围 | 修改内容 | 理由 |
|---------|---------|------|
| `04_rf_primary_model.py`（新建） | 完整训练+推理流程 | 解耦训练和推理，独立验证 |
| `04_ma_primary_model.py`（修改） | 增加 `generate_rf_signals()` | 从 parquet 读取 RF side，对齐现有信号格式 |
| `config.py`（修改） | 增加 `RF_PRIMARY_CONFIG` + 扩展 `PRIMARY_MODEL_TYPE` | 支持 'rf' 类型，配置特征子集 |
| `07_meta_model.py`（不变） | 无修改 | TBM label 来源不变，bin 含义不变 |
| `10_combined_backtest.py`（不变） | 无修改 | 消费 TBM 结果，不知道也不关心 side 来源 |

### 配置文件变更

```python
# config.py 新增

# Phase 4: RF Primary Model 参数
RF_PRIMARY_CONFIG = {
    # RF 参数
    'n_estimators': 1000,
    'n_jobs': -1,
    'random_state': 42,

    # PurgedKFold 参数
    'cv_n_splits': 5,
    'cv_embargo_pct': 0.01,  # 隔离期占总数据集的百分比

    # 训练/划分参数
    'holdout_months': 12,     # 保留最后 N 个月不参与训练（OOS 验证）

    # 特征选择：RF Primary 使用的特征子集（macro-focused）
    # leave empty list = 使用全部特征；指定前缀可控制特征范围
    'feature_prefixes': ['feat_rsi_', 'feat_roc_', 'feat_stoch_', 'feat_adx_', 'feat_vwap_', 'feat_cross_ma_', 'feat_shannon_', 'feat_lz_entropy_', 'feat_hl_vol_'],

    # 样本过滤：Trend Scanning side 阈值（绝对 t_value）
    'min_t_value': 0.0,       # |t_value| >= 此值才纳入训练；0 = 不额外过滤侧样本

    # avgU 抽样参数
    'max_samples_method': 'avgU',  # 'avgU' 动态唯一性或 'float' 固定比例
    't1_col': 'exit_ts',           # 用于计算唯一性的退出时间列
}
```

`PRIMARY_MODEL_TYPE` 可选值扩展为 `'ma' | 'cusum_direction' | 'rf'`。

## 核心组件

### 1. 样本唯一性平均唯一性

#### 1.1 唯一性计算

基于 `afmlkit.sample_weights` 模块的唯一性算法：

1. 每个样本的收益计算窗口 = `[t0, t1]`
   - `t0`: 样本在 `events_features.parquet` 中的时间戳
   - `t1`: 从 `trend_labels.parquet` 获取的退出时间
2. 统计每个时间索引上的并发收益数
3. 每个样本的唯一性 = `1 / max_concurrent_in_window`
4. 样本平均唯一性 = `mean(uniqueness_weights)`

此值（0~1）作为 `BaggingClassifier(max_samples=avgU)`。

#### 1.2 为什么不用固定 0.5

金融标签高度重叠（CUSUM 事件间隔短，趋势窗口重叠多），固定 0.5 会导致：
- 袋内抽取大量重叠样本
- 树之间高度相关，装袋效果退化
- 过拟合风险增加

动态 avgU 强制袋内样本抽样频率 ≤ 真实信息密度。

### 2. RF Primary Model 训练流程

#### 2.1 数据构建

```
X: events_features.parquet → 筛选 macro 特征子集
y: trend_labels.parquet.side → 剔除 side=0
t1: trend_labels.parquet.t1（或 tbm_results.parquet.exit_ts）→ PurgedKFold 用
```

- 对齐三者的索引，取交集
- 剔除 side=0（"无趋势"由低置信度隐式表达）
- 可选：通过 `min_t_value` 进一步过滤低置信度样本（|t_value| 过小）
- 划分 Train / Holdout（最后 12 个月）

#### 2.2 训练架构

```python
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance

# Base Learner: max_features=1 消除遮蔽效应
rf = RandomForestClassifier(
    n_estimators=1,       # Bagging 中用单棵树
    max_features=1,
    min_samples_leaf=len(X_train) // 10,
    random_state=config['random_state']
)

# Bagging: avgU-based max_samples
bagging = BaggingClassifier(
    estimator=rf,
    n_estimators=config['n_estimators'],
    max_samples=avgU,      # 动态唯一性，不是 0.5
    random_state=config['random_state'],
    n_jobs=config['n_jobs']
)

# PurgedKFold + Embargo: 交叉验证
# 从 afmlkit.validation.purged_cv 导入
cv = PurgedKFold(
    n_splits=config['cv_n_splits'],
    embargo_pct=config['cv_embargo_pct']
)

# OOF 预测
oof_probs = cross_val_predict(bagging, X_train, y_train, cv=cv)
oof_pred = (oof_probs >= 0.5).astype(int) * 2 - 1  # prob → side
```

#### 2.3 模型训练步骤详述

1. **加载数据**
   - 读取 `events_features.parquet` → `X_all`
   - 读取 `trend_labels.parquet` → `labels_all`（含 side, t_value, t1）
   - 可选读取 `tbm_results.parquet` → 获取更精确的 `exit_ts`

2. **构建标签**
   - `y = labels_all.side`
   - 剔除 `side == 0` 的样本
   - 可选：剔除 `|t_value| < min_t_value` 的样本
   - 输出 `y_train`, `y_holdout`

3. **筛选特征**
   - 根据 `RF_PRIMARY_CONFIG['feature_prefixes']` 筛选 X
   - 前缀匹配：如 `'feat_ma_'` 匹配 `feat_ma_5`, `feat_ma_20` 等
   - 输出 `X_train_filtered`, `X_holdout_filtered`

4. **计算 avgU**
   - 从 `labels_all.t1`（或 `tbm.exit_ts`）获取每个样本的退出时间
   - 计算并发收益数 → 唯一性权重 → avgU
   - 如 avgU 计算失败或为 0，回退到 0.5

5. **交叉验证（OOF 预测）**
   - 使用 PurgedKFold(n_splits, embargo_pct) 划分
   - 每个 fold：
     - 确定测试集的时间范围 [t_start, t_end] 和 embargo 范围 [t_end, t_end + embargo]
     - 训练集排除测试集时间重叠的样本
     - 在每个 fold 上训练 BaggingClassifier，预测测试集概率
   - 汇总所有 fold 的 OOF 概率 → `oof_probs`

6. **Holdout 预测**
   - 用全部训练集（或 cv 最优超参数）训练最终模型
   - 预测 Holdout 集 → `holdout_probs`

7. **全量训练（保存模型）**
   - 用全部训练集（Train + 可选 holdout）训练最终 RF
   - 保存模型到 `output/models/rf_primary.pkl`

8. **输出信号**
   - 合并 OOF + Holdout 预测为 `rf_primary_signals.parquet`:

     | 列 | 说明 |
     |-----|------|
     | `side` | `sign(prob - 0.5) → {+1, -1}` |
     | `y_prob` | RF 预测为 side=1 的概率 |
     | `y_pred` | RF 预测的整数 side (1 或 0 → 转为 ±1) |
     | `trend_side` | 原始 trend label（用于评估） |
     | `t1` | 事件退出时间 |
     | `is_holdout` | 是否属于 holdout 集 |

### 3. 与 04_ma_primary_model.py 整合

#### 新增 `generate_rf_signals()`

```python
def generate_rf_signals(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    rf_signals_path: str
) -> pd.DataFrame:
    """从 RF 预测信号文件读取 side 并整合信号。"""
    rf_df = pd.read_parquet(rf_signals_path)

    # 对齐事件点
    signals = events.copy()
    common_idx = signals.index.intersection(rf_df.index)
    signals = signals.loc[common_idx].copy()

    # 注入 RF 输出的 side
    signals['side'] = rf_df.loc[common_idx, 'side'].values
    signals['idx'] = [bars.index.get_loc(ts) for ts in signals.index]

    # 统计
    n_long = int((signals['side'] == 1).sum())
    n_short = int((signals['side'] == -1).sum())
    print(f"\n[RF Primary] 信号生成完成")
    print(f"   来源: {rf_signals_path}")
    print(f"   总信号数: {len(signals)}")
    print(f"   多头信号: {n_long} ({n_long/len(signals)*100:.1f}%)")
    print(f"   空头信号: {n_short} ({n_short/len(signals)*100:.1f}%)")

    return signals
```

#### 修改 `main()` 分发逻辑

```python
PRIMARY_MODEL_TYPE = 'rf'  # 配置中改为 'rf'

if primary_model_type == 'rf':
    signals = generate_rf_signals(
        bars, events,
        os.path.join(FEATURES_DIR, 'rf_primary_signals.parquet')
    )
```

#### 保持现有逻辑

- `generate_ma_signals()` — 不变
- `generate_cusum_direction_signals()` — 不变
- TBM 计算 — 不变
- 可视化 — 不变

### 4. 特征差异化策略

RF Primary 和 Meta Model 使用不同的特征子集，避免回音室效应：

| 模型 | 特征范围 | 关注维度 |
|------|---------|---------|
| **RF Primary** | `feat_ma_*`, `feat_momentum_*`, `feat_trend_*`, `feat_cross_ma_*`, `feat_entropy_*` | 趋势方向、动量强度、价格形态 |
| **Meta Model** | `feat_ewm_vol_*`, `feat_theil_*`, `feat_microstructure_*`, `feat_serial_corr_*`, `feat_structural_*`, `feat_open_interest_*`, `feat_time_*` + `rf_prob` | 波动率突变、量能不衡、流动性枯竭、模型置信度 |

通过 `RF_PRIMARY_CONFIG['feature_prefixes']` 控制 RF 可用特征，剩余特征自动留给 Meta Model。元模型像"冷静的旁观者"：在极端波动率或流动性枯竭时，评估 RF 的趋势预测是否可信。

**注意**：RF Primary 的概率输出 `y_prob` 将作为额外特征注入 Meta Model 的训练集，使 Meta Model 能够利用 RF 的置信度信息。这需要在 `07_meta_model.py` 中加载 RF 信号并与特征合并。

### 5. PurgedKFold + Embargo 验证

#### Embargo 机制

`PurgedKFold` 从 `afmlkit.validation.purged_cv` 导入，已内建 embargo 支持：
- `embargo_pct`: 隔离期占总数据集百分比（默认 0.01）
- Purge: 排除与测试集标签时间重叠的训练样本
- Embargo: 排除测试集之后紧邻的训练样本

**为什么需要 Embargo**：特征中的移动平均、波动率估计等具有序列相关性。即使测试集标签的 t1 之后紧挨着的训练样本在标签上不重叠，其使用的特征值仍然携带测试集的信息。Purging 解决标签重叠，Embargo 解决特征溢出。

#### 交叉验证输出

| 输出 | 用途 |
|------|------|
| OOF 准确率 | 评估 Primary Model 预测 side 的能力 |
| OOF 概率 | 用于计算 precision/recall/F1 |
| 特征重要性 (MDI) | 特征分析，指导特征选择 |
| Holdout 指标 | 独立 OOS 验证 |

## 错误处理

- **avgU 计算失败**：回退到 `max_samples=0.5` 固定比例
- **side=0 过滤后样本不足**：降低 `min_t_value` 或关闭过滤，输出警告
- **特征前缀未匹配**：输出警告日志，列出实际匹配的前缀，使用可用特征
- **RF 信号文件缺失**：`04_ma_primary_model.py` 回退到默认模型类型
- **RF Primary 侧分布极度偏态**（如 >90% 为同一侧）：输出警告，可能需要调整特征或标签

## 测试策略

### 单元级

- **avgU 计算**：给定已知的 t0/t1 列表，验证唯一性权重 = 1/n_concurrent
- **特征前缀筛选**：给定特征名列表和前缀列表，验证筛选结果正确
- **side=0 剔除**：验证过滤后 side 仅包含 {+1, -1}

### 集成级

- **端对端流程**：运行 `04_rf_primary_model.py`，验证：
  - `rf_primary_signals.parquet` 文件存在且列正确
  - `rf_primary.pkl` 模型文件存在
  - OOF 概率 ∈ [0, 1]，side ∈ {-1, 1}
  - 信号数量 > 0
- **与现有流程整合**：设置 `PRIMARY_MODEL_TYPE='rf'`，运行 `04_ma_primary_model.py`：
  - TBM 结果与现有格式一致
  - 后续脚本无需修改

### 验证级

- **PurgedKFold + Embargo 工作**：验证 fold 之间无信息泄漏
  - 测试集样本的 t1 之后的训练样本（在 embargo 范围内）被排除
- **Bagging avgU vs 固定 0.5**：对比两种抽样策略的 OOF 指标
  - avgU 应产生更低的过拟合（train-vs-oof 差距更小）

### 验证级

- **特征差异化验证**：RF Primary 使用的特征与 Meta Model 不完全重叠
  - 特征前缀交集为空或部分，不是全集
  - RF 概率作为 Meta Model 的额外特征

## 输出清单

| 文件 | 来源 | 说明 |
|------|------|------|
| `output/features/rf_primary_signals.parquet` | `04_rf_primary_model.py` | RF 预测信号（side, y_prob） |
| `output/models/rf_primary.pkl` | `04_rf_primary_model.py` | 训练好的 RF 模型 |
| `output/figures/04_rf_feature_importance_mdi.png` | `04_rf_primary_model.py` | 特征重要性（MDI） |
| `output/figures/04_rf_confusion_matrix.png` | `04_rf_primary_model.py` | 混淆矩阵 |
| `output/figures/04_rf_signal_distribution.png` | `04_rf_primary_model.py` | 侧分布 |
| `output/features/rf_primary_cv_report.parquet` | `04_rf_primary_model.py` | 交叉验证指标报告 |

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `strategies/AL9999/04_rf_primary_model.py` | **新建** | RF Primary Model 训练+推理+可视化 |
| `strategies/AL9999/04_ma_primary_model.py` | **修改** | 增加 rf 类型支持和 `generate_rf_signals()` |
| `strategies/AL9999/config.py` | **修改** | 增加 `RF_PRIMARY_CONFIG`，扩展 `PRIMARY_MODEL_TYPE` |

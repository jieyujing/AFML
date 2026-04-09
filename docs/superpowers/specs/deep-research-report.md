# 构建事件-候选层面的统一 Meta Model：设计与工程交付清单

## 执行摘要

你当前已有 6 个 primary candidates（`combo_id + primary_side`），并已冻结 TBM 主标签（`pt=1σ / sl=1σ`）与 primary 评分体系（rate-normalized EffectiveRecall）。本报告给出一套**统一 Meta Model**（事件-候选层面）的可执行工程方案：用一张 **event-candidate 长表**训练一个 Meta 分类器，将高召回候选池提纯为**高 precision 决策层**。核心要点是：**样本单位选“事件×候选”**、**标签来自 TBM outcome 与 primary_side 是否一致**、**验证必须做 purging/embargo 防止 label horizon 重叠泄露**，并输出可直接接入下游的 inference 与交付物清单。Purging/embargo 的必要性与实现习惯在金融 ML 的交叉验证工具中已有成熟范式（例如 CPCV：训练集剔除与测试标签区间重叠的样本，并支持 embargo 缓冲）。citeturn9view4turn8view0

---

## 数据资产与 meta_training_table 设计

### 事件-候选层面数据形态

你已经有 primary 层输出字段（至少包含 `event_time、combo_id、primary_side、cusum_rate、fast/slow windows、vertical_bars、Recall/CPR/Lift/EffectiveRecall/Uniqueness/OOS` 等）。Meta 层推荐将数据组织为：

- **一行 = 一个 event_time 上的一个候选（combo）触发**  
- 允许同一 `event_time` 出现多行（多个候选同时触发、可能同向/反向冲突）
- 训练标签 `meta_y` 由 TBM label 生成（见后文）

> 这类长表也被称为 *event-candidate panel*：它允许 meta 同时学习 **候选身份差异** 与 **当下环境条件**，并在推理时天然支持“同一事件内去重/选优”。

### meta_training_table 字段 schema（推荐 v1）

说明约定：  
- **必需**：训练/推理最小闭环必须存在  
- **可选**：建议保留，能显著增强效果/可解释性  
- **派生**：由原始列计算得到（注意：派生也可能依赖未来路径，但仅用于标签/切分或统计，不作为特征时不算泄露）

> 类型参考 pandas / parquet 友好类型。

| 列名 | 类型 | 角色 | 来源 | 说明 |
|---|---|---|---|---|
| event_time | datetime64[ns] | 必需 | Primary | 事件时间戳（bar 结束或事件触发时刻，约定一致） |
| event_id | string | 派生 | event_time | `hash(symbol + event_time)`，用于同一事件聚合/去重 |
| symbol | string | 可选 | 上游 | 多品种扩展；单品种可固定写死 |
| combo_id | string | 必需 | Primary | 例如 `rate=0.15_fast=5_slow=20_vb=10` |
| candidate_id | int16 | 派生 | combo_id | 将 6 个候选映射为 `0..5`（便于 categorical） |
| primary_side | int8 | 必需 | Primary | +1/-1（方向由 Primary 给出） |
| cusum_rate | float32 | 必需 | Primary | 0.05/0.10/0.15 |
| fast_window | int16 | 必需 | Primary | DMA 快线窗口 |
| slow_window | int16 | 必需 | Primary | DMA 慢线窗口 |
| vertical_bars | int16 | 必需 | Primary | TBM 垂直屏障 bars（horizon proxy） |
| tbm_pt_sigma | float32 | 可选 | 常量 | 固定为 1.0（记录版本/审计） |
| tbm_sl_sigma | float32 | 可选 | 常量 | 固定为 1.0（记录版本/审计） |
| label_end_time | datetime64[ns] | 派生 | TBM | 该候选标签的有效期结束（命中 barrier 或 timeout） |
| tbm_outcome | int8 | 派生 | TBM | 取值建议：`+1=hit_pt`、`-1=hit_sl`、`0=timeout` |
| meta_y | int8 | 必需(训练) | 派生 | 二分类目标：primary 是否成功（见后文） |
| is_timeout | int8 | 派生 | tbm_outcome | `1 if timeout else 0`（可用于分析或过滤） |
| shock_size | float32 | 可选 | 派生 | CUSUM 冲击幅度（触发时累计值或 abs return proxy） |
| ret_1 | float32 | 可选 | 派生 | 事件 bar 的收益（logret 或 simple ret） |
| ewm_vol | float32 | 可选 | 派生 | `EWMA` 波动（窗口见特征表） |
| ma_gap_pct | float32 | 可选 | 派生 | `(fast_ma - slow_ma) / price` |
| fast_slope | float32 | 可选 | 派生 | `fast_ma(t) - fast_ma(t-k)` / k |
| slow_slope | float32 | 可选 | 派生 | 同上（慢线） |
| price_to_slow | float32 | 可选 | 派生 | `(price - slow_ma)/price` |
| time_since_prev_event_h | float32 | 可选 | 派生 | 距离“上一次任何候选触发”的时间（小时假设） |
| n_fired | int8 | 可选 | 派生 | 同一 `event_time` 触发的候选数量（ensemble） |
| n_long | int8 | 可选 | 派生 | 同一事件触发中 `primary_side=+1` 数量 |
| n_short | int8 | 可选 | 派生 | 同一事件触发中 `primary_side=-1` 数量 |
| conflict_flag | int8 | 可选 | 派生 | 事件内多空同时出现则 1 |
| agreement_ratio | float32 | 可选 | 派生 | `max(n_long,n_short)/n_fired` |
| is_majority_side | int8 | 可选 | 派生 | 当前行 side 是否为事件内多数侧 |
| fold_id | int8 | 派生 | 切分逻辑 | 交叉验证折号（用于追踪） |
| sample_weight | float32 | 可选 | 派生 | 推荐：按 uniqueness 等权重（见训练部分） |
| data_version | string | 可选 | 常量 | 数据/特征版本号（复现用） |
| label_version | string | 可选 | 常量 | `tbm_pt1_sl1_vbX` 等（审计用） |
| model_cut_time | datetime64[ns] | 可选 | 派生 | 该行只允许使用 `<= cut_time` 的信息（防泄露审计） |

> 关键点：`label_end_time` 是**purging/embargo 的核心字段**。在 event-driven 标签里，很多样本的标签区间会重叠；CPCV/ PurgedKFold 的思想是将训练集中与测试标签区间重叠的样本剔除，并可加上 embargo 缓冲。citeturn9view4

### 示例 5 行样本（包含同一事件多候选同时触发）

> 示例仅演示形态与字段，不代表真实数值。假设小时级 bars（未指定频率时的默认假设）。

| event_time | event_id | combo_id | candidate_id | primary_side | cusum_rate | fast_window | slow_window | vertical_bars | tbm_outcome | meta_y | shock_size | ma_gap_pct | ewm_vol | n_fired | n_long | n_short | conflict_flag | agreement_ratio | is_majority_side |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-01-15 10:00 | EVT_a1 | rate=0.15_fast=5_slow=20_vb=10 | 0 | +1 | 0.15 | 5 | 20 | 10 | +1 | 1 | 0.018 | 0.0062 | 0.012 | 3 | 2 | 1 | 1 | 0.667 | 1 |
| 2026-01-15 10:00 | EVT_a1 | rate=0.15_fast=8_slow=20_vb=10 | 1 | +1 | 0.15 | 8 | 20 | 10 | +1 | 1 | 0.018 | 0.0049 | 0.012 | 3 | 2 | 1 | 1 | 0.667 | 1 |
| 2026-01-15 10:00 | EVT_a1 | rate=0.05_fast=8_slow=30_vb=10 | 5 | -1 | 0.05 | 8 | 30 | 10 | +1 | 0 | 0.018 | -0.0011 | 0.012 | 3 | 2 | 1 | 1 | 0.667 | 0 |
| 2026-01-18 04:00 | EVT_b7 | rate=0.10_fast=5_slow=60_vb=30 | 4 | -1 | 0.10 | 5 | 60 | 30 | -1 | 1 | 0.011 | -0.0054 | 0.009 | 1 | 0 | 1 | 0 | 1.000 | 1 |
| 2026-02-02 16:00 | EVT_c3 | rate=0.15_fast=5_slow=60_vb=10 | 2 | +1 | 0.15 | 5 | 60 | 10 | 0 | 0 | 0.014 | 0.0020 | 0.015 | 2 | 2 | 0 | 0 | 1.000 | 1 |

---

## 特征工程清单

### 归一化与编码原则

- **树模型（LightGBM）**：多数数值特征无需标准化；但对 heavy-tail 特征（如 `shock_size`、`ret_1`）建议使用 **rank percentile / winsorize**，改善鲁棒性与跨品种可迁移性。
- **线性模型（Logistic baseline）**：建议对连续特征做 `StandardScaler` 或 `RobustScaler`；对极端值仍建议 winsorize。
- **分类特征**：候选身份（`candidate_id/combo_id`）可作为 categorical；LightGBM 支持整数编码的 categorical 特征并提供专门分裂策略与正则参数。citeturn9view0turn3view3

下面按你指定的五大类给出特征清单（每类列出**推荐 v1**与**可选增强**）。

### Candidate identity 特征

| 特征名 | 公式/构造 | 窗口 | 归一化 | 类型建议 | 备注 |
|---|---|---|---|---|---|
| candidate_id | 6 候选映射 0..5 | 无 | 否 | categorical / embedding | v1 直接 categorical 即可 |
| cusum_rate | 原值 | 无 | 否 | numeric | 0.05/0.10/0.15 |
| fast_window / slow_window / vertical_bars | 原值 | 无 | 否 | numeric 或 categorical | 也可做组合编码 |
| combo_id_hash | `hash(combo_id) mod M` | 无 | 否 | categorical | 防超长字符串 |
| candidate_prior_precision | 训练集内：该候选历史 `meta_y` 均值 | expanding | 平滑(+β) | numeric | **必须按 fold 计算**防泄露 |
| candidate_prior_oos | 训练集内 OOS 指标（若可得） | expanding | 可选 | numeric | 同上：fold 内统计 |

> LightGBM categorical 的编码要求与处理细节见官方文档（需非负整数编码，并可用 `cat_smooth/min_data_per_group` 控制过拟合）。citeturn9view0turn3view3

### Event-level 特征

| 特征名 | 公式/构造 | 推荐窗口（bars） | 归一化建议 | 备注 |
|---|---|---:|---|---|
| shock_size | CUSUM 累计值/触发阈值距离/abs ret proxy（三选一统一口径） | 即时 | rank_pct / winsorize | Heavy-tail，鲁棒化优先 |
| ret_1 | 事件 bar 的收益（logret） | 即时 | winsorize | 可加 `ret_3/ret_5` |
| range_1 | `(high-low)/close` | 即时 | rank_pct | 若有 OHLC |
| time_since_prev_event_h | 与上一事件间隔（小时） | 即时 | log1p 后 z | 稀疏候选通常更独立 |
| vol_jump | `ewm_vol(t)/ewm_vol(t-k)` | k=10 | clip | 衡量突发波动 |

### Trend-state 特征（围绕 DMA/趋势状态）

| 特征名 | 公式/构造 | 推荐窗口 | 归一化建议 | 备注 |
|---|---|---:|---|---|
| ma_gap_pct | `(fast_ma - slow_ma)/price` | 即时 | z / rank_pct | 趋势强度 proxy |
| fast_slope | `(fast_ma(t) - fast_ma(t-5))/5` | 5 | z | |
| slow_slope | `(slow_ma(t) - slow_ma(t-20))/20` | 20 | z | |
| price_to_slow | `(price - slow_ma)/price` | 即时 | z / rank_pct | |
| cross_age | 距离最近一次金叉/死叉 bars | N=200 回看 | log1p | |
| trend_persistence | 最近 N bars 中 `sign(fast-slow)` 一致比例 | N=50 | 无 | 稳态趋势 vs 噪声段 |

### Regime 特征（波动/时段/结构状态）

| 特征名 | 公式/构造 | 推荐窗口 | 归一化建议 | 备注 |
|---|---|---:|---|---|
| ewm_vol | EWMA 波动 | span=50/100 | rank_pct | 与 TBM σ 一致口径更佳 |
| vol_percentile | `ewm_vol` 在过去 W 中分位数 | W=500 | 无 | 解释性强 |
| session_bucket | 小时/时段分箱 | 无 | 无 | categorical |
| dow | 星期几 | 无 | 无 | categorical |
| term_structure_state | contango/backwardation 分箱（如可得） | 1d/1w | categorical | 若做期货强烈建议 |

### Ensemble 特征（6 候选共振/冲突）

| 特征名 | 公式/构造 | 归一化 | 备注 |
|---|---|---|---|
| n_fired | 事件内触发候选数 | 否 | 最重要的“共振强度”特征之一 |
| n_long / n_short | 事件内各方向数量 | 否 | |
| conflict_flag | `1{n_long>0 and n_short>0}` | 否 | 冲突通常降低成功率 |
| agreement_ratio | `max(n_long,n_short)/n_fired` | 否 | 一致性强往往更可靠 |
| is_majority_side | 当前行 side 是否多数 | 否 | |
| consensus_side | `sign(n_long - n_short)`（平局置 0） | categorical | 用于推理阶段聚合 |

---

## 标签定义与处理规则

### meta_y 定义（与你的 pipeline 完全对齐）

你已冻结 TBM 主标签：`pt=1σ/sl=1σ`。对每条 event-candidate 样本（已知 `primary_side`）：

- `tbm_outcome = +1` 表示先 hit profit barrier（方向“正确”）
- `tbm_outcome = -1` 表示先 hit stop barrier（方向“错误”）
- `tbm_outcome = 0` 表示 vertical barrier 到期（timeout）

推荐 v1 二分类标签：

\[
meta\_y = \mathbb{1}\big( tbm\_outcome = primary\_side \big)
\]

即：primary 给的方向是否最终赢得 TBM 判定。

### timeout 处理

v1 推荐：**timeout 视为失败（meta_y=0）**，理由是样本量更稳定，且 meta 的任务是“筛选值得交易的候选”；timeout 通常意味着路径不够强。  
同时建议保留字段 `is_timeout` 以便做敏感性分析（例如：只在训练/评估时比较 “含 timeout” vs “删 timeout”）。

可选 v2：把 timeout 拆成三分类（win/lose/timeout），但工程复杂度上升，且你主要目标是 precision 提升，二分类更直接。

### ambiguous（同一根 bar 内同时触达上下 barrier）

如果你用的是 OHLC 并且 barrier 可能在同一根 bar 内同时被触达，路径顺序不可知，会造成标签歧义。建议规则：

- **默认丢弃 ambiguous 行**（训练更干净）
- 或定义一致的 tie-break（例如先 hit stop 更保守），但要在 `label_version` 中记录

### base_rate 与 lift（若需要）

Meta 的核心目标是提高 precision；你可以在报告中定义：

- `base_rate = mean(meta_y)`（训练集/验证集分别统计）
- `precision_lift = precision_meta / precision_primary`

如果你仍想保留 “lift 相对于朴素基线” 的解释，可用多数类基线：

\[
base\_rate^{maj}=\max(P(meta\_y=1), P(meta\_y=0))
\]

但更常用的业务解释是 **相对 Primary 的 precision uplift**，对交易更直观。

---

## 训练与验证流程

### 为什么必须做 purging/embargo

你的标签由 TBM 定义，天然依赖未来路径（直到 `label_end_time`），因此样本的“信息区间”存在重叠。金融 ML 常用做法是：**训练集剔除与测试标签区间重叠的样本（purging），并可在测试段之后再剔除一段缓冲（embargo）**，以降低信息泄露风险。CPCV 的实现描述中明确强调：训练集需要从与测试标签区间重叠的观测中被 purged，并支持 `pct_embargo` 参数。citeturn9view4

若你暂不实现完整 purged k-fold，至少也应使用时间序列切分并设置 gap：`TimeSeriesSplit` 明确指出其用于时间有序数据，并提供 `gap` 参数排除训练末尾靠近测试段的样本。citeturn8view0

### 推荐的验证策略（v1 可落地）

**策略 A（推荐）**：Walk-forward expanding + purging/embargo（事件级）  
- 按 `event_time` 排序  
- 每次用过去所有数据训练（expanding），用后续一段时间做验证  
- 在每个验证段：
  - 训练集 purging：剔除 `label_end_time` 与验证段区间有重叠的训练样本  
  - embargo：验证段结束后追加缓冲（建议见下）

**策略 B（替代）**：TimeSeriesSplit + gap（快速 baseline）  
- 用 `TimeSeriesSplit(n_splits=K, gap=embargo_n)`  
- 不如 A 严谨，但实现快，适合最先跑通

### embargo 窗口长度建议（与你的 vertical_bars 对齐）

你有 `vertical_bars`（例如 10/20/30）。建议：

- **embargo_bars = max(vertical_bars)**  
- 若 bar 的时间单位不是固定小时（未指定），用 `label_end_time` 的实际时间差来计算 embargo 时长更稳

更严格的建议（当你能获得实际 `label_end_time`）：  
- embargo_time = `max(label_end_time - event_time)` 的分位数（如 95%）  
- 避免个别极端持仓时间导致 embargo 过长

### 样本不平衡与权重

- **类不平衡**：用 `class_weight='balanced'`（Logistic），或 LightGBM 用 `scale_pos_weight = n_neg/n_pos`  
- **样本重叠**：可用 `sample_weight`  
  - v1 简化：同一 event_time 触发的 `n_fired` 行，每行权重乘以 `1/n_fired`  
  - v2 增强：叠加 uniqueness（你已有该字段），例如 `sample_weight = uniqueness / n_fired`

### 评价指标（建议同时做“行级”与“事件级”）

你最终交易决策是“同一事件内去重后做/不做”，因此评估必须包含：

- **行级（event-candidate）**：ROC-AUC、PR-AUC、Brier（校准）  
- **事件级（dedupe 后）**：precision、filtered recall、coverage、precision lift vs primary

PR 曲线在类不平衡场景下尤其有用，且精确定义与解读在 scikit-learn 文档中有清晰说明。citeturn3view5  
Brier score 可作为概率预测的严格 proper scoring rule（用于校准质量检验）。citeturn8view2

### OOS 报告模板（建议输出到 xlsx）

| Fold | Train 起止 | Val 起止 | Val 样本数(行) | Val 事件数 | 正类率 | Primary Precision | Meta Precision | Precision Lift | Filtered Recall | Coverage | PR-AUC | ROC-AUC | Brier | 备注 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 2024-01-01~2024-12-31 | 2025-01-01~2025-03-31 |  |  |  | 0.57 |  |  |  |  |  |  |  | |
| 1 | … | … |  |  |  | 0.57 |  |  |  |  |  |  |  | |
| … | … | … |  |  |  |  |  |  |  |  |  |  |  | |

---

## 模型、校准与可解释性

### 基线与主力模型选择

- **Baseline：Logistic Regression**  
  - 用于验证“是否线性可分”“候选身份/共振特征是否立刻带来 uplift”
- **主力：LightGBM（二分类）**  
  - 适合 tabular、非线性与交互项强的事件特征
  - 支持 categorical 特征（整数编码）与相应正则控制项citeturn9view0turn3view3

### Logistic baseline 建议

- 特征：先用 v1 最小集合（identity + event + ensemble）
- 预处理：winsorize（可选）+ StandardScaler（连续特征）
- 正则：L2（默认）  
- 类权重：`class_weight='balanced'`

### LightGBM 主力模型建议（起步超参）

- `objective='binary'`
- `learning_rate=0.03~0.07`
- `num_leaves=31~127`
- `min_data_in_leaf=100~500`（视样本量）
- `feature_fraction=0.7~0.95`
- `bagging_fraction=0.7~0.95`
- `bagging_freq=1`
- `lambda_l1/lambda_l2`：适度开启
- categorical：`categorical_feature=[candidate_id, session_bucket, dow, ...]`

Early stopping：建议使用验证集 early stopping（需要至少一个 valid set 与 metric），并记录 `best_iteration`。LightGBM 的 early stopping 回调与参数行为在官方文档中有明确说明。citeturn9view1turn9view5

### 概率校准（非常建议 v1 就做）

Meta 输出会直接用于阈值与仓位，因此要关注**概率是否可用**。scikit-learn 推荐用 `CalibratedClassifierCV` 进行 sigmoid（Platt）或 isotonic 校准；并给出 sigmoid 基于 Platt logistic model 的形式与 isotonic 的约束最小二乘目标。citeturn8view1turn3view2

实务建议：

- 样本量较小或正类稀缺时：优先 `method='sigmoid'`  
- 样本量足够（经验上>~1000 calibration 样本）时可试 `isotonic`，但其更易在小样本上过拟合。citeturn8view1turn3view2

### 可解释性与归因输出

- **feature importance**：LightGBM gain/ split importance 只作快速 sanity check
- **SHAP**：用于解释单笔决策与输出 reason_tags；SHAP 被提出为统一的 additive 解释框架，可为每次预测分配特征贡献值。citeturn10view0

> 工程落地建议：reason_tags 不直接输出“数值”，而输出 Top-k 贡献特征的“标签化解释”（例如 `high_agreement`, `low_vol_regime`, `strong_ma_gap`）。

### 训练伪代码（可直接落地改造）

```python
# 伪代码：构建 event-candidate 长表 + walk-forward + 校准

df = load_meta_training_table("meta_training_table.parquet")
df = df.sort_values("event_time")

# 1) 定义信息区间（用于 purging/embargo）
info_end = df["label_end_time"]  # 每行标签区间终点

# 2) walk-forward folds（按时间）
folds = build_walk_forward_folds(df["event_time"], n_folds=6, val_horizon="90D")

reports = []
for fold_id, (train_idx, val_idx) in enumerate(folds):
    train = df.iloc[train_idx].copy()
    val = df.iloc[val_idx].copy()

    # 3) purging：剔除训练集中与验证标签区间重叠的样本
    train = purge_by_label_overlap(train, val_start=val["event_time"].min(),
                                   val_end=val["label_end_time"].max())

    # 4) embargo：在 val_end 之后追加缓冲
    train = embargo_after(train, embargo_end=val["label_end_time"].max() + embargo_delta)

    X_tr, y_tr = train[FEATURES], train["meta_y"]
    X_va, y_va = val[FEATURES], val["meta_y"]

    # 5) fit LGBM
    model = fit_lgbm(X_tr, y_tr, X_va, y_va,
                     categorical_features=CATS,
                     scale_pos_weight=calc_spw(y_tr),
                     early_stopping_rounds=200)

    # 6) 概率校准（用 training 内再切一段 calibration 或用 CV 方式做）
    cal = CalibratedClassifierCV(model, method="sigmoid", cv=3)
    cal.fit(X_tr, y_tr)

    # 7) 评估：行级 + 事件级（去重后）
    p_va = cal.predict_proba(X_va)[:, 1]
    rep = evaluate_meta(val, p_va, dedupe_mode="per_event_max_p")
    reports.append(rep)

save_oos_report(reports, "meta_oos_report.xlsx")
save_model(cal, "meta_model_v1.pkl")
```

---

## 推理与决策规则

### 同一 event 内如何去重

你有 6 个候选，事件内可能多行。v1 推荐两种规则（都要在评估里对齐）：

- **模式一：每事件只取 max p（全局最优）**  
  - 适合你最终一笔只想做一单
- **模式二：每事件按方向各取 top1（per-direction top1）**  
  - 适合允许同一事件同时持有多空不同策略（通常不建议，除非能净敞口控制）

默认建议：**模式一**（最清晰，减少重复与过拟合风险）。

### 阈值选择方法

阈值不是拍脑袋，推荐按目标函数选：

- **目标 precision**：如希望从 primary `~0.57` 提升到 `0.65`，在验证集上选使 precision 达标的最小阈值  
- **PR 曲线选 operating point**：PR 曲线展示 precision 与 recall 的权衡，且在类不平衡时更适用。citeturn3view5  
- **校准后再选阈值**：先保证概率可信，再谈阈值稳定；校准方法与注意点见 scikit-learn calibration 文档。citeturn8view1turn3view2  

### 仓位映射（p → size）

你未给出仓位约束，本报告给出可落地的三种简单映射（任选其一作为 v1）：

- **线性截断**：`size = clip((p - p0) / (1 - p0), 0, 1)`，`p0` 为交易阈值  
- **分段**：`p<0.60 -> 0`，`0.60~0.70 -> 0.5`，`>0.70 -> 1.0`  
- **Kelly-like（保守版）**：若你能估计 payoff（win/loss 比），可用 `f = (p*b - (1-p))/b` 再截断；否则 v1 不启用

### 下游输出 schema（inference 输出）

建议输出一张 “已去重后的决策表”：

| 字段 | 类型 | 说明 |
|---|---|---|
| event_time | datetime64[ns] | 事件时间 |
| event_id | string | 事件 ID |
| chosen_combo_id | string | 被选中的候选 |
| side | int8 | +1/-1 |
| p_meta | float32 | meta 概率（校准后） |
| threshold | float32 | 本次使用阈值 |
| decision | string | `TAKE/SKIP` |
| reason_tags | string[] | 解释标签（来自 SHAP/规则） |
| model_version | string | 模型版本与训练截止时间 |
| features_snapshot | string/json | 可选：用于审计复盘 |

---

## 实验计划、工程交付物与风险

### 实验计划与对照组（至少四组）

| 实验组 | 说明 | 切分策略 | 去重规则 | 主要指标 | 目标（示例） |
|---|---|---|---|---|---|
| Baseline（no meta） | 直接用 primary 候选交易 | 仅用于评估 | per_event 去重（按 primary score 或任一 rule） | precision / coverage | precision ≈ 0.57（基线） |
| Meta-Logistic | 统一 meta + logistic | walk-forward + purging/embargo | per_event_max_p | precision lift / filtered recall / Brier | precision 0.57 → 0.62 |
| Meta-LGBM | 统一 meta + LGBM | 同上 | per_event_max_p | precision / PR-AUC / calibration | precision 0.57 → 0.65 |
| Meta-LGBM + embeddings | 对 `candidate_id/session` 引入 embedding（NN 预训练或 target-encoding fold-safe）后再喂 LGBM | 同上（encoding 必须 fold 内） | per_event_max_p | precision / stability / drift | OOS 更稳、阈值更稳定 |

> 说明：embedding 组的关键不在“更强模型”，而在“更好地表达类别结构”；如果你只有 6 个 candidate，LightGBM categorical 往往已经足够，embedding 主要为未来候选扩展做准备。citeturn9view0

### 工程交付物清单（文件名与格式）

沿用你当前目录风格，建议新增：

```text
strategies/AL9999/output/meta_model/
├── meta_training_table.parquet        # 事件-候选长表（训练资产）
├── meta_features_spec.csv             # 特征清单：类别/公式/窗口/归一化/类型
├── meta_splits.json                   # fold 切分与 embargo/purge 配置（可复现）
├── meta_model_logistic_v1.pkl         # 基线模型
├── meta_model_lgbm_v1.txt             # LightGBM booster（或 .pkl）
├── meta_calibrator_v1.pkl             # 概率校准器（如 sigmoid）
├── meta_oos_report.xlsx               # OOS 报告（含 per-fold 表 + 图表）
├── meta_inference_output.parquet      # 推理结果（事件级去重后）
├── inference_api_spec.md              # 推理输入/输出协议（字段、版本、阈值）
├── meta_feature_importance.csv        # gain/permutation/SHAP summary
└── README_meta_model.md               # 运行方式、版本、注意事项
```

### 风险与注意事项

**泄露风险**：最大风险来自 label horizon 重叠（vertical barriers 导致样本标签区间重叠），以及在特征里混入 `label_end_time` 之后的信息。应通过 purging/embargo 严格切分，并保持“特征只取 event_time 及之前可见信息”；CPCV/Purged 思路强调训练集需从与测试标签区间重叠的样本中剔除，并可用 embargo 缓冲。citeturn9view4  

**label shift**：TBM 参数冻结后，若市场波动结构变化（σ 的统计口径漂移），`meta_y` 的正类率会漂移。建议在 OOS 报告里持续监控 `base_rate = mean(meta_y)` 与 Brier（校准），Brier 作为 proper scoring rule 可帮助评估概率质量。citeturn8view2  

**candidate overlap**：同一 event 内多候选高度相关，会造成训练样本“表面扩大、有效样本不足”。应使用 event-level 去重评估、并可用 `1/n_fired` 权重降低重复计数。  

**样本量不足**：若有效 event-candidate 行数偏少，复杂模型（尤其 isotonic 校准）容易过拟合；scikit 文档指出 isotonic 在小数据上更易过拟合，且通常需要足够样本（经验上 >~1000）更稳。citeturn8view1turn3view2  

**class imbalance**：正类稀缺时 precision 易被少量 FP 拉垮；需要用 PR 曲线选择阈值并重点关注 precision/recall tradeoff。citeturn3view5  

---

## 下一步行动清单（按优先级）

1. **生成并固化 `meta_training_table.parquet`**：确认同一事件多候选的展开逻辑、`label_end_time/tbm_outcome/meta_y` 的一致性，并做基本数据诊断（正类率、timeout 占比、n_fired 分布）。  
2. **跑通 v1 baseline（Meta-Logistic）**：使用 walk-forward + 简化 purging/embargo（先用 TimeSeriesSplit+gap 也可），输出首版 `meta_oos_report.xlsx`，看 precision uplift 是否显著。citeturn8view0turn3view5  
3. **上线主力 v1（Meta-LGBM + sigmoid 校准）**：加入 categorical 的 candidate_id、ensemble 特征、early-stopping 与校准，固定 `model_version` 产出可复现 artefacts。citeturn9view1turn8view1turn9view0  
4. **建立事件级 inference pipeline**：实现同一 event 去重（默认 per_event_max_p）、阈值选择（目标 precision 或 PR 曲线），并输出 `meta_inference_output.parquet` 与 `inference_api_spec.md`。citeturn3view5  
5. **做可解释性与监控**：产出 SHAP-based reason_tags（Top features），并把 `base_rate、Brier、precision lift、coverage` 纳入持续监控，准备应对 label shift 与市场 regime 变化。citeturn10view0turn8view2
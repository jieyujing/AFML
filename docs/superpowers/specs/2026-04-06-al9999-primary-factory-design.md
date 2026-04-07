# AL9999 Primary Model Factory 设计

**日期**: 2026-04-06 / Updated 2026-04-07
**状态**: 设计完成，已实现并上线

---

## 1. 背景与目标

现有 Primary Model（DMA / MA / CUSUM Direction / RF）各自独立，由配置文件控制参数，没有参数搜索机制。

新目标：**Primary Model Factory** 是一个参数搜索工厂，批量枚举参数组合，按 Recall-First 目标排序，只保留少量候选交给 Meta Model。

---

## 2. 实验矩阵（第一轮）

### 2.1 参数范围

| 参数类 | 变量 | 实验值 | 说明 |
|--------|------|--------|------|
| CUSUM | 目标采样率 | 5%, 10%, 15% | 按率校准 k 值 |
| 双均线 | fast | 5, 8, 10, 12, 15 | |
| 双均线 | slow | 20, 30, 40, 50, 60 | fast < slow |
| TBM | pt / sl | 固定 1.0σ | 第一轮不动 |
| TBM | vertical | 10, 20, 30 bars | 唯一敏感性变量 |

### 2.2 有效组合数

- 有效 (fast, slow) pairs：5 × 5 - fast >= slow = 20
- CUSUM rate × DMA pairs × vertical = 3 × 20 × 3 = **180 → 过滤后 90 组有效**

---

## 3. 评分指标体系

### 3.1 核心指标（轻量计算，全量 90 组）

| 指标 | 公式 | 权重 | 说明 |
|------|------|------|------|
| **Recall** | TP / (TP + FN) | 0.45 | 主目标：抓住多少正样本事件 |
| **Lift** | CPR / base_rate | 0.20 | 相对随机基线的信息增益 |
| **CPR** | positive / all candidates | 0.15 | 候选池正样本比例 |
| **Coverage** | candidates / all_events | — | 采样密度（辅助参考） |

**Coverage 不直接进入 Score**，用于辅助分析。

**Lift 定义**：
```
base_rate = max(n_positive, n_negative) / len(trend_events)  # 始终预测多数类的准确率
Lift = CPR / base_rate
# Lift > 1：比随机好；= 1：与随机相同；< 1：比随机更差
```

### 3.2 深评指标（Top-20 候选）

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **Uniqueness** | 样本独立性 | 平均 uniqueness / overlap ratio |
| **Turnover** | 换手负担 | 年候选次数 × 事件并发度 |
| **avg_inter_event_time** | 平均事件间隔 | 辅助判断换手合理性 |
| **Regime Stability** | 跨 regime 稳健性 | 高/低波动期 recall 方差 |
| **OOS Recall** | 样本外 recall | 70/30 时间切分 |

### 3.3 综合评分

```
EffectiveRecall = Recall × Lift

# Rate-normalized z-score：在每个 CUSUM rate 组内计算
EffectiveRecall_rate_z = rank percentile z within each CUSUM rate
z = 1.0 - (rank - 1) / (n_rate - 1)   # 1.0 = best, 0.0 = worst

Score = 0.45·EffectiveRecall_rate_z - 0.10·Turnover_z + 0.10·Uniqueness_z
```

**Rate-normalized 的含义**：
- 每个 CUSUM rate 内部单独排名，比较"各自制度下谁做得更好"
- 缓解高采样率（15%）的先天优势
- Turnover 和 Uniqueness 使用全局 z-score

**z-Score 计算（Rank Percentile）**：
```
# Turnover（全局，越低越好）
 ranks ascending = lower turnover → higher z
z = 1.0 - (rank - 1) / (n - 1)

# Uniqueness（全局，越高越好）
 ranks ascending = higher uniqueness → higher z
z = 1.0 - (rank - 1) / (n - 1)
```

---

## 4. 流程

```
Step 0: CUSUM 校准
  输入: 历史 bars
  输出: rate → k lookup table {5%: k1, 10%: k2, 15%: k3}

Step 1a: 参数网格生成
  CUSUM rate × (fast,slow) × vertical
  → 90 组有效组合

Step 1b: 轻量扫描（全量 90 组）
  计算: Recall, CPR, Coverage, Lift
  → 毫秒级/组合，快速排序，取 Top-20

Step 2: 深评阶段（Top-20 候选）
  计算: Uniqueness, Turnover, Regime Stability, OOS Recall, avg_inter_event_time
  → 加权综合评分 Score

Step 3: 综合评分 + 分层约束
  分层约束：每个 CUSUM rate 至少 1 个 Top-K 候选
  → 补齐缺失 rate，重新评分

Step 4: 输出
  Top-5+ candidates → 交给 Meta Model
```

---

## 5. 关键设计决策

### 5.1 CUSUM 采样率校准

**二分搜索校准**：
- 对每个目标采样率 (5%, 10%, 15%)，在 k ∈ [k_min, k_max] 上用二分搜索
- 收敛条件：|actual_rate - target_rate| < 1e-4
- 产出可复用的 k lookup table

### 5.2 异常处理

| 情况 | 处理方式 |
|------|----------|
| CPR = 0（无正样本） | Lift = 0，跳过该组合 |
| 所有指标 std=0 | 该组合排末尾 |
| OOS 样本太少（<10） | 标记 oos_unreliable=True，降低权重 |
| Uniqueness = 0 | 标记 low_info=True |

### 5.3 分层约束

Top-K 必须覆盖所有 CUSUM rate（3%, 10%, 15% 各至少 1 个）。若初始排序未覆盖，则：
1. 从缺失 rate 中取 Best combo
2. 补充计算深评指标
3. 重新计算综合评分
4. 输出最终 Top-K

---

## 6. 输出结构

```
output/primary_search/
├── cusum_calibration.parquet   # rate → k 映射表
├── scoring_lightweight.csv     # 90 组轻量指标
├── scoring_deep.csv            # Top-20+ 深评指标
├── scoring_final.csv           # 综合评分排名（全部 90 组）
└── top_candidates.parquet     # Top-5+ 交给 Meta Model
```

### 6.1 Top Candidates 输出字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `combo_id` | str | 唯一标识：rate=X.XX_fast=X_slow=X_vb=X |
| `cusum_rate` | float | CUSUM 采样率 |
| `fast_window` | int | 快线窗口 |
| `slow_window` | int | 慢线窗口 |
| `vertical_bars` | int | 垂直屏障 bar 数 |
| `score` | float | 综合评分 |
| `recall` | float | 主指标 |
| `lift` | float | 相对随机基线 |
| `cpr` | float | 候选正率 |
| `uniqueness` | float | 样本独立性 |
| `turnover` | float | 换手负担 |
| `rank` | int | 综合排名 |

---

## 7. 文件结构

```
strategies/AL9999/
├── config.py                       # PRIMARY_FACTORY_CONFIG
└── primary_factory/
    ├── __init__.py
    ├── cusum_calibrator.py         # Step 0: k 值校准
    ├── param_grid.py               # Step 1a: 参数网格
    ├── lightweight_scorer.py       # Step 1b: 轻量指标
    ├── deep_scorer.py              # Step 2: 深评指标
    ├── scorer.py                   # Step 3: 综合评分
    └── runner.py                   # Main orchestrator
```

**测试**：19 个测试全部通过。

---

## 8. 已知限制与第一轮不解决的问题

1. **TBM pt/sl 固定为 1.0σ**：第一轮不展开，防止标签空间爆炸
2. **只扫描 DMA**：MA、CUSUM Direction、RF 作为后续扩展
3. **不考虑交易成本**：Primary Factory 是信号评分，不做完整回测
4. **OOS 只做简单时间切分**：不做 walk-forward

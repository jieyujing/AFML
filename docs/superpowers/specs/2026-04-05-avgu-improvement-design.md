# avgU 改进设计文档 — CUSUM×4 + t_val≥3.0

## 问题

当前 RF 主模型的 avg_uniqueness = 0.209，意味着每个样本只有 ~21% 的时间窗口是独立的。这导致：
- 有效训练信息量 ≈ 2641 × 0.21 ≈ 554 个独立样本
- Log-Loss 接近随机水平（0.70 vs 0.693），模型输出概率挤压在 0.5 附近
- Short Recall 仅 0.39，错过大量真实空头趋势

## 根因

CUSUM 事件过于密集（2641 个事件 / ~5 年数据），Trend Scanning 为每个事件定义的趋势窗口大量重叠，导致样本标签高度相关。

## 方案

从源头提高 avgU，双管齐下：

| 杠杆 | 改动 | 预期效果 |
|------|------|----------|
| CUSUM 阈值 | `CUSUM_MULTIPLIER = 3 → 4` | 事件 ~2641 → ~1300-1400 |
| t_value 过滤 | `min_t_value = 0.0 → 3.0` | 只保留统计检验显著的强趋势 |
| Trend Windows | `TREND_WINDOWS = [5,10,20,30,50] → [5,10,15]` | 最大窗口 15 bars ≈ 1 交易日 |

预估 avgU 目标：0.45-0.55（翻倍+）

## 架构

### 1. `config.py` 变更

```python
CUSUM_MULTIPLIER = 4         # 从 3 提高到 4，降低事件触发频率

RF_PRIMARY_CONFIG = {
    ...
    'min_t_value': 3.0,      # 从 0.0 提高到 3.0，过滤弱趋势
    ...
}
```

### 2. `02_feature_engineering.py` — 无需改动

脚本从 `config.py` 读取参数，自动生效。

### 3. 重新跑流水线

```bash
uv run python strategies/AL9999/02_feature_engineering.py
uv run python strategies/AL9999/03_trend_scanning.py
uv run python strategies/AL9999/04_rf_primary_model.py
```

### 4. 对比验证

对比旧/新指标表：avgU、Log-Loss、Recall (Long/Short)、Macro F1、Holdout Accuracy

## 回退策略

- 如果 avgU 提升到目标但 RF 性能下降（样本太少、训练不足），可回退到 `CUSUM×5 + t_val≥2.0`
- 原始输出文件会被覆盖，但可通过 git checkout config + 重新运行回退
- 建议备份旧输出：`cp -r strategies/AL9999/output strategies/AL9999/output_backup_mult3`

## 风险与权衡

| 风险 | 缓解 |
|------|------|
| RF 训练样本太少（~1000-1200 vs ~2200） | max_samples 会跟随 avgU 提升到 0.45+，每棵树看到的独立样本更多 |
| Holdout 样本更少 | 12 个月 Holdout 可能只剩 ~150-200 个样本，统计显著性下降 |
| 2025-2026 市场变化 | 如果事件进一步集中在某些时段，可能仍有时序偏移 |

## 后续优化方向

avgU 提升后，若 RF 仍有改善空间：
1. 进一步调整 `CUSUM_MULTIPLIER` 到 4.5 或 5
2. 调整 `t1` 窗口定义（目前由 Trend Scanning 决定）
3. 考虑使用 `min_t_value=5.0` 只保留极强趋势信号

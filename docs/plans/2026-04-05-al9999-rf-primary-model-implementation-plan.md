# AL9999 RF Primary Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 `AL9999` 新增基于 Random Forest 的 Primary Model，并将其无缝接入现有 `TBM -> Meta Model -> Combined Backtest` 工作流。

**Architecture:** 保持现有 Phase 2/3/4/5/6 产物契约不变，只新增一个独立训练脚本 `04_rf_primary_model.py` 生成 `rf_primary_signals.parquet` 与模型文件，然后在 `04_ma_primary_model.py` 中增加 `rf` 分支消费该信号。`07_meta_model.py` 仅增加一个额外特征 `rf_prob`，其余 Meta Label / Backtest 逻辑不改，确保改动边界清晰、易回退。

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, joblib, matplotlib, pytest, parquet, `afmlkit.validation.PurgedKFold`, `afmlkit.label.weights.average_uniqueness`

---

## 执行前约束（先做）

- 在独立 worktree 执行，避免污染当前工作区。
- 安装依赖：`pip install -e .[dev]`
- 测试入口统一使用：`NUMBA_DISABLE_JIT=1 pytest ...`
- 本计划关联技能：`@test-driven-development` `@systematic-debugging` `@verification-before-completion`
- 只做本设计要求的最小变更，不顺手重构已有 `AL9999` 研究脚本。

### Task 1: 锁定 RF Primary 配置契约

**Files:**
- Modify: `strategies/AL9999/config.py`
- Test: `tests/strategies/test_al9999_rf_primary_config.py`

**Step 1: Write the failing test**

```python
from strategies.AL9999 import config


def test_primary_model_type_supports_rf():
    allowed = {"ma", "cusum_direction", "rf"}
    assert config.PRIMARY_MODEL_TYPE in allowed


def test_rf_primary_config_contains_expected_keys():
    cfg = config.RF_PRIMARY_CONFIG
    assert cfg["n_estimators"] == 1000
    assert cfg["cv_n_splits"] == 5
    assert cfg["cv_embargo_pct"] == 0.01
    assert cfg["holdout_months"] == 12
    assert cfg["max_samples_method"] in {"avgU", "float"}
    assert isinstance(cfg["feature_prefixes"], list)
    assert cfg["t1_col"] == "exit_ts"
```

**Step 2: Run test to verify it fails**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_rf_primary_config.py -v`  
Expected: FAIL，缺少 `RF_PRIMARY_CONFIG` 且 `PRIMARY_MODEL_TYPE` 尚未声明支持 `rf`。

**Step 3: Write minimal implementation**

```python
# PRIMARY_MODEL_TYPE: 'ma' | 'cusum_direction' | 'rf'
PRIMARY_MODEL_TYPE = 'cusum_direction'

RF_PRIMARY_CONFIG = {
    'n_estimators': 1000,
    'n_jobs': -1,
    'random_state': 42,
    'cv_n_splits': 5,
    'cv_embargo_pct': 0.01,
    'holdout_months': 12,
    'feature_prefixes': [
        'feat_rsi_', 'feat_roc_', 'feat_stoch_', 'feat_adx_',
        'feat_vwap_', 'feat_cross_ma_', 'feat_shannon_',
        'feat_lz_entropy_', 'feat_hl_vol_',
    ],
    'min_t_value': 0.0,
    'max_samples_method': 'avgU',
    't1_col': 'exit_ts',
}
```

**Step 4: Run test to verify it passes**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_rf_primary_config.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/strategies/test_al9999_rf_primary_config.py strategies/AL9999/config.py
git commit -m "feat(al9999): add RF primary model config contract"
```

---

### Task 2: 先测后写 RF 数据准备与特征筛选工具

**Files:**
- Create: `tests/strategies/test_al9999_rf_primary_data_prep.py`
- Create: `strategies/AL9999/04_rf_primary_model.py`

**Step 1: Write the failing test**

```python
import pandas as pd

import importlib

rf_module = importlib.import_module("strategies.AL9999.04_rf_primary_model")


def test_select_feature_columns_by_prefix():
    cols = [
        "feat_rsi_14",
        "feat_roc_5",
        "feat_microstructure_20",
        "raw_value",
    ]
    selected = rf_module.select_feature_columns(
        cols,
        ["feat_rsi_", "feat_roc_"],
    )
    assert selected == ["feat_rsi_14", "feat_roc_5"]


def test_prepare_rf_dataset_filters_side_zero_and_low_t_value():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    features = pd.DataFrame(
        {
            "feat_rsi_14": [1.0, 2.0, 3.0, 4.0],
            "feat_roc_5": [0.1, 0.2, 0.3, 0.4],
            "feat_microstructure_20": [5, 6, 7, 8],
        },
        index=idx,
    )
    labels = pd.DataFrame(
        {
            "side": [1, 0, -1, 1],
            "t_value": [2.0, 3.0, 0.2, -1.5],
            "t1": idx + pd.Timedelta(days=2),
        },
        index=idx,
    )

    X, y, t1, meta = rf_module.prepare_rf_dataset(
        features,
        labels,
        feature_prefixes=["feat_rsi_", "feat_roc_"],
        min_t_value=1.0,
    )

    assert list(X.columns) == ["feat_rsi_14", "feat_roc_5"]
    assert list(y.tolist()) == [1, 1]
    assert set(meta["trend_side"].tolist()) == {1}
    assert (meta["abs_t_value"] >= 1.0).all()
```

**Step 2: Run test to verify it fails**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_rf_primary_data_prep.py -v`  
Expected: FAIL，模块或函数不存在。

**Step 3: Write minimal implementation**

```python
def select_feature_columns(columns, prefixes):
    if not prefixes:
        return [c for c in columns if c.startswith("feat_")]
    selected = []
    for col in columns:
        if any(col.startswith(prefix) for prefix in prefixes):
            selected.append(col)
    return selected


def prepare_rf_dataset(features, labels, feature_prefixes, min_t_value):
    common_idx = features.index.intersection(labels.index)
    merged = features.loc[common_idx].copy()
    merged["trend_side"] = labels.loc[common_idx, "side"]
    merged["t_value"] = labels.loc[common_idx, "t_value"]
    merged["t1"] = pd.to_datetime(labels.loc[common_idx, "t1"], errors="coerce")

    merged = merged[merged["trend_side"] != 0].copy()
    merged["abs_t_value"] = merged["t_value"].abs()
    if min_t_value > 0:
        merged = merged[merged["abs_t_value"] >= min_t_value].copy()

    feature_cols = select_feature_columns(merged.columns.tolist(), feature_prefixes)
    X = merged[feature_cols].fillna(0.0)
    y = merged["trend_side"].astype(int)
    t1 = merged["t1"]
    meta = merged[["trend_side", "abs_t_value"]].copy()
    return X, y, t1, meta
```

**Step 4: Run test to verify it passes**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_rf_primary_data_prep.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/strategies/test_al9999_rf_primary_data_prep.py strategies/AL9999/04_rf_primary_model.py
git commit -m "feat(al9999): add RF primary dataset preparation helpers"
```

---

### Task 3: 锁定 avgU 抽样与 Holdout 切分行为

**Files:**
- Modify: `strategies/AL9999/04_rf_primary_model.py`
- Create: `tests/strategies/test_al9999_rf_primary_sampling.py`

**Step 1: Write the failing test**

```python
import importlib
import math
import pandas as pd

rf_module = importlib.import_module("strategies.AL9999.04_rf_primary_model")


def test_compute_avg_uniqueness_from_t1_returns_fraction():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    t1 = pd.Series(
        [
            idx[1],
            idx[2],
            idx[3],
            idx[4],
            idx[4],
        ],
        index=idx,
    )

    avg_u = rf_module.compute_avg_uniqueness_from_t1(idx, t1)
    assert 0.0 < avg_u <= 1.0
    assert math.isfinite(avg_u)


def test_split_train_holdout_keeps_last_months_as_holdout():
    idx = pd.date_range("2024-01-01", periods=16, freq="30D")
    X = pd.DataFrame({"feat_rsi_14": range(16)}, index=idx)
    y = pd.Series([1, -1] * 8, index=idx)
    t1 = pd.Series(idx + pd.Timedelta(days=5), index=idx)

    train_pack, holdout_pack, holdout_start = rf_module.split_train_holdout(
        X, y, t1, holdout_months=6
    )

    X_train, _, _ = train_pack
    X_holdout, _, _ = holdout_pack
    assert len(X_train) > 0
    assert len(X_holdout) > 0
    assert X_train.index.max() < holdout_start
    assert X_holdout.index.min() >= holdout_start
```

**Step 2: Run test to verify it fails**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_rf_primary_sampling.py -v`  
Expected: FAIL，`compute_avg_uniqueness_from_t1` / `split_train_holdout` 尚不存在。

**Step 3: Write minimal implementation**

```python
from afmlkit.label.weights import average_uniqueness


def compute_avg_uniqueness_from_t1(index, t1):
    timestamps = index.view("int64")
    event_pos = pd.Index(index).get_indexer(index)
    touch_pos = pd.Index(index).get_indexer(pd.DatetimeIndex(t1).fillna(index[-1]), method="backfill")
    touch_pos = np.maximum(touch_pos, event_pos)
    uniq, _ = average_uniqueness(
        timestamps.astype(np.int64),
        event_pos.astype(np.int64),
        touch_pos.astype(np.int64),
    )
    avg_u = float(np.nanmean(uniq))
    if not np.isfinite(avg_u) or avg_u <= 0:
        return 0.5
    return min(avg_u, 1.0)


def split_train_holdout(X, y, t1, holdout_months):
    holdout_start = X.index.max() - pd.DateOffset(months=holdout_months)
    train_mask = X.index < holdout_start
    holdout_mask = X.index >= holdout_start
    return (
        X.loc[train_mask],
        y.loc[train_mask],
        t1.loc[train_mask],
    ), (
        X.loc[holdout_mask],
        y.loc[holdout_mask],
        t1.loc[holdout_mask],
    ), holdout_start
```

**Step 4: Run test to verify it passes**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_rf_primary_sampling.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/strategies/test_al9999_rf_primary_sampling.py strategies/AL9999/04_rf_primary_model.py
git commit -m "feat(al9999): add avgU sampling and holdout split helpers"
```

---

### Task 4: 先测后写 RF 训练脚本主流程与产物契约

**Files:**
- Modify: `strategies/AL9999/04_rf_primary_model.py`
- Create: `tests/strategies/test_al9999_rf_primary_pipeline.py`

**Step 1: Write the failing test**

```python
import importlib
import pandas as pd

rf_module = importlib.import_module("strategies.AL9999.04_rf_primary_model")


def test_build_signal_frame_contains_required_columns():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    result = rf_module.build_signal_frame(
        index=idx,
        trend_side=pd.Series([1, -1, 1], index=idx),
        probs=pd.Series([0.8, 0.3, 0.55], index=idx),
        t1=pd.Series(idx + pd.Timedelta(days=2), index=idx),
        is_holdout=pd.Series([False, False, True], index=idx),
    )

    assert list(result.columns) == [
        "side", "y_prob", "y_pred", "trend_side", "t1", "is_holdout"
    ]
    assert set(result["side"].unique()) == {-1, 1}
    assert result["y_prob"].between(0, 1).all()


def test_build_model_uses_avg_u_for_bagging_max_samples():
    model = rf_module.build_model(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_samples=0.37,
    )
    assert model.n_estimators == 200
    assert model.max_samples == 0.37
```

**Step 2: Run test to verify it fails**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_rf_primary_pipeline.py -v`  
Expected: FAIL，`build_signal_frame` / `build_model` 尚不存在。

**Step 3: Write minimal implementation**

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def build_model(n_estimators, random_state, n_jobs, max_samples):
    base_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,
        min_samples_leaf=5,
        random_state=random_state,
    )
    return BaggingClassifier(
        estimator=base_tree,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=1.0,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def build_signal_frame(index, trend_side, probs, t1, is_holdout):
    y_pred = (probs >= 0.5).astype(int)
    side = y_pred.replace({0: -1, 1: 1})
    return pd.DataFrame(
        {
            "side": side.astype(int),
            "y_prob": probs.astype(float),
            "y_pred": y_pred.astype(int),
            "trend_side": trend_side.astype(int),
            "t1": pd.to_datetime(t1),
            "is_holdout": is_holdout.astype(bool),
        },
        index=index,
    )
```

在同一任务中补全主流程 `main()`，但先不追求全量端到端测试，先做到：
- 读取 `events_features.parquet` 和 `trend_labels.parquet`
- 构建训练集 / Holdout
- 用 `PurgedKFold` 产出 OOF 概率
- 保存：
  - `strategies/AL9999/output/features/rf_primary_signals.parquet`
  - `strategies/AL9999/output/models/rf_primary.pkl`
  - `strategies/AL9999/output/features/rf_primary_cv_report.parquet`

**Step 4: Run test to verify it passes**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_rf_primary_pipeline.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/strategies/test_al9999_rf_primary_pipeline.py strategies/AL9999/04_rf_primary_model.py
git commit -m "feat(al9999): add RF primary training pipeline contract"
```

---

### Task 5: 将 `rf` 接入现有 Primary Model + TBM 流程

**Files:**
- Modify: `strategies/AL9999/04_ma_primary_model.py`
- Modify: `strategies/AL9999/run_workflow.py`
- Create: `tests/strategies/test_al9999_rf_primary_integration.py`

**Step 1: Write the failing test**

```python
import importlib
import pandas as pd
import pytest

primary_module = importlib.import_module("strategies.AL9999.04_ma_primary_model")


def test_generate_rf_signals_reads_rf_signal_file(tmp_path):
    bars_idx = pd.date_range("2024-01-01", periods=5, freq="D")
    bars = pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=bars_idx)
    events = pd.DataFrame({"price": [1.1, 2.2, 4.4]}, index=bars_idx[[0, 1, 3]])
    rf_df = pd.DataFrame(
        {
            "side": [1, -1, 1],
            "y_prob": [0.7, 0.2, 0.8],
        },
        index=events.index,
    )
    path = tmp_path / "rf_primary_signals.parquet"
    rf_df.to_parquet(path)

    signals = primary_module.generate_rf_signals(bars, events, str(path))

    assert list(signals["side"]) == [1, -1, 1]
    assert list(signals["idx"]) == [0, 1, 3]


def test_generate_rf_signals_raises_when_file_missing(tmp_path):
    bars = pd.DataFrame({"close": [1.0]}, index=pd.date_range("2024-01-01", periods=1))
    events = pd.DataFrame({"price": [1.0]}, index=bars.index)
    with pytest.raises(FileNotFoundError):
        primary_module.generate_rf_signals(bars, events, str(tmp_path / "missing.parquet"))
```

**Step 2: Run test to verify it fails**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_rf_primary_integration.py -v`  
Expected: FAIL，`generate_rf_signals` 尚不存在。

**Step 3: Write minimal implementation**

```python
def generate_rf_signals(bars, events, rf_signals_path):
    if not os.path.exists(rf_signals_path):
        raise FileNotFoundError(rf_signals_path)

    rf_df = pd.read_parquet(rf_signals_path)
    common_idx = events.index.intersection(rf_df.index)
    signals = events.loc[common_idx].copy()
    signals["side"] = rf_df.loc[common_idx, "side"].astype(int).values
    signals["idx"] = [bars.index.get_loc(ts) for ts in signals.index]
    if "y_prob" in rf_df.columns:
        signals["rf_prob"] = rf_df.loc[common_idx, "y_prob"].astype(float).values
    return signals
```

同时修改 `main()` 分发逻辑：

```python
if primary_model_type == "rf":
    signals = generate_rf_signals(
        bars,
        events,
        os.path.join(FEATURES_DIR, "rf_primary_signals.parquet"),
    )
elif primary_model_type == "cusum_direction":
    ...
else:
    ...
```

`run_workflow.py` 最小改动：
- 保持 Phase 4 入口仍为 `04_ma_primary_model.py`
- 在阶段描述中明确当 `PRIMARY_MODEL_TYPE='rf'` 时，必须先执行 `04_rf_primary_model.py`
- 可选地给 `--phase 4` 增加前置日志，而不是隐式自动执行多个脚本

**Step 4: Run test to verify it passes**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_rf_primary_integration.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/strategies/test_al9999_rf_primary_integration.py strategies/AL9999/04_ma_primary_model.py strategies/AL9999/run_workflow.py
git commit -m "feat(al9999): integrate RF primary signals into TBM workflow"
```

---

### Task 6: 将 RF 概率注入 Meta Model 特征集

**Files:**
- Modify: `strategies/AL9999/07_meta_model.py`
- Create: `tests/strategies/test_al9999_meta_model_rf_prob.py`

**Step 1: Write the failing test**

```python
import pandas as pd

from strategies.AL9999 import config
from strategies.AL9999 import _07_meta_model_testable as meta_loader


def test_merge_rf_probability_adds_rf_prob_feature(tmp_path, monkeypatch):
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    features = pd.DataFrame({"feat_rsi_14": [1.0, 2.0, 3.0]}, index=idx)
    labels = pd.DataFrame({"bin": [1, 0, 1], "sample_weight": [1.0, 1.0, 1.0]}, index=idx)
    tbm = pd.DataFrame({"exit_ts": idx + pd.Timedelta(days=1)}, index=idx)
    rf = pd.DataFrame({"y_prob": [0.7, 0.4, 0.8]}, index=idx)

    merged = meta_loader.merge_rf_probability(features, rf)
    assert "rf_prob" in merged.columns
    assert merged["rf_prob"].tolist() == [0.7, 0.4, 0.8]
```

**Step 2: Run test to verify it fails**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_meta_model_rf_prob.py -v`  
Expected: FAIL，缺少 RF 概率注入逻辑。

**Step 3: Write minimal implementation**

不要真的创建 `_07_meta_model_testable.py`。做法是把可测逻辑抽成 `07_meta_model.py` 内的纯函数，并在测试中用 `importlib.import_module("strategies.AL9999.07_meta_model")` 导入。

```python
def merge_rf_probability(features: pd.DataFrame, rf_signals: pd.DataFrame) -> pd.DataFrame:
    merged = features.copy()
    merged["rf_prob"] = rf_signals.reindex(merged.index)["y_prob"].astype(float)
    merged["rf_prob"] = merged["rf_prob"].fillna(0.5)
    return merged
```

在 `load_data()` 中追加：

```python
rf_path = os.path.join(FEATURES_DIR, "rf_primary_signals.parquet")
if os.path.exists(rf_path):
    rf_signals = pd.read_parquet(rf_path)
    X = merge_rf_probability(X, rf_signals)
else:
    X["rf_prob"] = 0.5
```

确保最终 `feature_cols` 仍包含 `rf_prob`。

**Step 4: Run test to verify it passes**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_meta_model_rf_prob.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/strategies/test_al9999_meta_model_rf_prob.py strategies/AL9999/07_meta_model.py
git commit -m "feat(al9999): add RF probability as meta model feature"
```

---

### Task 7: 端到端冒烟验证 RF Primary 流程

**Files:**
- Modify: `tests/strategies/test_al9999_live.py`
- Test: `tests/strategies/test_al9999_live.py`

**Step 1: Write the failing test**

```python
import importlib
import pandas as pd


def test_al9999_rf_primary_smoke_pipeline(tmp_path, monkeypatch):
    rf_module = importlib.import_module("strategies.AL9999.04_rf_primary_model")
    primary_module = importlib.import_module("strategies.AL9999.04_ma_primary_model")

    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    features = pd.DataFrame(
        {
            "feat_rsi_14": range(8),
            "feat_roc_5": range(10, 18),
            "feat_ewm_vol_20": [0.01] * 8,
        },
        index=idx,
    )
    labels = pd.DataFrame(
        {
            "side": [1, -1, 1, -1, 1, -1, 1, -1],
            "t_value": [2.0] * 8,
            "t1": idx + pd.Timedelta(days=2),
        },
        index=idx,
    )
    bars = pd.DataFrame({"close": range(100, 108)}, index=idx)
    events = pd.DataFrame({"price": range(100, 108)}, index=idx)

    # 通过 monkeypatch 将读写落到 tmp_path，验证 rf_signals 可被 TBM 前置流程消费
    assert True
```

把占位断言替换成至少这些具体断言：
- `rf_primary_signals.parquet` 被生成
- `side` 只包含 `{-1, 1}`
- `generate_rf_signals()` 能读回生成结果并产出 `idx`

**Step 2: Run test to verify it fails**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_live.py -k rf_primary_smoke_pipeline -v`  
Expected: FAIL，因为端到端冒烟路径尚未建立。

**Step 3: Write minimal implementation**

补齐 `04_rf_primary_model.py` 中这些缺口：
- `main()` 支持从真实配置路径读写
- 产物目录自动创建
- Holdout 太小、单类标签、缺失 `t1` 等场景给出清晰报错或回退
- 输出图表和 parquet 时保证列名稳定

如端到端测试太重，可把“训练并保存 + 读取并对接”拆成两个小测试，但不要跳过产物契约验证。

**Step 4: Run test to verify it passes**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_live.py -k rf_primary_smoke_pipeline -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/strategies/test_al9999_live.py strategies/AL9999/04_rf_primary_model.py
git commit -m "test(al9999): add RF primary smoke pipeline coverage"
```

---

### Task 8: 完整验证与手工研究回放

**Files:**
- Modify: `strategies/AL9999/AFML_SOP.md`
- Test: `tests/strategies/test_al9999_rf_primary_config.py`
- Test: `tests/strategies/test_al9999_rf_primary_data_prep.py`
- Test: `tests/strategies/test_al9999_rf_primary_sampling.py`
- Test: `tests/strategies/test_al9999_rf_primary_pipeline.py`
- Test: `tests/strategies/test_al9999_rf_primary_integration.py`
- Test: `tests/strategies/test_al9999_meta_model_rf_prob.py`

**Step 1: Write the failing doc/test delta**

在文档里补一节“RF Primary Model 运行顺序”，明确：
- `python strategies/AL9999/04_rf_primary_model.py`
- 将 `PRIMARY_MODEL_TYPE = 'rf'`
- 再运行 `python strategies/AL9999/04_ma_primary_model.py`
- 然后 `python strategies/AL9999/07_meta_model.py`

如果当前文档无此章节，视为“缺失即失败”。

**Step 2: Run focused test suite**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_rf_primary_config.py tests/strategies/test_al9999_rf_primary_data_prep.py tests/strategies/test_al9999_rf_primary_sampling.py tests/strategies/test_al9999_rf_primary_pipeline.py tests/strategies/test_al9999_rf_primary_integration.py tests/strategies/test_al9999_meta_model_rf_prob.py -v`  
Expected: PASS

**Step 3: Run broader AL9999 regression checks**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_filter_first_pipeline.py tests/strategies/test_al9999_execution_guard.py tests/strategies/test_al9999_threshold_optimizer.py -v`  
Expected: PASS，证明 RF 接入未破坏现有研究工作流关键路径。

**Step 4: Run manual workflow smoke commands**

Run: `python strategies/AL9999/04_rf_primary_model.py`  
Expected: 生成：
- `strategies/AL9999/output/features/rf_primary_signals.parquet`
- `strategies/AL9999/output/models/rf_primary.pkl`
- `strategies/AL9999/output/features/rf_primary_cv_report.parquet`

Run: `python strategies/AL9999/04_ma_primary_model.py`  
Expected: 当 `PRIMARY_MODEL_TYPE='rf'` 时可成功消费 RF side，生成 `tbm_results.parquet`。

Run: `python strategies/AL9999/07_meta_model.py`  
Expected: 特征集中包含 `rf_prob`，训练完成且输出 OOF / holdout 结果。

**Step 5: Commit**

```bash
git add strategies/AL9999/AFML_SOP.md tests/strategies/test_al9999_rf_primary_config.py tests/strategies/test_al9999_rf_primary_data_prep.py tests/strategies/test_al9999_rf_primary_sampling.py tests/strategies/test_al9999_rf_primary_pipeline.py tests/strategies/test_al9999_rf_primary_integration.py tests/strategies/test_al9999_meta_model_rf_prob.py strategies/AL9999/04_rf_primary_model.py strategies/AL9999/04_ma_primary_model.py strategies/AL9999/07_meta_model.py strategies/AL9999/config.py strategies/AL9999/run_workflow.py
git commit -m "feat(al9999): add RF primary model workflow"
```

---

## 实现提示

- `04_rf_primary_model.py` 顶层只保留流程编排，纯函数尽量拆为：
  - `select_feature_columns`
  - `prepare_rf_dataset`
  - `split_train_holdout`
  - `compute_avg_uniqueness_from_t1`
  - `build_model`
  - `build_signal_frame`
- `BaggingClassifier` 的 `estimator` 先用 `DecisionTreeClassifier(max_features=1)`，这是当前仓库 Meta Model 已采用、且更贴近 AFML “去遮蔽”实践的最小安全选择；不要在第一版计划里强行套一层 `RandomForestClassifier(n_estimators=1)`，避免引入不必要复杂度。
- `y_pred` 保留 `0/1`，`side` 统一转换到 `{-1, 1}`，这样既方便 sklearn 度量，又满足现有 Primary/TBM 接口。
- 如果 `feature_prefixes` 匹配为空，打印 warning 并回退到全部 `feat_` 列，不要直接崩溃。
- 如果训练集只剩单类，优先抛出带样本统计的 `ValueError`，不要默默生成伪信号。

## 完成定义

- `strategies/AL9999/04_rf_primary_model.py` 可独立运行并产出模型、信号、CV 报告。
- `strategies/AL9999/04_ma_primary_model.py` 支持 `PRIMARY_MODEL_TYPE='rf'`。
- `strategies/AL9999/07_meta_model.py` 可消费 `rf_prob`。
- 所有新增测试通过，且现有关键 `AL9999` 回测测试未回归。


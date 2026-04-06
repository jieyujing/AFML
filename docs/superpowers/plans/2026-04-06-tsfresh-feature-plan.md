# AL9999 tsfresh 特征工程实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 AL9999 新增独立的 tsfresh 特征提取脚本，在 CUSUM 事件点上提取基于滚动窗口的 tsfresh 特征。

**Architecture:** 三层架构：输入层(4列) → 变换层(4种) → 提取层(tsfresh 16函数)。事件切片机制：每个 CUSUM 事件点回看 N bars 提取特征。

**Tech Stack:** tsfresh, pandas, numpy, afmlkit.feature.core.frac_diff

---

## 文件变更映射

| 文件 | 操作 | 职责 |
|------|------|------|
| `pyproject.toml` | Modify | 添加 `tsfresh` 依赖 |
| `strategies/AL9999/config.py` | Modify | 添加 `TSFRESH_CONFIG` 配置块 |
| `strategies/AL9999/02b_tsfresh_feature_engineering.py` | Create | 主脚本：变换+提取+输出 |
| `tests/strategies/test_al9999_tsfresh_feature_engineering.py` | Create | 单元测试 |

---

## Task 1: 添加 tsfresh 依赖

- Modify: `pyproject.toml`

- [ ] **Step 1: 添加 tsfresh 到 pyproject.toml**

找到 `[project.dependencies]` 段，在末尾添加 `"tsfresh~=0.20.0"`，注意与现有依赖格式一致（波浪号版本约束）。

Run: `grep -n "scipy" pyproject.toml` 确认添加位置。

```toml
dependencies = [
    # ... existing deps ...
    "numpy~=2.2.0",
    "scipy>=1.10.0,<2.0",
    # ...
    "tsfresh~=0.20.0",  # 新增
]
```

- [ ] **Step 2: 安装依赖**

```bash
uv sync
```

Expected: tsfresh 安装成功，无冲突。

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add tsfresh dependency"
```

---

## Task 2: 添加 TSFRESH_CONFIG 到 config.py

- Modify: `strategies/AL9999/config.py`

- [ ] **Step 1: 找到插入位置**

Run: `grep -n "# Feature Engineering" strategies/AL9999/config.py` 确认插入点在 `FEATURE_CONFIG` 之前。

- [ ] **Step 2: 添加配置块**

在 `FEATURE_CONFIG` 定义之前插入：

```python
# ============================================================
# Phase 2b: tsfresh 特征参数
# ============================================================

TSFRESH_CONFIG = {
    "enabled": True,
    "lookback": 20,                           # 回看 bars 数量
    "fracdiff_cols": ["close", "log_close"],  # 只对这些列做 fracdiff
    "zscore_windows": [10, 20, 40],
    "features": [
        "mean",
        "median",
        "standard_deviation",
        "skewness",
        "kurtosis",
        "minimum",
        "maximum",
        "abs_energy",
        "mean_change",
        "mean_abs_change",
        "count_above_mean",
        "count_below_mean",
        "first_location_of_maximum",
        "first_location_of_minimum",
        "last_location_of_maximum",
        "last_location_of_minimum",
    ],
}
```

- [ ] **Step 3: Commit**

```bash
git add strategies/AL9999/config.py
git commit -m "feat(AL9999): add TSFRESH_CONFIG"
```

---

## Task 3: 实现主脚本 02b_tsfresh_feature_engineering.py

- Create: `strategies/AL9999/02b_tsfresh_feature_engineering.py`

- [ ] **Step 1: 写框架和导入**

```python
"""
02b_tsfresh_feature_engineering.py - AL9999 Phase 2b tsfresh 特征工程

流程:
1. 加载 Dollar Bars + CUSUM 事件点
2. 对每个事件点回看 lookback bars，构建时间片 DataFrame
3. 应用 4 种变换（raw, pct_change, fracdiff, zscore）
4. 用 tsfresh 提取 16 个特征函数
5. 输出到 features/tsfresh_features.parquet
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tsfresh import extract_features
from tsfresh.utilities.dataframe_utils import make_forecasting_frame

from strategies.AL9999.config import (
    BARS_DIR, FEATURES_DIR, FEATURES_DIR,
    FRACDIFF_THRES, FRACDIFF_D_STEP, FRACDIFF_MAX_D,
    TARGET_DAILY_BARS, TSFRESH_CONFIG,
)
from afmlkit.feature.core.frac_diff import optimize_d, frac_diff_ffd
```

- [ ] **Step 2: 实现变换函数**

在文件顶部（导入之后）定义 4 个变换函数：

```python
def apply_raw(series: pd.Series) -> pd.Series:
    """原始序列，无变换。"""
    return series


def apply_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """百分比变化率。"""
    return series.pct_change(periods=periods)


def apply_fracdiff(series: pd.Series, thres: float = 1e-4) -> pd.Series:
    """分数差分 FFD。"""
    # 用前 80% 数据优化 d
    n_train = int(len(series) * 0.8)
    train_series = series.iloc[:n_train].dropna()
    if len(train_series) < 20:
        return pd.Series(np.nan, index=series.index)
    optimal_d = optimize_d(train_series, thres=thres, d_step=0.05, max_d=1.0, min_corr=0.0)
    fd_series = frac_diff_ffd(series, d=optimal_d, thres=thres)
    return fd_series


def apply_zscore(series: pd.Series, window: int) -> pd.Series:
    """滚动窗口标准化。"""
    rolling_mean = series.rolling(window, min_periods=1).mean()
    rolling_std = series.rolling(window, min_periods=1).std()
    zscored = (series - rolling_mean) / rolling_std
    return zscored
```

- [ ] **Step 3: 实现特征名构建函数**

```python
def build_feature_name(col: str, transform: str, func_name: str, window: int = None) -> str:
    """构建 tsfresh 特征命名。"""
    if transform == "zscore":
        return f"feat_{col}_z{window}_{func_name}"
    elif transform == "pct_change":
        return f"feat_{col}_pct_{func_name}"
    elif transform == "fracdiff":
        return f"feat_{col}_fd_{func_name}"
    else:  # raw
        return f"feat_{col}_raw_{func_name}"
```

- [ ] **Step 4: 实现单事件点特征提取**

```python
def extract_tsfresh_for_slice(
    slice_df: pd.DataFrame,
    col: str,
    transform: str,
    func_names: List[str],
    zscore_window: int = None
) -> Dict[str, float]:
    """对一个时间片的一列提取 tsfresh 特征。"""
    from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
    from tsfresh.feature_extraction.settings import TimeBasedFCParameters

    # 取原始序列
    raw_series = slice_df[col].values

    # 应用变换
    if transform == "raw":
        transformed = raw_series
    elif transform == "pct_change":
        transformed = np.diff(raw_series) / raw_series[:-1]
        # pct_change 长度少1，前补 NaN
        transformed = np.concatenate([[np.nan], transformed])
    elif transform == "fracdiff":
        series_pd = pd.Series(raw_series)
        fd_result = apply_fracdiff(series_pd)
        transformed = fd_result.values
    elif transform == "zscore":
        transformed = apply_zscore(pd.Series(raw_series), zscore_window).values
    else:
        raise ValueError(f"Unknown transform: {transform}")

    # 构建 tsfresh 输入 DataFrame（id, time, value）
    n = len(transformed)
    ids = np.zeros(n, dtype=np.int64)
    times = np.arange(n, dtype=np.int64)
    df_input = pd.DataFrame({
        "id": ids,
        "time": times,
        "value": transformed,
    })

    # 提取特征
    settings = MinimalFCParameters()
    # 只保留需要的特征函数
    from tsfresh.feature_extraction import FeatureExtractionSettings
    settings = FeatureExtractionSettings(time_dependency_solver=TimeBasedFCParameters())
    # 使用 EfficientFCParameters 作为平衡
    features = extract_features(
        df_input,
        column_id="id",
        column_time="time",
        column_value="value",
        feature_extraction_settings=settings,
        kind_to_fc_parameters=EfficientFCParameters(),
    )

    results = {}
    for func_name in func_names:
        col_name = build_feature_name(col, transform, func_name, zscore_window)
        if func_name in features.columns:
            val = features[func_name].iloc[0]
            results[col_name] = val if not (isinstance(val, float) and np.isnan(val)) else np.nan
        else:
            results[col_name] = np.nan

    return results
```

- [ ] **Step 5: 实现主提取流程**

```python
def extract_tsfresh_features(
    bars: pd.DataFrame,
    event_indices: np.ndarray,
    config: Dict
) -> pd.DataFrame:
    """对所有事件点提取 tsfresh 特征。"""
    lookback = config["lookback"]
    fracdiff_cols = config["fracdiff_cols"]
    zscore_windows = config["zscore_windows"]
    func_names = config["features"]

    results = []
    total = len(event_indices)

    for i, event_idx in enumerate(event_indices):
        if i % 500 == 0:
            print(f"  [{i}/{total}] Processing event {i}")

        # 边界检查
        start_idx = max(0, event_idx - lookback)
        end_idx = event_idx + 1
        slice_df = bars.iloc[start_idx:end_idx].copy()

        event_result = {"event_idx": event_idx}
        event_result["timestamp"] = bars.index[event_idx]

        cols = ["close", "log_close", "volume", "open_interest"]

        # 1. raw
        for col in cols:
            feats = extract_tsfresh_for_slice(slice_df, col, "raw", func_names)
            event_result.update(feats)

        # 2. pct_change
        for col in cols:
            feats = extract_tsfresh_for_slice(slice_df, col, "pct_change", func_names)
            event_result.update(feats)

        # 3. fracdiff（仅限 fracdiff_cols）
        for col in fracdiff_cols:
            feats = extract_tsfresh_for_slice(slice_df, col, "fracdiff", func_names)
            event_result.update(feats)

        # 4. zscore（多窗口）
        for window in zscore_windows:
            for col in cols:
                feats = extract_tsfresh_for_slice(slice_df, col, "zscore", func_names, zscore_window=window)
                event_result.update(feats)

        results.append(event_result)

    return pd.DataFrame(results)
```

- [ ] **Step 6: 实现 main 函数**

```python
def main():
    """AL9999 Phase 2b tsfresh 特征工程主流程。"""
    print("=" * 70)
    print("  AL9999 Phase 2b tsfresh Feature Engineering")
    print("=" * 70)

    os.makedirs(FEATURES_DIR, exist_ok=True)

    # Step 1: 加载 Dollar Bars
    print("\n[Step 1] 加载 Dollar Bars...")
    bars_path = os.path.join(BARS_DIR, f'dollar_bars_target{TARGET_DAILY_BARS}.parquet')
    bars = pd.read_parquet(bars_path)
    print(f"  ✅ 加载 Dollar Bars: {len(bars)} bars")

    # 准备 log_close
    bars["log_close"] = np.log(bars["close"])

    # Step 2: 加载 CUSUM 事件点
    print("\n[Step 2] 加载 CUSUM 事件点...")
    events_path = os.path.join(FEATURES_DIR, "cusum_events.parquet")
    events_df = pd.read_parquet(events_path)
    print(f"  ✅ 加载 CUSUM 事件: {len(events_df)} 个事件")

    # 获取事件索引（相对于 bars DataFrame）
    event_timestamps = events_df["timestamp"]
    event_indices = np.array([
        bars.index.get_loc(ts) for ts in event_timestamps
    ])

    # Step 3: 提取 tsfresh 特征
    print("\n[Step 3] 提取 tsfresh 特征...")
    tsfresh_df = extract_tsfresh_features(bars, event_indices, TSFRESH_CONFIG)

    # Step 4: 保存
    print("\n[Step 4] 保存特征...")
    output_path = os.path.join(FEATURES_DIR, "tsfresh_features.parquet")
    tsfresh_df.to_parquet(output_path)
    print(f"  ✅ tsfresh 特征已保存: {output_path}")
    print(f"     事件数: {len(tsfresh_df)}, 特征数: {len(tsfresh_df.columns) - 2}")  # -2 for event_idx, timestamp

    print("\n" + "=" * 70)
    print("  Phase 2b tsfresh Feature Engineering 完成")
    print("=" * 70)
```

- [ ] **Step 7: 添加 if __name__ == "__main__": main()**

- [ ] **Step 8: 测试运行**

```bash
cd /Users/link/Documents/AFMLKIT
python strategies/AL9999/02b_tsfresh_feature_engineering.py
```

Expected: 成功生成 `output/features/tsfresh_features.parquet`，无报错。

- [ ] **Step 9: Commit**

```bash
git add strategies/AL9999/02b_tsfresh_feature_engineering.py
git commit -m "feat(AL9999): add tsfresh feature engineering module"
```

---

## Task 4: 编写单元测试

- Create: `tests/strategies/test_al9999_tsfresh_feature_engineering.py`

- [ ] **Step 1: 写基础测试**

```python
"""
test_al9999_tsfresh_feature_engineering.py
"""
import numpy as np
import pandas as pd
import pytest

# 测试变换函数
def test_apply_raw():
    from strategies.AL9999.02b_tsfresh_feature_engineering import apply_raw
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = apply_raw(s)
    np.testing.assert_array_equal(result.values, s.values)


def test_apply_pct_change():
    from strategies.AL9999.02b_tsfresh_feature_engineering import apply_pct_change
    s = pd.Series([100.0, 101.0, 103.0, 102.0, 105.0])
    result = apply_pct_change(s)
    expected = pd.Series([np.nan, 0.01, 0.0198, -0.0097, 0.0294])
    np.testing.assert_allclose(result.dropna().values, expected.dropna().values, rtol=1e-3)


def test_apply_zscore():
    from strategies.AL9999.02b_tsfresh_feature_engineering import apply_zscore
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = apply_zscore(s, window=3)
    # rolling mean of [1,2,3]=2, rolling std>0, 结果应标准化
    assert result.iloc[2] == pytest.approx(0.0, abs=1e-10)


def test_build_feature_name():
    from strategies.AL9999.02b_tsfresh_feature_engineering import build_feature_name

    assert build_feature_name("close", "raw", "mean") == "feat_close_raw_mean"
    assert build_feature_name("volume", "pct_change", "skewness") == "feat_volume_pct_skewness"
    assert build_feature_name("close", "fracdiff", "kurtosis") == "feat_close_fd_kurtosis"
    assert build_feature_name("close", "zscore", "mean", window=20) == "feat_close_z20_mean"
```

- [ ] **Step 2: 运行测试验证**

```bash
cd /Users/link/Documents/AFMLKIT
NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_tsfresh_feature_engineering.py -v
```

Expected: 所有测试 PASS。

- [ ] **Step 3: Commit**

```bash
git add tests/strategies/test_al9999_tsfresh_feature_engineering.py
git commit -m "test(AL9999): add tsfresh feature engineering tests"
```

---

## Task 5: Spec 自检

在提交前，快速扫描：

- [ ] 所有 TSD-ADD 依赖已添加到 pyproject.toml
- [ ] TSFRESH_CONFIG 包含 `lookback`, `fracdiff_cols`, `zscore_windows`, `features`
- [ ] 变换层 4 种（raw, pct_change, fracdiff, zscore）全部实现
- [ ] fracdiff 仅应用于 `fracdiff_cols`（close, log_close）
- [ ] zscore 有 3 个窗口 [10, 20, 40]
- [ ] 16 个 tsfresh 特征函数全部覆盖
- [ ] 输出命名为 `feat_{col}_{transform}_{func}` 和 `feat_{col}_z{window}_{func}`
- [ ] 输出到 `FEATURES_DIR/tsfresh_features.parquet`
- [ ] 测试覆盖变换函数和特征命名构建

---

## 执行方式选择

Plan complete and saved to `docs/superpowers/plans/`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

# Meta Model Inference API Spec

## Data Separation (CRITICAL)

| Table | Purpose | Contains meta_y? |
|---|---|---|
| meta_training_table.parquet | Training + evaluation | YES — NEVER use at inference |
| meta_inference_table.parquet | Live inference | NO — safe to use |

## Inference Usage

```python
from strategies.AL9999.unified_meta_model import (
    build_inference_output,
    select_threshold_by_precision,
)
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Load artifacts
cal_model = joblib.load("strategies/AL9999/output/meta_model/meta_calibrator_v1.pkl")
inf_df = pd.read_parquet("strategies/AL9999/output/meta_model/meta_inference_table.parquet")
oof_df = pd.read_parquet("strategies/AL9999/output/meta_model/meta_oof_signals.parquet")

# Select threshold on OOF
threshold, prec, n = select_threshold_by_precision(oof_df, target_precision=0.65)
print(f"Threshold={threshold:.2f} achieves precision={prec:.4f} on {n} events")

# Build inference
output = build_inference_output(inf_df, oof_df, threshold, cal_model)
output.to_parquet("strategies/AL9999/output/meta_model/meta_inference_output.parquet", index=False)
```

## Output Schema

| Field | Type | Description |
|---|---|---|
| event_time | datetime64[ns] | Event timestamp |
| event_id | string | Unique event identifier |
| chosen_combo_id | string | Selected candidate |
| side | int8 | +1 (long) or -1 (short) |
| p_meta | float32 | Calibrated meta probability |
| threshold | float32 | Decision threshold used |
| decision | string | "TAKE" or "SKIP" |
| model_version | string | "meta_model_v1" |

## Decision Rules

1. Deduplication: `per_event_max_p` — one row per event, highest probability wins
2. Threshold: selected by target precision (default 0.65) on OOS validation
3. Decision: TAKE if p_meta >= threshold, else SKIP

## Artifacts

| File | Purpose |
|---|---|
| meta_calibrator_v1.pkl | Trained + calibrated model for inference |
| meta_inference_table.parquet | Event-time observable features only |
| meta_oof_signals.parquet | OOF predictions for threshold selection |
| meta_inference_output.parquet | Final inference output |
| meta_oos_report.xlsx | Walk-forward CV performance report |
| meta_feature_importance.csv | Feature importance from final model |
| meta_splits.json | Fold configuration for reproducibility |

# API Reference: test_labeling_fixed_time_horizon.py

**Language**: Python

**Source**: `tests/test_labeling_fixed_time_horizon.py`

---

## Classes

### TestLabelingFixedTime

Tests regarding fixed time horizon labeling method.

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for the sample dollar bars data.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_basic(self)

Tests for basic case, constant threshold and no standardization, lag.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_dynamic_threshold(self)

Tests for when threshold is a pd.Series rather than a constant.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_with_standardization(self)

Test cases with standardization, with constant and dynamic threshold.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_resample(self)

Tests for when a resample period is used.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_exceptions_warnings(self)

Tests the exceptions and warning that can be raised.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




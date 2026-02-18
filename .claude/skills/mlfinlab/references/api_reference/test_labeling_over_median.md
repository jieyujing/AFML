# API Reference: test_labeling_over_median.py

**Language**: Python

**Source**: `tests/test_labeling_over_median.py`

---

## Classes

### TestLabelingOverMedian

Tests regarding labeling excess over median.

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for the sample data.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_basic(self)

Test basic case for a small set with manually inputted results, with numerical and categorical outputs, with
no resampling or forward looking labels.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_resample_period(self)

Test numerical and categorical with a resample period.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_forward(self)

Tests with lagged returns.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_nan(self)

Tests to check that NaN values in prices get ignored.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




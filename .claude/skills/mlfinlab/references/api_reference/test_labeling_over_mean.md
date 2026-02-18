# API Reference: test_labeling_over_mean.py

**Language**: Python

**Source**: `tests/test_labeling_over_mean.py`

---

## Classes

### TestLabelingOverMean

Tests regarding labeling excess over median.

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for the sample data.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_small_set(self)

Check for a small set with manually inputted results, with numerical and categorical outputs.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_large_set(self)

Checks a specific row for a large dataset, and ensures the last row is NaN.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_resample_period(self)

Test numerical and categorical labels with a resample period.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




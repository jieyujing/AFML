# API Reference: test_sample_weights.py

**Language**: Python

**Source**: `tests/test_sample_weights.py`

---

## Classes

### TestSampling

Test Triple barrier, meta-labeling, dropping rare labels, and daily volatility.

**Inherits from**: unittest.TestCase

#### Methods

##### setUp(self)

Set the file path for the sample dollar bars data and get triple barrier events

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_ret_attribution(self)

Assert that return attribution length equals triple barrier length, check particular values

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_time_decay_weights(self)

Assert that time decay weights length equals triple barrier length, check particular values

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_value_error_raise(self)

Test seq_bootstrap and ind_matrix functions for raising ValueError on nan values

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




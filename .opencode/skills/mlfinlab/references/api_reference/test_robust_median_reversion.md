# API Reference: test_robust_median_reversion.py

**Language**: Python

**Source**: `tests/test_robust_median_reversion.py`

---

## Classes

### TestRobustMedianReversion

Tests different functions of the Robust Median Reversion class.

**Inherits from**: TestCase

#### Methods

##### setUp(self)

Sets the file path for the tick data csv.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_rmr_solution(self)

Test the calculation of RMR with the original method.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_rmr_epsilon_error(self)

Tests ValueError if epsilon is greater than 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_rmr_n_iteration_error(self)

Tests ValueError if n_iteration is not an integer or less than 2.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_rmr_window_error(self)

Tests ValueError if window is not an integer or less than 2.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_rmr_break_solution(self)

Test the calculation of RMR with the break case in _calc_median.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_rmr_transform_non_mu(self)

Tests edge case for _transform non_mu edge case.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_rmr_norm2_0_mu(self)

Tests edge case for norm2 = 0 in _transform method.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_rmr_tau_error(self)

Tests ValueError if tau is less than 0 or greater than or equal to 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




# API Reference: test_functional_correlation_driven_nonparametric_learning_k.py

**Language**: Python

**Source**: `tests/test_functional_correlation_driven_nonparametric_learning_k.py`

---

## Classes

### TestFunctionalCorrelationDrivenNonparametricLearningK

Tests different functions of the Functional Correlation Driven Nonparametric Learning - K class.

**Inherits from**: TestCase

#### Methods

##### setUp(self)

Sets the file path for the tick data csv.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_fcorn_k_solution(self)

Test the calculation of FCORN-K.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_fcorn_k_window_error(self)

Tests ValueError if window is not an integer or less than 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_fcorn_k_rho_error(self)

Tests ValueError if rho is not an integer or less than 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_fcorn_k_lambd_error(self)

Tests ValueError if lambd is not an integer or less than 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_fcorn_k_k_error(self)

Tests ValueError if k is not an integer of greater than window * rho * lambd

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




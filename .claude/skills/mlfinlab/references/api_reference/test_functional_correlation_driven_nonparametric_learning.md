# API Reference: test_functional_correlation_driven_nonparametric_learning.py

**Language**: Python

**Source**: `tests/test_functional_correlation_driven_nonparametric_learning.py`

---

## Classes

### TestFunctionalCorrelationDrivenNonparametricLearning

Tests different functions of the Functional Correlation Driven Nonparametric Learning class.

**Inherits from**: TestCase

#### Methods

##### setUp(self)

Sets the file path for the tick data csv.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_fcorn_solution(self)

Test the calculation of FCORN.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_fcorn_window_error(self)

Tests ValueError if window is not an integer or less than 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_fcorn_rho_error(self)

Tests ValueError if rho is less than -1 or more than 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_sigmoid(self)

Tests Sigmoid Calculation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_fcorn_solution1(self)

Test the calculation of FCORN for edge case that activation function is 0.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




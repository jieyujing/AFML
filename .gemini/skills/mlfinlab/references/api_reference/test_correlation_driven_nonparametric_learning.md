# API Reference: test_correlation_driven_nonparametric_learning.py

**Language**: Python

**Source**: `tests/test_correlation_driven_nonparametric_learning.py`

---

## Classes

### TestCorrelationDrivenNonparametricLearning

Tests different functions of the Correlation Driven Nonparametric Learning class.

**Inherits from**: TestCase

#### Methods

##### setUp(self)

Sets the file path for the tick data csv.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_corn_solution(self)

Test the calculation of CORN.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_corn_window_error(self)

Tests ValueError if window is not an integer or less than 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_corn_rho_error(self)

Tests ValueError if rho is less than -1 or more than 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




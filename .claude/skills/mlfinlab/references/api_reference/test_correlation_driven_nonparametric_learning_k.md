# API Reference: test_correlation_driven_nonparametric_learning_k.py

**Language**: Python

**Source**: `tests/test_correlation_driven_nonparametric_learning_k.py`

---

## Classes

### TestCorrelationDrivenNonparametricLearningK

Tests different functions of the Correlation Driven Nonparametric Learning - K class.

**Inherits from**: TestCase

#### Methods

##### setUp(self)

Sets the file path for the tick data csv.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_corn_k_solution(self)

Test the calculation of CORN-K.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_corn_k_window_error(self)

Tests ValueError if window is not an integer or less than 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_corn_k_rho_error(self)

Tests ValueError if rho is not an integer or less than 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_corn_k_k_error(self)

Tests ValueError if k is greater than window * rho, greater than 1, or an integer.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




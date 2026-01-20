# API Reference: test_online_moving_average_reversion.py

**Language**: Python

**Source**: `tests/test_online_moving_average_reversion.py`

---

## Classes

### TestOnlineMovingAverageReversion

Tests different functions of the Online Moving Average Reversion class.

**Inherits from**: TestCase

#### Methods

##### setUp(self)

Sets the file path for the tick data csv.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_olmar_solution(self)

Test the calculation of online moving average reversion with the original reversion method.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_olmar1_solution(self)

Test the calculation of online moving average reversion with the second reversion method.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_olmar_epsilon_error(self)

Tests ValueError if epsilon is below than 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_olmar_window_error(self)

Tests ValueError if reversion method is 1 and window is less than 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_olmar_alpha_error(self)

Tests ValueError if reversion method is 2 and alpha is greater than 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_olmar_alpha1_error(self)

Tests ValueError if reversion method is 2 and alpha is less than 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_olmar_method_error(self)

Tests ValueError if reversion method is 2 and alpha is less than 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_olmar_edge_case_error(self)

Tests that lambd returns 0 if predicted change is mean change.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




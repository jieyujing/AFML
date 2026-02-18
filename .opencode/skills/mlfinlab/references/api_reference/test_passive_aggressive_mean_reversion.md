# API Reference: test_passive_aggressive_mean_reversion.py

**Language**: Python

**Source**: `tests/test_passive_aggressive_mean_reversion.py`

---

## Classes

### TestPassiveAggressiveMeanReversion

Tests different functions of the Passive Aggressive Mean Reversion class.

**Inherits from**: TestCase

#### Methods

##### setUp(self)

Set the file path for the tick data csv.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_pamr_solution(self)

Test the calculation of passive aggressive mean reversion with the original optimization
method.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_pamr1_solution(self)

Test the calculation of passive aggressive mean reversion with PAMR-1 optimization method.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_pamr2_solution(self)

Test the calculation of passive aggressive mean reversion with the PAMR-2 optimization method

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_pamr_epsilon_error(self)

Tests ValueError if epsilon is less than 0.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_pamr_agg_error(self)

Tests ValueError if aggressiveness is less than 0.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_pamr_method_error(self)

Tests ValueError if optimization method is not 0, 1, or 2.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




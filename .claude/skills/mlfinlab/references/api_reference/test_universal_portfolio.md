# API Reference: test_universal_portfolio.py

**Language**: Python

**Source**: `tests/test_universal_portfolio.py`

---

## Classes

### TestUniversalPortfolio

Tests different functions of the UP class.

**Inherits from**: TestCase

#### Methods

##### setUp(self)

Sets the file path for the tick data csv.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_up_solution(self)

Test the calculation of UP weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_up_progress_solution(self)

Tests that UP prints progress bar.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_up_uniform_solution(self)

Tests UP with uniform capital allocation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_up_top_k_solution(self)

Tests UP with top-k experts capital allocation.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_up_wrong_method(self)

Tests ValueError if the method is not 'hist_performance', 'uniform', or 'top-k'.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_up_recalculate_solution(self)

Tests recalculate method in UP.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_up_recalculate_error(self)

Tests ValueError if k is greater number of experts for recalculate.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_up_recalculate1_error(self)

Tests ValueError if k is not an integer for recalculate.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_up_recalculate2_error(self)

Tests ValueError if k is not greater than or equal to 1.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |




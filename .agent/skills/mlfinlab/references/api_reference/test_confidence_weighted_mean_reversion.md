# API Reference: test_confidence_weighted_mean_reversion.py

**Language**: Python

**Source**: `tests/test_confidence_weighted_mean_reversion.py`

---

## Classes

### TestConfidenceWeightedMeanReversion

Tests different functions of the Confidence Weighted Mean Reversion class.

**Inherits from**: TestCase

#### Methods

##### setUp(self)

Sets the file path for the tick data csv.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_cwmr_solution(self)

Test the calculation of CWMR with the original method.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_cwmr_sd_solution(self)

Test the calculation of CWMR with the second method.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_cwmr_epsilon_error(self)

Tests ValueError if epsilon is greater than 1 or less than 0.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_cwmr_confidence_error(self)

Tests ValueError if confidence is greater than 1 or less than 0.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_cwmr_method_error(self)

Tests ValueError if method is not 'sd' or 'var'.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### test_cwmr_weights_solution(self)

Test the calculation of CWMR with given weights.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |



